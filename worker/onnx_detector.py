"""Pure-onnxruntime YOLO detector — zero torch / libreyolo imports.

Output format of best.onnx: (1, 5, 8400)
  dim-1 rows:  [cx, cy, w, h, conf_cls0]   (YOLOv8 single-class layout)
  dim-2 cols:  8400 anchor proposals at 640×640 input resolution
"""
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort


class OnnxDetector:
    def __init__(self, model_path: str, conf: float = 0.385, iou: float = 0.5):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.conf_thresh = conf
        self.iou_thresh = iou
        self.input_size = 640
        self.class_names = {0: "signature"}  # single-class model

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # Letterbox: keep aspect ratio, pad to square
        scale = self.input_size / max(orig_w, orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, 640, 640)
        return arr, scale, pad_x, pad_y, orig_w, orig_h

    # ------------------------------------------------------------------
    # NMS helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert (cx,cy,w,h) → (x1,y1,x2,y2)."""
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter + 1e-6
        return inter / union

    def _nms(self, boxes_xyxy: np.ndarray, scores: np.ndarray) -> list[int]:
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            ious = self._iou(boxes_xyxy[i], boxes_xyxy[order[1:]])
            order = order[1:][ious < self.iou_thresh]
        return keep

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    def _postprocess(
        self,
        raw: np.ndarray,          # (1, 5, 8400)
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> list[dict]:
        # Transpose → (8400, 5)
        preds = raw[0].T  # (8400, 5)

        # Last row(s) are class confidences; for single-class: col-4 is score
        scores = preds[:, 4]
        mask = scores >= self.conf_thresh
        preds = preds[mask]
        scores = scores[mask]

        if preds.shape[0] == 0:
            return []

        # Decode cx,cy,w,h → x1,y1,x2,y2 in letterboxed space
        boxes_xywh = preds[:, :4]
        boxes_xyxy = self._xywh2xyxy(boxes_xywh)

        # NMS
        keep = self._nms(boxes_xyxy, scores)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]

        # Map back to original image coordinates
        detections = []
        for box, score in zip(boxes_xyxy, scores):
            x1 = (box[0] - pad_x) / scale
            y1 = (box[1] - pad_y) / scale
            x2 = (box[2] - pad_x) / scale
            y2 = (box[3] - pad_y) / scale

            # Clamp to image bounds
            x1 = max(0.0, min(x1, orig_w))
            y1 = max(0.0, min(y1, orig_h))
            x2 = max(0.0, min(x2, orig_w))
            y2 = max(0.0, min(y2, orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "box": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                    "confidence": round(float(score), 4),
                    "label": self.class_names[0],
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Public API — matches the interface expected by pipeline.detect()
    # ------------------------------------------------------------------
    def predict(self, image_path: str, conf: float | None = None, iou: float | None = None) -> list[dict]:
        """Run inference and return list of detection dicts."""
        old_conf, old_iou = self.conf_thresh, self.iou_thresh
        if conf is not None:
            self.conf_thresh = conf
        if iou is not None:
            self.iou_thresh = iou

        try:
            arr, scale, pad_x, pad_y, orig_w, orig_h = self._preprocess(image_path)
            raw = self.session.run(None, {"images": arr})[0]
            return self._postprocess(raw, scale, pad_x, pad_y, orig_w, orig_h)
        finally:
            self.conf_thresh = old_conf
            self.iou_thresh = old_iou
