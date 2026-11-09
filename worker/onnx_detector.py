"""Pure-onnxruntime YOLO detector — zero torch / libreyolo imports.

Model Architecture: LibreYOLO9 (YOLOv9)
- Input: images (1, 3, 640, 640) float32 in [0, 1]
- Preprocessing: top-left letterbox padding (standard for LibreYOLO YOLOv9 checkpoints)
- Output format: (1, 5, 8400) where dim-1 rows 0:4 are already xyxy (x1, y1, x2, y2)
  and row 4 is confidence score.
"""
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
import onnxruntime as ort


class OnnxDetector:
    def __init__(self, model_path: str, conf: float = 0.385, iou: float = 0.5):
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
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
        self.class_names = {0: "signature"}

    # ------------------------------------------------------------------
    # Preprocessing: top-left letterboxing (matches LibreYOLO YOLOv9)
    # ------------------------------------------------------------------
    def _preprocess(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # Scale keeping aspect ratio
        ratio = min(self.input_size / orig_h, self.input_size / orig_w)
        new_w = max(int(orig_w * ratio), 1)
        new_h = max(int(orig_h * ratio), 1)
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        # LibreYOLO YOLOv9 uses topleft letterbox (image at (0, 0), pad right/bottom)
        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))
        canvas.paste(resized, (0, 0))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, 640, 640)
        return arr, ratio, orig_w, orig_h

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------
    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            a2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            iou = inter / (a1 + a2 - inter + 1e-6)
            order = order[1:][iou < self.iou_thresh]
        return np.array(keep, dtype=np.int64)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    def _postprocess(
        self,
        raw: np.ndarray,      # (1, 5, 8400)
        ratio: float,
        orig_w: int,
        orig_h: int,
    ) -> list[dict]:
        preds = raw[0].T
        boxes_xyxy = preds[:, :4]
        scores = preds[:, 4]

        mask = scores >= self.conf_thresh
        boxes_xyxy = boxes_xyxy[mask]
        scores = scores[mask]

        if len(boxes_xyxy) == 0:
            return []

        keep = self._nms(boxes_xyxy, scores)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]

        boxes_orig = boxes_xyxy / ratio
        boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_w)
        boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_h)

        detections = []
        for box, score in zip(boxes_orig, scores):
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "box": [
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2), 1),
                        round(float(y2), 1),
                    ],
                    "confidence": round(float(score), 4),
                    "label": self.class_names[0],
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        image_path: Union[str, Path],
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[dict]:
        old_conf, old_iou = self.conf_thresh, self.iou_thresh
        if conf is not None:
            self.conf_thresh = conf
        if iou is not None:
            self.iou_thresh = iou

        try:
            arr, ratio, orig_w, orig_h = self._preprocess(str(image_path))
            raw = self.session.run(None, {"images": arr})[0]
            return self._postprocess(raw, ratio, orig_w, orig_h)
        finally:
            self.conf_thresh = old_conf
            self.iou_thresh = old_iou
