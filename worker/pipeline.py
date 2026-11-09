"""Pure functions: no Celery, no FastAPI, no database. Easy to test alone.
Supports both ONNX runtime and PyTorch models seamlessly.
"""
from pathlib import Path
import numpy as np
from PIL import Image


def detect(detector, image_path: Path, conf: float, iou: float) -> list[dict]:
    results = detector.predict(source=str(image_path), conf=conf, iou=iou, save=False)
    result = results[0] if isinstance(results, list) else results
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append(
            {
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(float(box.conf[0]), 4),
                "label": result.names[int(box.cls[0])],
            }
        )
    return detections


def crop_and_save(image: Image.Image, box: list[float], out_path: Path) -> Path:
    w, h = image.size
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((x1, y1, x2, y2)).save(out_path)
    return out_path


def compare(sim_model, inference_module, crop_path: Path, reference_path: Path) -> tuple[float, str]:
    if hasattr(sim_model, "run"):  # ONNX runtime InferenceSession
        mean = np.array(inference_module.CFG["normalisation_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(inference_module.CFG["normalisation_std"], dtype=np.float32).reshape(1, 3, 1, 1)

        arr_a = np.asarray(inference_module.preprocess(str(crop_path)), dtype=np.float32) / 255.0
        arr_b = np.asarray(inference_module.preprocess(str(reference_path)), dtype=np.float32) / 255.0
        x = np.stack([arr_a, arr_b])
        x = np.expand_dims(x, 1)
        x = np.repeat(x, 3, axis=1)
        x = (x - mean) / std

        emb = sim_model.run(None, {"input": x.astype(np.float32)})[0]
        ea, eb = emb[0:1], emb[1:2]
        score = float(((ea * eb).sum(1) + 1)[0] / 2)
    else:
        score = float(inference_module.compare(sim_model, str(crop_path), str(reference_path)))

    threshold = 0.7785
    verdict = "SIMILAR" if score >= threshold else "NOT_SIMILAR"
    return round(score, 4), verdict
