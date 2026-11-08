"""Pure functions: no Celery, no FastAPI, no database. Easy to test alone."""
from pathlib import Path

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
    score = float(inference_module.compare(sim_model, str(crop_path), str(reference_path)))
    threshold = 0.7785
    verdict = "SIMILAR" if score >= threshold else "NOT_SIMILAR"
    return round(score, 4), verdict
