"""Pure functions: no Celery, no FastAPI, no database. Easy to test alone.
Works exclusively with ONNX runtime models — no torch required.
"""
from pathlib import Path
import numpy as np
from PIL import Image


def detect(detector, image_path: Path, conf: float, iou: float) -> list[dict]:
    """Run OnnxDetector.predict() and return detection dicts."""
    return detector.predict(str(image_path), conf=conf, iou=iou)


def crop_and_save(image: Image.Image, box: list[float], out_path: Path, pad: int = 6) -> Path:
    """Crop bounding box with a small margin so signature strokes aren't cut at edges."""
    w, h = image.size
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((x1, y1, x2, y2)).save(out_path)
    return out_path


def compare(sim_session, inference_module, crop_path: Path, reference_path: Path) -> tuple[float, str]:
    """Compare two signature crops using the ONNX similarity session."""
    mean = np.array(inference_module.CFG["normalisation_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array(inference_module.CFG["normalisation_std"],  dtype=np.float32).reshape(1, 3, 1, 1)

    arr_a = np.asarray(inference_module.preprocess(str(crop_path)),      dtype=np.float32) / 255.0
    arr_b = np.asarray(inference_module.preprocess(str(reference_path)), dtype=np.float32) / 255.0

    # Stack into (2, H, W), expand to (2, 1, H, W), repeat to (2, 3, H, W)
    x = np.stack([arr_a, arr_b])          # (2, H, W)
    x = np.expand_dims(x, 1)              # (2, 1, H, W)
    x = np.repeat(x, 3, axis=1)          # (2, 3, H, W)
    x = (x - mean) / std                 # normalize

    emb = sim_session.run(None, {"input": x.astype(np.float32)})[0]  # (2, 256)
    ea, eb = emb[0:1], emb[1:2]
    score = float(((ea * eb).sum(1) + 1)[0] / 2)

    threshold = 0.7785
    verdict = "SIMILAR" if score >= threshold else "NOT_SIMILAR"
    return round(score, 4), verdict
