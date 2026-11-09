"""Loads both models once per worker / server process.
Pure onnxruntime — zero torch / torchvision / libreyolo imports at runtime.
"""
import os
import sys

import onnxruntime as ort
from huggingface_hub import snapshot_download

from app import config

# Restrict ONNX thread pools to stay within 512 MB
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_detector = None
_sim_session = None   # ort.InferenceSession for similarity.onnx
_sim_inference = None  # inference.py module from HF repo (preprocessing only)


def get_detector():
    """Returns OnnxDetector (pure onnxruntime, no torch)."""
    global _detector
    if _detector is None:
        from worker.onnx_detector import OnnxDetector

        onnx_path = config.BASE_DIR / "weights" / "best.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Detector ONNX not found at {onnx_path}. "
                "Run 'python scripts/export_to_onnx.py' during the build step."
            )
        print(f"[models] Loading detector ONNX from {onnx_path}")
        _detector = OnnxDetector(str(onnx_path), conf=config.DETECT_CONF, iou=config.DETECT_IOU)
    return _detector


def get_similarity():
    """Returns (ort.InferenceSession, inference_module)."""
    global _sim_session, _sim_inference

    if _sim_session is None:
        onnx_path = config.BASE_DIR / "weights" / "similarity.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Similarity ONNX not found at {onnx_path}. "
                "Run 'python scripts/export_to_onnx.py' during the build step."
            )

        print(f"[models] Loading similarity ONNX from {onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _sim_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    if _sim_inference is None:
        # snapshot_download only downloads metadata/inference.py, not model weights (~few MB)
        root = snapshot_download(config.SIMILARITY_REPO)
        if root not in sys.path:
            sys.path.insert(0, root)
        import inference  # preprocessing helpers only
        _sim_inference = inference

    return _sim_session, _sim_inference


def load_all():
    get_detector()
    get_similarity()
