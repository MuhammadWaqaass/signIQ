"""Loads both models once per worker / server process (supports ONNX & PyTorch)."""
import os
import sys
from pathlib import Path

# Restrict multi-threading memory pools to stay safely within 512 MB
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from huggingface_hub import hf_hub_download, snapshot_download
from libreyolo import LibreYOLO

from app import config

_detector = None
_sim_model = None
_sim_inference = None  # the repo's inference.py module


def get_detector():
    """Loads LibreYOLO detector: prefers lightweight ONNX if available, else PyTorch .pt."""
    global _detector
    if _detector is None:
        onnx_path = config.BASE_DIR / "weights" / "best.onnx"
        if onnx_path.exists():
            weights = str(onnx_path)
        elif config.DETECTOR_LOCAL_PATH and Path(config.DETECTOR_LOCAL_PATH).exists():
            weights = config.DETECTOR_LOCAL_PATH
        elif (config.BASE_DIR / "weights" / "best.pt").exists():
            weights = str(config.BASE_DIR / "weights" / "best.pt")
        else:
            weights = hf_hub_download(config.DETECTOR_REPO, config.DETECTOR_FILENAME)
        
        print(f"Loading detector from {weights}")
        _detector = LibreYOLO(weights)
    return _detector


def get_similarity():
    """Returns (model, inference_module).
    Prefers lightweight ONNX session for ConvNeXt, falls back to PyTorch if missing.
    """
    global _sim_model, _sim_inference
    if _sim_model is None:
        root = snapshot_download(config.SIMILARITY_REPO)
        if root not in sys.path:
            sys.path.insert(0, root)
        import inference  # inference.py from the Hub repo
        _sim_inference = inference

        onnx_path = config.BASE_DIR / "weights" / "similarity.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                print(f"Loading similarity model from ONNX runtime: {onnx_path}")
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                _sim_model = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
                return _sim_model, _sim_inference
            except Exception as e:
                print(f"ONNX session load fallback to PyTorch: {e}")

        print(f"Loading similarity PyTorch model from {config.SIMILARITY_REPO}")
        _sim_model = inference.load_model()
    return _sim_model, _sim_inference


def load_all():
    get_detector()
    get_similarity()
