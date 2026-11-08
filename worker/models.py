"""Loads both models once per worker process."""
import sys

from huggingface_hub import hf_hub_download, snapshot_download
from libreyolo import LibreYOLO

from app import config

_detector = None
_sim_model = None
_sim_inference = None  # the repo's inference.py module


def get_detector():
    global _detector
    if _detector is None:
        if config.DETECTOR_LOCAL_PATH:
            weights = config.DETECTOR_LOCAL_PATH
        else:
            weights = hf_hub_download(config.DETECTOR_REPO, config.DETECTOR_FILENAME)
        print(f"Loading detector from {weights}")
        _detector = LibreYOLO(weights)
    return _detector


def get_similarity():
    """Returns (model, inference_module). The module has compare() and decide()."""
    global _sim_model, _sim_inference
    if _sim_model is None:
        print(f"Loading similarity model from {config.SIMILARITY_REPO}")
        root = snapshot_download(config.SIMILARITY_REPO)
        if root not in sys.path:
            sys.path.insert(0, root)
        import inference  # inference.py from the Hub repo

        _sim_inference = inference
        _sim_model = inference.load_model()
    return _sim_model, _sim_inference


def load_all():
    get_detector()
    get_similarity()
