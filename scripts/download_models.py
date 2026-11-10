"""Pre-download ONNX weights during Docker build to eliminate runtime cold-start."""
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN") or None

# Default to the dedicated ONNX repository
REPO_ID = os.getenv("ONNX_REPO_ID", "ayse67895/signiq-onnx-models")
DETECTOR_FILENAME = os.getenv("DETECTOR_ONNX_FILENAME", "best.onnx")
SIMILARITY_FILENAME = os.getenv("SIMILARITY_ONNX_FILENAME", "similarity.onnx")
SIMILARITY_DATA_FILENAME = os.getenv("SIMILARITY_DATA_FILENAME", "similarity.onnx.data")


def download_artifact(repo_id: str, filename: str, target_name: str) -> Path:
    target_path = WEIGHTS_DIR / target_name
    print(f"Downloading {filename} from {repo_id}...")

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=HF_TOKEN,
        local_files_only=False,
    )

    shutil.copyfile(downloaded_path, str(target_path))

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded model is empty or missing: {target_path}")

    size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"Saved {target_name} ({size_mb:.2f} MB) to {target_path}")
    return target_path


if __name__ == "__main__":
    # Download detector ONNX
    download_artifact(REPO_ID, DETECTOR_FILENAME, "best.onnx")

    # Download similarity ONNX model header + its external data file
    # IMPORTANT: similarity.onnx uses external data format; both files must
    # reside in the same directory (weights/) for onnxruntime to load correctly.
    download_artifact(REPO_ID, SIMILARITY_FILENAME, "similarity.onnx")
    download_artifact(REPO_ID, SIMILARITY_DATA_FILENAME, "similarity.onnx.data")

    print("All ONNX models successfully downloaded and verified into weights/")
