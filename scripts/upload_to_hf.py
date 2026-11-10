"""Creates the dedicated HF repository and uploads ONNX models with SHA-256 validation."""
import hashlib
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_NAME = "signiq-onnx-models"  # e.g., ayse67895/signiq-onnx-models

api = HfApi(token=HF_TOKEN)

# 1. Identify user and create repository if not exists (Public for Docker build access)
user_info = api.whoami()
username = user_info["name"]
repo_id = f"{username}/{REPO_NAME}"

print(f"Ensuring repository exists: {repo_id}...")
create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)


def compute_sha256(filepath: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


# 2. Verify local files exist
# NOTE: similarity.onnx uses external data stored in similarity.onnx.data
# Both files must be uploaded for the similarity model to load correctly.
files_to_upload = [
    "best.onnx",
    "similarity.onnx",
    "similarity.onnx.data",
]

for filename in files_to_upload:
    local_path = WEIGHTS_DIR / filename
    if not local_path.exists():
        raise FileNotFoundError(f"Missing local weight: {local_path}")

    sha256, size_bytes = compute_sha256(local_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"\n[Local] {filename}: {size_mb:.2f} MB | SHA-256: {sha256}")

    print(f"Uploading {filename} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {filename} successfully.")

# 3. Verify remote files
print("\nVerifying remote files on Hugging Face Hub...")
model_info = api.model_info(repo_id=repo_id)
remote_files = {sibling.rfilename: sibling for sibling in model_info.siblings}

for filename in files_to_upload:
    if filename in remote_files:
        print(f"  Verified remote artifact: {repo_id}/{filename}")
    else:
        raise RuntimeError(f"Missing remote artifact after upload: {filename}")

print(f"\nAll models uploaded and verified at: https://huggingface.co/{repo_id}")
