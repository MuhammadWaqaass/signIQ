"""Pre-download weights during Docker build to eliminate runtime cold-start."""
from huggingface_hub import hf_hub_download, snapshot_download

print("Pre-caching similarity model...")
snapshot_download(repo_id="izamscientist/signature-similarity-convnext-tiny")

print("Pre-caching detector weights...")
hf_hub_download(repo_id="ayse67895/Signature-detection", filename="best.pt")

print("All models successfully cached!")
