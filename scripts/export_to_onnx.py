"""Exports LibreYOLO detector and ConvNeXt Siamese similarity models to ONNX.
This reduces RAM from ~800MB to ~150-250MB and accelerates CPU inference.
"""
import os
import shutil
import sys
from pathlib import Path

# Fix Windows cp1252 UnicodeEncodeError for PyTorch exporter emoji printing
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import torchvision
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def export_detector():
    onnx_target = WEIGHTS_DIR / "best.onnx"
    if onnx_target.exists():
        print(f"Detector already exported at {onnx_target}. Skipping.")
        return onnx_target

    pt_path = WEIGHTS_DIR / "best.pt"
    if not pt_path.exists():
        print("Downloading detector weights from Hugging Face Hub (ayse67895/Signature-detection)...")
        downloaded = hf_hub_download(repo_id="ayse67895/Signature-detection", filename="best.pt")
        shutil.copyfile(downloaded, str(pt_path))

    from libreyolo import LibreYOLO
    print(f"Loading LibreYOLO detector from {pt_path}...")
    model = LibreYOLO(str(pt_path))
    
    print(f"Exporting detector to ONNX...")
    out_file = model.export(format="onnx", imgsz=640)
    out_path = Path(out_file)
    if out_path.exists() and out_path != onnx_target:
        shutil.move(str(out_path), str(onnx_target))

    print(f"Detector successfully exported to: {onnx_target}")
    return onnx_target


def export_similarity():
    onnx_target = WEIGHTS_DIR / "similarity.onnx"
    if onnx_target.exists():
        print(f"Similarity model already exported at {onnx_target}. Skipping.")
        return onnx_target

    sim_repo = "izamscientist/signature-similarity-convnext-tiny"
    print(f"Fetching similarity model snapshot from {sim_repo}...")
    root = snapshot_download(sim_repo)
    if root not in sys.path:
        sys.path.insert(0, root)
    import inference

    print("Loading similarity PyTorch model...")
    pt_model = inference.load_model()
    pt_model.eval()

    print(f"Exporting similarity model to {onnx_target}...")

    # ConvNeXt expects (batch, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # Wrap to call .embed() directly
    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, x):
            return self.core.embed(x)

    wrapper = EmbeddingWrapper(pt_model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_target),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    )
    print(f"Similarity model successfully exported to: {onnx_target}")
    return onnx_target


if __name__ == "__main__":
    export_detector()
    export_similarity()
