"""
Pure numpy/pillow reimplementation of the HF inference.py preprocessing pipeline.
Exact copy of preprocess() from izamscientist/signature-similarity-convnext-tiny,
with CFG values hardcoded from model/config.json.

Zero torch / torchvision imports.
"""
import numpy as np
from PIL import Image

# Exact values from model/config.json (commit e1f3bf7e71f79ab4e5fd044eb186d7810adaf786)
CFG = {
    "image_size": 224,
    "normalisation_mean": [0.485, 0.456, 0.406],
    "normalisation_std":  [0.229, 0.224, 0.225],
    "threshold_main":     0.7784532904624939,
    "threshold_accept":   0.9106097221374512,
    "threshold_reject":   0.5376065969467163,
    "arch": "convnext_tiny",
    "dim": 256,
}

SIZE = CFG["image_size"]


def to_gray_on_white(im: Image.Image) -> Image.Image:
    """Step 1: transparent background → white paper, then grayscale."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        paper = Image.new("RGBA", im.size, (255, 255, 255, 255))
        paper.alpha_composite(im)
        im = paper
    return im.convert("L")


def preprocess(path_or_image) -> Image.Image:
    """
    Steps 2-9 of the dataset pipeline — identical logic to inference.py preprocess().
    Returns a (224, 224) PIL Image in mode 'L'.
    """
    im = Image.open(path_or_image) if isinstance(path_or_image, str) else path_or_image
    a = np.asarray(to_gray_on_white(im), dtype=np.float32)

    if np.median(a) < 127:          # dark background → invert so ink is dark on white
        a = 255 - a

    lo, hi = np.percentile(a, 1), np.percentile(a, 90)   # contrast stretch
    if hi - lo < 10:
        raise ValueError("picture is blank")
    a = np.clip((a - lo) / (hi - lo) * 255, 0, 255)

    ink = a < 160                                          # ink pixels
    h, w = ink.shape
    ink[ink.sum(1) / w > 0.4, :] = False                  # remove horizontal box lines
    ink[:, ink.sum(0) / h > 0.4] = False                  # remove vertical box lines

    if ink.sum() < 30:
        raise ValueError("picture is blank")

    rows = np.where(ink.any(1))[0]
    cols = np.where(ink.any(0))[0]
    pad = max(2, int(0.04 * max(rows[-1] - rows[0], cols[-1] - cols[0])))
    r0, r1 = max(rows[0] - pad, 0),  min(rows[-1] + pad + 1, h)
    c0, c1 = max(cols[0] - pad, 0),  min(cols[-1] + pad + 1, w)

    crop = np.where(ink[r0:r1, c0:c1], a[r0:r1, c0:c1], 255.0)
    ch, cw = crop.shape
    s = max(ch, cw)
    canvas = np.full((s, s), 255.0, np.float32)
    canvas[(s - ch) // 2:(s - ch) // 2 + ch,
           (s - cw) // 2:(s - cw) // 2 + cw] = crop

    return Image.fromarray(canvas.astype(np.uint8), "L").resize((SIZE, SIZE), Image.LANCZOS)
