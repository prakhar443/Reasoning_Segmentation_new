"""
baseline_allforeground.py — Degenerate all-foreground baseline for segmentation.

Computes the metrics a trivial predictor that labels EVERY pixel as ice would
obtain on the test split, under exactly the published mask pipeline:

  1. Load the `_scat` scattering map (native ~139x187, portrait).
  2. Otsu-binarise BEFORE any resize (with the same degenerate-result guard
     as data/dataset.py::binarize_mask).
  3. NEAREST-resize ("stretch") to the image frame (256x256), then to the
     processing resolution (512x512) — matching mask_resize_mode="stretch"
     and image_size=(512,512) in config.py.
  4. Reproduce the per-class deterministic split of data/dataset.py
     (random.Random(seed + label), 70/15/15) and keep the test slice.

For an all-foreground prediction, per-image IoU equals the ground-truth
foreground fraction, Dice equals 2f/(1+f), pixel accuracy equals f, precision
equals f and recall equals 1. The script needs only numpy + PIL + cv2 (no
torch), so it runs anywhere the repo is checked out.

Usage:
    python baseline_allforeground.py [--split test] [--data_root dataset]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Keep in sync with config.ICE_CLASSES (importing config pulls in torch).
ICE_CLASSES = [
    "Young Ice",
    "First Year Ice",
    "Floating Ice",
    "Glaciers",
    "Icebergs",
    "Old Ice",
]
SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
IMAGE_SIZE = 512
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def otsu_threshold(gray: np.ndarray) -> int:
    """Identical to data/dataset.py::_otsu_threshold."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, thr = 0.0, 0.0, 0.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > max_var:
            max_var = between
            thr = t
    return thr


def binarize(gray: np.ndarray) -> np.ndarray:
    """Identical to data/dataset.py::binarize_mask(mode='otsu')."""
    thr = otsu_threshold(gray)
    binm = (gray > thr).astype(np.uint8)
    fg = binm.mean()
    if fg < 0.005 or fg > 0.995:
        binm = (gray > gray.mean()).astype(np.uint8)
    return binm


def mask_name_from_image(img_name: str) -> str:
    p = Path(img_name)
    return p.stem.rstrip("_") + "_scat" + p.suffix


def collect_split(data_root: Path, split: str):
    """Reproduce the per-class deterministic split of data/dataset.py."""
    selected = []
    for label, ice_class in enumerate(ICE_CLASSES):
        img_dir = data_root / ice_class / "images"
        mask_dir = data_root / ice_class / "masks"
        if not img_dir.exists():
            continue
        items = []
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            mask_path = mask_dir / mask_name_from_image(img_path.name)
            if not mask_path.exists():
                alt = mask_dir / img_path.name
                if alt.exists():
                    mask_path = alt
                else:
                    cands = list(mask_dir.glob(f"{img_path.stem.rstrip('_')}*"))
                    if not cands:
                        continue
                    mask_path = cands[0]
            items.append((str(img_path), str(mask_path), label))
        rng = random.Random(SEED + label)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        if n >= 3 and n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        if split == "train":
            selected.extend(items[:n_train])
        elif split == "val":
            selected.extend(items[n_train:n_train + n_val])
        else:
            selected.extend(items[n_train + n_val:])
    return selected


def gt_mask_512(img_path: str, mask_path: str) -> np.ndarray:
    """Published mask pipeline: Otsu on native map -> stretch to image frame
    -> resize to 512 (nearest, as albumentations does for masks)."""
    img = Image.open(img_path)
    mask_gray = np.array(Image.open(mask_path).convert("L"))
    mask_bin = binarize(mask_gray)
    if (mask_bin.shape[1], mask_bin.shape[0]) != img.size:
        mask_bin = np.array(Image.fromarray(mask_bin).resize(img.size, Image.NEAREST))
    return cv2.resize(mask_bin, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="dataset")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    samples = collect_split(Path(args.data_root), args.split)
    print(f"{args.split} split: {len(samples)} images")

    fracs, total_fg, total_px = [], 0, 0
    per_class = {c: [] for c in ICE_CLASSES}
    for img_path, mask_path, label in samples:
        m = gt_mask_512(img_path, mask_path)
        f = float(m.mean())
        fracs.append(f)
        per_class[ICE_CLASSES[label]].append(f)
        total_fg += int(m.sum())
        total_px += m.size

    fracs = np.array(fracs)
    miou = fracs.mean()                       # per-image IoU of all-fg = fg fraction
    dice = (2 * fracs / (1 + fracs)).mean()   # per-image Dice of all-fg
    ciou = total_fg / total_px                # cumulative IoU of all-fg
    print("\nAll-foreground baseline (predict every pixel = ice):")
    print(f"  mIoU            : {miou:.4f}")
    print(f"  cIoU            : {ciou:.4f}")
    print(f"  Dice            : {dice:.4f}")
    print(f"  Pixel accuracy  : {miou:.4f}  (= mean fg fraction)")
    print(f"  Precision/Recall: {miou:.4f} / 1.0000")
    print("\nPer-class mean foreground fraction (= per-class all-fg IoU):")
    for c in ICE_CLASSES:
        v = per_class[c]
        if v:
            print(f"  {c:<16} {np.mean(v):.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
