"""
data/dataset.py — PyTorch Dataset for sea ice SAR segmentation.

Expected directory layout:
    dataset/
    ├── Young Ice/
    │   ├── images/                                  (SAR images:  1_1003_.jpg)
    │   ├── masks/                                   (binary masks: 1_1003_scat.jpg)
    │   └── descriptions/
    │       └── young_ice_descriptions_appended.xlsx (columns: image, short_descriptions,
    │                                                  long_descriptions)
    ├── First Year Ice/
    │   └── ...
    └── ...  (one folder per class listed in ICE_CLASSES)

Mask naming convention:
    image filename : 1_1003_.jpg
    mask  filename : 1_1003_scat.jpg   (stem's trailing '_' replaced by '_scat')

Description xlsx image column uses the mask filename (e.g. 1_1134_scat.jpg);
the loader converts it back to the image filename for the lookup key.
"""

import ast
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import ICE_CLASS_TO_IDX, ICE_CLASSES, cfg
from data.preprocessing import SARPreprocessor


# ─── Albumentations version detection ────────────────────────────────────────
# Breaking API differences between albumentations 1.x and 2.x (verified on 2.0.8):
#
#   Transform          | 1.x signature          | 2.x signature
#   -------------------+------------------------+---------------------------
#   RandomResizedCrop  | height=h, width=w      | size=(h, w)    ← CHANGED
#   Resize             | height=h, width=w      | height=h, width=w (same)
#   GaussNoise         | var_limit=(lo, hi)     | no var_limit arg (use defaults)
#   ElasticTransform   | alpha=30, sigma=5      | no alpha/sigma (use defaults)

def _albu_major() -> int:
    try:
        return int(A.__version__.split(".")[0])
    except Exception:
        return 1

_ALBU2 = _albu_major() >= 2  # True when albumentations >= 2.0 is installed


# ─── Augmentation pipelines ───────────────────────────────────────────────────

def get_train_augmentations(image_size: Tuple[int, int]) -> A.Compose:
    h, w = image_size
    # Only RandomResizedCrop differs between 1.x and 2.x
    if _ALBU2:
        crop = A.RandomResizedCrop(size=(h, w), scale=(0.7, 1.0), p=1.0)
        noise = A.GaussNoise(p=1.0)
        elastic = A.ElasticTransform(p=0.2)
    else:
        crop = A.RandomResizedCrop(height=h, width=w, scale=(0.7, 1.0), p=1.0)
        noise = A.GaussNoise(var_limit=(0.001, 0.005), p=1.0)
        elastic = A.ElasticTransform(alpha=30, sigma=5, p=0.2)

    return A.Compose([
        crop,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.OneOf([noise, A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        elastic,
    ], additional_targets={"mask": "mask"})


def get_val_augmentations(image_size: Tuple[int, int]) -> A.Compose:
    # A.Resize(height=h, width=w) works in BOTH albumentations 1.x and 2.x
    h, w = image_size
    return A.Compose(
        [A.Resize(height=h, width=w)],
        additional_targets={"mask": "mask"},
    )


# ─── Description parser ───────────────────────────────────────────────────────

def parse_description_cell(cell: str) -> str:
    """
    The CSV stores descriptions as Python list literals: ['text1'].
    Parse and return the first string element, stripping whitespace.
    """
    try:
        parsed = ast.literal_eval(str(cell))
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0]).strip()
    except Exception:
        pass
    return str(cell).strip()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SeaIceDataset(Dataset):
    """
    Dataset for sea ice SAR reasoning segmentation.

    Returns a dict with keys:
        image         (3, H, W) float32 tensor — normalised SAR pseudo-RGB
        mask          (1, H, W) float32 tensor — binary segmentation mask
        label         int — ice class index (0–5)
        short_desc    str — short description / query
        long_desc     str — long description with spatial question
        image_path    str — original file path (for debugging)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",       # "train" | "val" | "test"
        data_cfg=None,
        use_augmentation: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.data_cfg = data_cfg or cfg.data
        self.use_augmentation = use_augmentation and split == "train"

        self.preprocessor = SARPreprocessor(self.data_cfg)
        image_size = self.data_cfg.image_size

        if self.use_augmentation:
            self.aug = get_train_augmentations(image_size)
        else:
            self.aug = get_val_augmentations(image_size)

        self.samples: List[Dict] = []
        self._load_all_samples()

        # Split deterministically
        random.seed(self.data_cfg.seed)
        random.shuffle(self.samples)
        n = len(self.samples)
        n_train = int(n * self.data_cfg.train_ratio)
        n_val = int(n * self.data_cfg.val_ratio)

        if split == "train":
            self.samples = self.samples[:n_train]
        elif split == "val":
            self.samples = self.samples[n_train: n_train + n_val]
        else:
            self.samples = self.samples[n_train + n_val:]

    @staticmethod
    def _mask_name_from_image(img_name: str) -> str:
        """
        Convert image filename to mask filename.

        Convention used in this dataset:
            image : 1_1003_.jpg   (stem ends with a trailing underscore)
            mask  : 1_1003_scat.jpg

        We strip the trailing '_' from the stem and append '_scat'.
        """
        p = Path(img_name)
        mask_stem = p.stem.rstrip("_") + "_scat"
        return mask_stem + p.suffix

    def _load_all_samples(self):
        for ice_class in ICE_CLASSES:
            class_dir = self.data_root / ice_class
            if not class_dir.exists():
                continue

            img_dir  = class_dir / self.data_cfg.image_subdir
            mask_dir = class_dir / self.data_cfg.mask_subdir
            desc_dir = class_dir / self.data_cfg.descriptions_subdir

            # ── Load descriptions from per-class xlsx files ────────────────────
            # The xlsx 'image' column uses the *mask* filename (e.g. 1_1134_scat.jpg).
            # We convert it to the *image* filename for the lookup key.
            desc_map: Dict[str, Dict[str, str]] = {}
            if desc_dir.exists():
                for xlsx_path in sorted(desc_dir.glob("*.xlsx")):
                    try:
                        df = pd.read_excel(xlsx_path, engine="openpyxl")
                    except Exception as e:
                        print(f"[dataset] Warning: could not read {xlsx_path}: {e}")
                        continue
                    for _, row in df.iterrows():
                        mask_fname = str(row.get("image", "")).strip()
                        if not mask_fname:
                            continue
                        # xlsx stores mask filename; map back to image filename
                        img_fname = mask_fname.replace("_scat.", "_.")
                        short = parse_description_cell(row.get("short_descriptions", ""))
                        long  = parse_description_cell(row.get("long_descriptions", ""))
                        desc_map[img_fname] = {"short": short, "long": long}

            # ── Collect image–mask pairs ───────────────────────────────────────
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    continue

                # Derive mask path using dataset naming convention
                mask_name = self._mask_name_from_image(img_path.name)
                mask_path = mask_dir / mask_name

                if not mask_path.exists():
                    # Fallback 1: same filename as image
                    alt = mask_dir / img_path.name
                    if alt.exists():
                        mask_path = alt
                    else:
                        # Fallback 2: search for any file sharing the base number
                        base = img_path.stem.rstrip("_")
                        candidates = list(mask_dir.glob(f"{base}*"))
                        if candidates:
                            mask_path = candidates[0]
                        else:
                            continue   # no mask found → skip sample

                # Retrieve description (default if missing)
                desc = desc_map.get(img_path.name, {
                    "short": f"Segment the {ice_class} in this SAR image.",
                    "long":  (
                        f"Identify and segment the {ice_class} region. "
                        f"Where is the most characteristic feature of this ice type visible?"
                    ),
                })

                self.samples.append({
                    "image_path": str(img_path),
                    "mask_path":  str(mask_path),
                    "label":      ICE_CLASS_TO_IDX[ice_class],
                    "ice_class":  ice_class,
                    "short_desc": desc["short"],
                    "long_desc":  desc["long"],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # ── Load image (PIL → numpy for albumentations) ────────────────────────
        img_pil = Image.open(sample["image_path"])
        if img_pil.mode != "L":
            img_pil = img_pil.convert("L")
        img_np = np.array(img_pil)                  # (H, W) uint8

        # ── Load mask ──────────────────────────────────────────────────────────
        mask_pil = Image.open(sample["mask_path"]).convert("L")
        mask_np = (np.array(mask_pil) > 127).astype(np.uint8)  # (H, W) binary

        # ── Augmentation (spatial transforms applied to both img and mask) ─────
        augmented = self.aug(image=img_np, mask=mask_np)
        img_np = augmented["image"]
        mask_np = augmented["mask"]

        # ── SAR preprocessing (speckle filter + dB + normalise) ───────────────
        img_tensor = self.preprocessor(img_np.astype(np.float32) / 255.0)

        # ── Mask tensor ────────────────────────────────────────────────────────
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

        return {
            "image": img_tensor,              # (3, H, W)
            "mask": mask_tensor,              # (1, H, W)
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "short_desc": sample["short_desc"],
            "long_desc": sample["long_desc"],
            "image_path": sample["image_path"],
        }

    def get_class_weights_for_sampler(self) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler."""
        from config import ICE_CLASS_WEIGHTS
        labels = [s["label"] for s in self.samples]
        weights = [ICE_CLASS_WEIGHTS[l] for l in labels]
        return torch.tensor(weights, dtype=torch.float32)


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(data_cfg=None, train_cfg=None):
    """
    Returns (train_loader, val_loader, test_loader).
    Applies WeightedRandomSampler to train split for class balance.
    """
    data_cfg = data_cfg or cfg.data
    train_cfg = train_cfg or cfg.train

    train_ds = SeaIceDataset(data_cfg.data_root, split="train",
                              data_cfg=data_cfg, use_augmentation=True)
    val_ds = SeaIceDataset(data_cfg.data_root, split="val",
                            data_cfg=data_cfg, use_augmentation=False)
    test_ds = SeaIceDataset(data_cfg.data_root, split="test",
                             data_cfg=data_cfg, use_augmentation=False)

    # Balanced sampler for training
    sample_weights = train_ds.get_class_weights_for_sampler()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict]) -> Dict:
    """Stack tensors, keep string fields as lists."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "short_desc": [b["short_desc"] for b in batch],
        "long_desc": [b["long_desc"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }
