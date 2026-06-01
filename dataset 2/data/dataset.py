"""
data/dataset.py — PyTorch Dataset for sea ice SAR segmentation.

Expected directory layout:
    dataset/
    ├── first_year_ice/
    │   ├── images/          (original SAR .jpg/.tif)
    │   ├── masks/           (binary mask images, same filename)
    │   └── descriptions.csv (columns: image, short_descriptions, long_descriptions)
    ├── young_ice/
    │   └── ...
    └── ...  (one folder per class in ICE_CLASSES)

Each row in descriptions.csv looks like the example provided:
    image, short_descriptions, long_descriptions
    1_4719_scat.jpg, ['...'], ["...question..."]
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


# ─── Augmentation pipelines ───────────────────────────────────────────────────

def get_train_augmentations(image_size: Tuple[int, int]) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.RandomResizedCrop(height=h, width=w, scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(0.001, 0.005), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
    ], additional_targets={"mask": "mask"})


def get_val_augmentations(image_size: Tuple[int, int]) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.Resize(height=h, width=w),
    ], additional_targets={"mask": "mask"})


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

    def _load_all_samples(self):
        for ice_class in ICE_CLASSES:
            class_dir = self.data_root / ice_class
            if not class_dir.exists():
                continue

            img_dir = class_dir / self.data_cfg.image_subdir
            mask_dir = class_dir / self.data_cfg.mask_subdir
            desc_file = class_dir / self.data_cfg.descriptions_file

            # Load descriptions CSV
            desc_map = {}
            if desc_file.exists():
                df = pd.read_csv(desc_file)
                for _, row in df.iterrows():
                    fname = str(row["image"]).strip()
                    short = parse_description_cell(row.get("short_descriptions", ""))
                    long = parse_description_cell(row.get("long_descriptions", ""))
                    desc_map[fname] = {"short": short, "long": long}

            # Collect image–mask pairs
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    continue

                # Find corresponding mask
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    # Try common suffix variants
                    for ext in [".png", ".jpg", ".tif"]:
                        alt = mask_dir / (img_path.stem + ext)
                        if alt.exists():
                            mask_path = alt
                            break
                    else:
                        # No mask found — skip
                        continue

                # Find description
                desc = desc_map.get(img_path.name, {
                    "short": f"Segment the {ice_class.replace('_', ' ')} in this SAR image.",
                    "long": f"Identify and segment the {ice_class.replace('_', ' ')} region. "
                            f"Where is the most characteristic feature of this ice type visible?",
                })

                self.samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "label": ICE_CLASS_TO_IDX[ice_class],
                    "ice_class": ice_class,
                    "short_desc": desc["short"],
                    "long_desc": desc["long"],
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
