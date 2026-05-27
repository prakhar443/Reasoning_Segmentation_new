"""
data/preprocessing.py — SAR image preprocessing pipeline.

Steps:
  1. Load SAR image (GeoTIFF or JPEG)
  2. Apply speckle filter (Enhanced Lee)
  3. Convert to dB scale and clip
  4. Stack into 3-channel pseudo-RGB tensor for CLIP/ViT ingestion
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
from torchvision import transforms


# ─── Speckle filters ──────────────────────────────────────────────────────────

def lee_filter(img: np.ndarray, window: int = 7, damping: float = 1.0) -> np.ndarray:
    """
    Standard Lee speckle filter for SAR images.
    Assumes single-channel float32 image.
    """
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    pad = window // 2

    # Local mean using box filter
    local_mean = cv2.boxFilter(img, ddepth=-1, ksize=(window, window),
                                borderType=cv2.BORDER_REFLECT)

    # Local variance
    local_mean_sq = cv2.boxFilter(img ** 2, ddepth=-1, ksize=(window, window),
                                   borderType=cv2.BORDER_REFLECT)
    local_var = local_mean_sq - local_mean ** 2
    local_var = np.maximum(local_var, 1e-10)

    # Noise variance estimate (assumed constant across image)
    noise_var = np.mean(local_var) / (np.mean(local_mean) ** 2 + 1e-10)

    # Weighting factor
    weight = local_var / (local_var + noise_var * local_mean ** 2 + 1e-10)
    weight = np.clip(weight, 0.0, 1.0)

    filtered = local_mean + weight * (img - local_mean)
    return filtered.astype(np.float32)


def enhanced_lee_filter(img: np.ndarray, window: int = 7,
                         damping: float = 1.0) -> np.ndarray:
    """
    Enhanced Lee filter — preserves edges better by using a
    damping factor that suppresses filtering near edges.
    """
    img = img.astype(np.float64)

    local_mean = cv2.boxFilter(img, ddepth=-1, ksize=(window, window),
                                borderType=cv2.BORDER_REFLECT)
    local_mean_sq = cv2.boxFilter(img ** 2, ddepth=-1, ksize=(window, window),
                                   borderType=cv2.BORDER_REFLECT)
    local_var = np.maximum(local_mean_sq - local_mean ** 2, 1e-10)

    # Coefficient of variation
    cv_local = np.sqrt(local_var) / (np.abs(local_mean) + 1e-10)
    cv_image = np.sqrt(np.var(img)) / (np.abs(np.mean(img)) + 1e-10)

    # Three-region filter (homogeneous / heterogeneous / edge)
    weight = np.exp(-damping * (cv_local - cv_image) /
                    (cv_local + 1e-10))
    weight = np.clip(weight, 0.0, 1.0)

    filtered = local_mean + weight * (img - local_mean)
    return filtered.astype(np.float32)


# ─── dB conversion ────────────────────────────────────────────────────────────

def linear_to_db(img: np.ndarray,
                  clip_min: float = -30.0,
                  clip_max: float = 5.0) -> np.ndarray:
    """
    Convert linear backscatter to dB: 10 * log10(img).
    Clips to [clip_min, clip_max] dB then normalises to [0, 1].
    """
    img = np.maximum(img, 1e-10)
    db = 10.0 * np.log10(img)
    db = np.clip(db, clip_min, clip_max)
    # Normalise to [0, 1]
    db = (db - clip_min) / (clip_max - clip_min)
    return db.astype(np.float32)


# ─── Channel stacking ─────────────────────────────────────────────────────────

def stack_to_rgb(sar: np.ndarray,
                 dual_pol: bool = False,
                 hv: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert SAR single/dual-pol to a 3-channel (H, W, 3) array.

    Single-pol:  [HH, HH, HH]  — trivial triplication
    Dual-pol:    [HH, HV, HH-HV]  — exploit polarisation difference
    """
    if dual_pol and hv is not None:
        diff = np.abs(sar - hv)
        rgb = np.stack([sar, hv, diff], axis=-1)
    else:
        rgb = np.stack([sar, sar, sar], axis=-1)
    return rgb.astype(np.float32)


# ─── Main SARPreprocessor class ───────────────────────────────────────────────

class SARPreprocessor:
    """
    Full SAR preprocessing pipeline.

    Usage:
        prep = SARPreprocessor(cfg.data)
        tensor = prep(img_array)        # returns (3, H, W) torch.Tensor
    """

    def __init__(self, data_cfg):
        self.filter_type = data_cfg.speckle_filter
        self.window = data_cfg.speckle_window
        self.damping = data_cfg.speckle_damping
        self.to_db = data_cfg.to_db
        self.db_clip_min = data_cfg.db_clip_min
        self.db_clip_max = data_cfg.db_clip_max
        self.dual_pol = data_cfg.dual_pol
        self.image_size = data_cfg.image_size

        self.normalize = transforms.Normalize(
            mean=data_cfg.image_mean,
            std=data_cfg.image_std,
        )
        self.resize = transforms.Resize(
            self.image_size, interpolation=transforms.InterpolationMode.BILINEAR
        )

    def apply_speckle_filter(self, img: np.ndarray) -> np.ndarray:
        if self.filter_type == "lee":
            return lee_filter(img, self.window, self.damping)
        elif self.filter_type == "enhanced_lee":
            return enhanced_lee_filter(img, self.window, self.damping)
        return img.astype(np.float32)

    def __call__(self,
                 img: Union[np.ndarray, Image.Image, str, Path],
                 hv_img: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Args:
            img:    SAR image as numpy array (H,W) or (H,W,1),
                    PIL Image, or file path.
            hv_img: Optional HV polarisation array (H,W) for dual-pol mode.

        Returns:
            Normalised float32 tensor of shape (3, H, W).
        """
        # ── 1. Load ────────────────────────────────────────────────────────────
        if isinstance(img, (str, Path)):
            img = self._load_from_path(img)
        elif isinstance(img, Image.Image):
            img = np.array(img.convert("L")).astype(np.float32)
        elif isinstance(img, np.ndarray):
            if img.ndim == 3:
                img = img[..., 0]
            img = img.astype(np.float32)

        # ── 2. Speckle filter ──────────────────────────────────────────────────
        img = self.apply_speckle_filter(img)
        if hv_img is not None:
            hv_img = self.apply_speckle_filter(hv_img.astype(np.float32))

        # ── 3. dB conversion ──────────────────────────────────────────────────
        if self.to_db:
            img = linear_to_db(img, self.db_clip_min, self.db_clip_max)
            if hv_img is not None:
                hv_img = linear_to_db(hv_img, self.db_clip_min, self.db_clip_max)
        else:
            img = np.clip(img, 0.0, 1.0)
            if hv_img is not None:
                hv_img = np.clip(hv_img, 0.0, 1.0)

        # ── 4. Stack to RGB ────────────────────────────────────────────────────
        rgb = stack_to_rgb(img, dual_pol=self.dual_pol, hv=hv_img)  # (H,W,3)

        # ── 5. Resize & to tensor ─────────────────────────────────────────────
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (3,H,W)
        tensor = self.resize(tensor)
        tensor = self.normalize(tensor)

        return tensor

    def preprocess_mask(self, mask: Union[np.ndarray, Image.Image, str, Path]) -> torch.Tensor:
        """
        Load and resize a binary segmentation mask.
        Returns (1, H, W) float32 tensor with values in {0, 1}.
        """
        if isinstance(mask, (str, Path)):
            mask = np.array(Image.open(mask).convert("L"))
        elif isinstance(mask, Image.Image):
            mask = np.array(mask.convert("L"))

        mask = (mask > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)
        mask_t = transforms.Resize(
            self.image_size, interpolation=transforms.InterpolationMode.NEAREST
        )(mask_t)
        return mask_t

    @staticmethod
    def _load_from_path(path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            try:
                import rasterio
                with rasterio.open(path) as src:
                    data = src.read(1).astype(np.float32)
            except ImportError:
                data = np.array(Image.open(path)).astype(np.float32)
        else:
            img = Image.open(path)
            if img.mode != "L":
                img = img.convert("L")
            data = np.array(img).astype(np.float32)
            # Normalise 8-bit to [0,1] for subsequent dB step (treat as pseudo-linear)
            data = data / 255.0
        return data
