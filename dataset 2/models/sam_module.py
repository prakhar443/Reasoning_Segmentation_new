"""
models/sam_module.py — Segment Anything Model (SAM) wrapper.

Takes geometric prompts (boxes + points) from the prompt generator
and produces pixel-level binary masks with IoU confidence scores.
Supports SAM1 (vit_h/l/b) and SAM2 (for temporal propagation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SAMModule(nn.Module):
    """
    Wraps SAM to accept batched prompts and return binary masks.

    NOTE: SAM's image encoder runs once per image; mask decoder runs
    once per prompt. We encode the image once and decode for each prompt.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.sam = self._load_sam(model_cfg)

        if model_cfg.sam_freeze:
            for param in self.sam.parameters():
                param.requires_grad = False

    @staticmethod
    def _load_sam(model_cfg):
        checkpoint = Path(model_cfg.sam_checkpoint)
        model_type = model_cfg.sam_model_type

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {checkpoint}.\n"
                f"Download from: https://github.com/facebookresearch/segment-anything\n"
                f"  vit_h: sam_vit_h_4b8939.pth (2.5GB)\n"
                f"  vit_l: sam_vit_l_0b3195.pth (1.2GB)\n"
                f"  vit_b: sam_vit_b_01ec64.pth (375MB)"
            )

        from segment_anything import sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        return sam

    def encode_image(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Pre-encode image features (run once, reuse for all prompts).
        image_np: (H, W, 3) uint8 numpy array
        Returns image embedding stored in SAM's internal state.
        """
        from segment_anything import SamPredictor
        predictor = SamPredictor(self.sam)
        predictor.set_image(image_np)
        return predictor

    def forward(
        self,
        images_np: List[np.ndarray],      # List of (H,W,3) uint8 arrays
        prompts_batch: List[List[Dict]],  # Output of GeometricPromptGenerator
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images_np:     List[B] of (H,W,3) uint8 arrays (original SAR images)
            prompts_batch: List[B] of prompt-list; each prompt is a dict with
                           keys: boxes, point_coords, point_labels

        Returns:
            masks_batch   (B, 1, H, W) float32 — best mask per image
            iou_scores    (B,) float32          — confidence per mask
        """
        from segment_anything import SamPredictor

        all_masks = []
        all_ious = []

        for img_np, prompts in zip(images_np, prompts_batch):
            predictor = SamPredictor(self.sam)
            predictor.set_image(img_np)

            best_mask = None
            best_iou = -1.0

            for prompt in prompts:
                boxes = prompt["boxes"].numpy()            # (1, 4)
                point_coords = prompt["point_coords"].numpy()  # (1, K, 2)
                point_labels = prompt["point_labels"].numpy()  # (1, K)

                # Decode masks
                masks, iou_preds, _ = predictor.predict(
                    point_coords=point_coords[0],   # (K, 2)
                    point_labels=point_labels[0],   # (K,)
                    box=boxes[0],                    # (4,)
                    multimask_output=True,
                )
                # masks: (num_masks, H, W)  iou_preds: (num_masks,)

                # Pick the mask with highest predicted IoU
                best_idx = np.argmax(iou_preds)
                if iou_preds[best_idx] > best_iou:
                    best_iou = float(iou_preds[best_idx])
                    best_mask = masks[best_idx]  # (H, W)

            if best_mask is None:
                H, W = img_np.shape[:2]
                best_mask = np.zeros((H, W), dtype=bool)
                best_iou = 0.0

            mask_t = torch.from_numpy(best_mask.astype(np.float32)).unsqueeze(0)
            all_masks.append(mask_t)
            all_ious.append(best_iou)

        masks_batch = torch.stack(all_masks)           # (B, 1, H, W)
        iou_scores = torch.tensor(all_ious, dtype=torch.float32)

        return masks_batch, iou_scores


class LightweightMaskDecoder(nn.Module):
    """
    Lightweight alternative to SAM for training on GPU-constrained setups.
    Takes fused visual features and decodes a segmentation mask without SAM.
    Useful when SAM cannot be included due to memory constraints.
    """

    def __init__(self, in_dim: int, image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.image_size = image_size
        h, w = image_size
        side = int(256 ** 0.5)  # assume 256 patch tokens → 16x16

        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
        )

        # Convolutional upsampler
        self.upsample = nn.Sequential(
            # 16x16 → 64x64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 64x64 → 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 128x128 → 256x256
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 256x256 → 512x512
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_tokens: (B, N, fusion_dim)

        Returns:
            masks: (B, 1, H, W) — raw logits (apply sigmoid for probability)
        """
        B, N, D = fused_tokens.shape
        side = int(N ** 0.5)

        x = self.decoder(fused_tokens)          # (B, N, 256)
        x = x.permute(0, 2, 1)                  # (B, 256, N)
        x = x.reshape(B, 256, side, side)       # (B, 256, side, side)
        masks = self.upsample(x)                 # (B, 1, H, W)

        # Resize to exact image_size
        masks = F.interpolate(masks, size=self.image_size,
                               mode="bilinear", align_corners=False)
        return masks
