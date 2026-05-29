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


def _group_norm(num_channels: int) -> nn.GroupNorm:
    """GroupNorm with a divisor-safe group count (batch-size independent,
    unlike BatchNorm which is unusable at the batch_size=2 used on a T4)."""
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv-GN-GELU ×2 — the standard U-Net block, GroupNorm for small batches."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        _group_norm(out_ch), nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        _group_norm(out_ch), nn.GELU(),
    )


class ImageUNetDecoder(nn.Module):
    """
    U-Net over the raw SAR image, conditioned on CLIP patch tokens at the
    bottleneck.

    Why this exists: the `_scat` masks are fine-grained *texture / backscatter*
    segmentations (high-scatter vs low-scatter regions that follow the image at
    pixel scale). The token-only decoder is capped at the 16×16 CLIP patch grid,
    which is far too coarse to reproduce that detail — it collapses to a constant
    blob (flat mIoU). A U-Net sees the image at full resolution via skip
    connections, so it can actually trace the high-scatter boundaries, while the
    injected CLIP features still provide ice-type semantic context.
    """

    def __init__(self, image_size: Tuple[int, int] = (512, 512),
                 cond_dim: int = 1024, base: int = 32):
        super().__init__()
        self.image_size = tuple(image_size)

        self.enc1 = _double_conv(3, base)            # full res
        self.enc2 = _double_conv(base, base * 2)     # /2
        self.enc3 = _double_conv(base * 2, base * 4)  # /4
        self.enc4 = _double_conv(base * 4, base * 8)  # /8  (bottleneck)
        self.pool = nn.MaxPool2d(2)

        # CLIP semantic context injected at the bottleneck
        self.cond_proj = nn.Linear(cond_dim, base * 8)

        self.up3  = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = _double_conv(base * 8, base * 4)   # concat(up + enc3)
        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = _double_conv(base * 4, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = _double_conv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

        # Deep supervision: auxiliary mask head at the /4 resolution level.
        # Provides a shorter gradient path to enc3/enc4, which helps the
        # U-Net learn coarse foreground/background structure faster and
        # prevents the bottleneck from being the only error signal.
        self.aux_head = nn.Conv2d(base * 4, 1, 1)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize x to ref's spatial size if an odd input made them differ."""
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
        return x

    def forward(self, image: torch.Tensor,
                cond_tokens: Optional[torch.Tensor] = None,
                return_aux: bool = False):
        """
        Args:
            image:       (B, 3, H, W) — preprocessed SAR tensor (full resolution)
            cond_tokens: (B, N, cond_dim) — CLIP patch tokens (semantic context)
            return_aux:  if True, also return the /4-resolution auxiliary logits
                         for deep supervision (only used during training)

        Returns:
            mask logits (B, 1, H, W) — always
            aux logits  (B, 1, H/4, W/4) — only when return_aux=True
        """
        e1 = self.enc1(image)            # (B, base,   H,   W)
        e2 = self.enc2(self.pool(e1))    # (B, 2base,  H/2, W/2)
        e3 = self.enc3(self.pool(e2))    # (B, 4base,  H/4, W/4)
        e4 = self.enc4(self.pool(e3))    # (B, 8base,  H/8, W/8)

        if cond_tokens is not None:
            B, N, _ = cond_tokens.shape
            s = int(N ** 0.5)
            cond = self.cond_proj(cond_tokens)                 # (B, N, 8base)
            cond = cond.permute(0, 2, 1).reshape(B, -1, s, s)  # (B, 8base, s, s)
            cond = F.interpolate(cond, size=e4.shape[-2:],
                                 mode="bilinear", align_corners=False)
            e4 = e4 + cond                                       # inject semantics

        d3 = self.dec3(torch.cat([self._match(self.up3(e4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        out = self.head(d1)                                     # (B, 1, H, W)

        if out.shape[-2:] != self.image_size:
            out = F.interpolate(out, size=self.image_size,
                                mode="bilinear", align_corners=False)

        if return_aux:
            return out, self.aux_head(d3)   # aux is at /4 resolution
        return out


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

    Two design choices that matter for this dataset:
      * GroupNorm (not BatchNorm) — the T4 path trains at batch_size=2, where
        BatchNorm statistics are pure noise and make the mask output oscillate
        wildly / collapse to a constant. GroupNorm is batch-size independent.
      * Optional `spatial_tokens` skip — the fused tokens are scrambled by 4
        layers of cross-attention against a *generic* text prompt, so they carry
        little image-specific spatial signal. Feeding the raw CLIP patch tokens
        (which preserve spatial layout) gives the decoder something real to
        segment, instead of producing a near-constant blob.
    """

    def __init__(self, in_dim: int, image_size: Tuple[int, int] = (512, 512),
                 spatial_dim: int = 0):
        super().__init__()
        self.image_size = tuple(image_size)
        self.spatial_dim = int(spatial_dim)
        proj_in = in_dim + self.spatial_dim

        # Token → 256-d projection (input to the conv upsampler)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, 256),
            nn.GELU(),
        )

        # Convolutional upsampler (16x16 → 512x512), GroupNorm throughout
        self.upsample = nn.Sequential(
            # 16x16 → 64x64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            _group_norm(128), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            _group_norm(128), nn.GELU(),
            # 64x64 → 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            _group_norm(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            _group_norm(64), nn.GELU(),
            # 128x128 → 256x256
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            _group_norm(32), nn.GELU(),
            # 256x256 → 512x512
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            _group_norm(16), nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, fused_tokens: torch.Tensor,
                spatial_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            fused_tokens:   (B, N, fusion_dim)
            spatial_tokens: (B, N, spatial_dim) — CLIP patch tokens (optional skip)

        Returns:
            masks: (B, 1, H, W) — raw logits (apply sigmoid for probability)
        """
        if self.spatial_dim > 0:
            assert spatial_tokens is not None, (
                "LightweightMaskDecoder built with spatial_dim>0 but no "
                "spatial_tokens were passed to forward()."
            )
            x = torch.cat([fused_tokens, spatial_tokens], dim=-1)
        else:
            x = fused_tokens

        B, N, _ = x.shape
        side = int(N ** 0.5)

        x = self.token_proj(x)                  # (B, N, 256)
        x = x.permute(0, 2, 1)                  # (B, 256, N)
        x = x.reshape(B, 256, side, side)       # (B, 256, side, side)
        masks = self.upsample(x)                 # (B, 1, H, W)

        # Resize to exact image_size
        if masks.shape[-2:] != self.image_size:
            masks = F.interpolate(masks, size=self.image_size,
                                   mode="bilinear", align_corners=False)
        return masks
