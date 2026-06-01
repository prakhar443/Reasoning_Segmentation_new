"""
utils/losses.py — Combined loss functions for the sea ice segmentation pipeline.

  L_total = λ_mask * (L_bce + L_dice) + λ_cls * L_ce + λ_cot * L_cot

L_mask:  Segmentation mask loss (BCE + Dice)
L_cls:   6-class ice type classification (weighted cross-entropy)
L_cot:   Attention map regularisation (optional CoT proxy supervision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ICE_CLASS_WEIGHTS


# ─── Dice loss ────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 1, H, W) — sigmoid probability
            target: (B, 1, H, W) — binary ground truth
        """
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()


# ─── Combined mask loss (BCE + Dice) ─────────────────────────────────────────

class MaskLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(
        self,
        mask_logits: torch.Tensor,  # (B, 1, H, W) — raw logits
        mask_target: torch.Tensor,  # (B, 1, H, W) — binary mask
    ) -> torch.Tensor:
        bce = self.bce(mask_logits, mask_target)
        prob = torch.sigmoid(mask_logits)
        dice = self.dice(prob, mask_target)
        return self.bce_w * bce + self.dice_w * dice


# ─── Weighted classification loss ─────────────────────────────────────────────

class WeightedClassificationLoss(nn.Module):
    def __init__(self, class_weights=None, device: str = "cpu"):
        super().__init__()
        if class_weights is None:
            class_weights = ICE_CLASS_WEIGHTS
        w = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weights", w)

    def forward(
        self,
        logits: torch.Tensor,   # (B, num_classes)
        targets: torch.Tensor,  # (B,) int64
    ) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights)


# ─── Attention regularisation loss (CoT proxy) ───────────────────────────────

class AttentionGuidanceLoss(nn.Module):
    """
    Encourages the model's attention map to concentrate on the masked region.
    Acts as a lightweight proxy for CoT spatial supervision.

    KL divergence between predicted attention and mask-derived distribution.
    """

    def forward(
        self,
        attn_weights: torch.Tensor,  # (B, N) — model attention over patches
        masks: torch.Tensor,          # (B, 1, H, W) — ground truth mask
    ) -> torch.Tensor:
        B, N = attn_weights.shape
        side = int(N ** 0.5)

        # Downsample mask to patch grid
        mask_patches = F.adaptive_avg_pool2d(masks, (side, side))  # (B,1,s,s)
        mask_flat = mask_patches.view(B, N)                          # (B, N)

        # Normalise to probability distribution
        attn_dist = F.softmax(attn_weights, dim=-1)         # (B, N)
        mask_dist = mask_flat / (mask_flat.sum(dim=-1, keepdim=True) + 1e-8)

        # KL(mask_dist || attn_dist)
        kl = F.kl_div(
            attn_dist.log(), mask_dist,
            reduction="batchmean", log_target=False
        )
        return kl


# ─── Combined pipeline loss ───────────────────────────────────────────────────

class SeaIceLoss(nn.Module):
    """
    Master loss combining all three components with configurable weights.
    """

    def __init__(self, train_cfg=None):
        super().__init__()
        from config import cfg
        train_cfg = train_cfg or cfg.train

        self.mask_loss = MaskLoss()
        self.cls_loss = WeightedClassificationLoss()
        self.attn_loss = AttentionGuidanceLoss()

        self.lambda_mask = train_cfg.lambda_mask
        self.lambda_cls = train_cfg.lambda_cls
        self.lambda_cot = train_cfg.lambda_cot

    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> dict:
        """
        Args:
            outputs: dict from pipeline forward()
            targets: dict with keys:
                mask (B, 1, H, W) float32
                label (B,) int64

        Returns:
            dict with total loss and individual components
        """
        mask_logits = outputs["mask_logits"]
        cls_logits_tc = outputs["cls_logits_tc"]
        attn = outputs["attn_weights"]

        gt_mask = targets["mask"]
        gt_label = targets["label"]

        # Resize prediction to match GT mask size (in case they differ)
        if mask_logits.shape[-2:] != gt_mask.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits, size=gt_mask.shape[-2:],
                mode="bilinear", align_corners=False
            )

        l_mask = self.mask_loss(mask_logits, gt_mask)
        l_cls = self.cls_loss(cls_logits_tc, gt_label)
        l_attn = self.attn_loss(attn, gt_mask)

        total = (self.lambda_mask * l_mask
                 + self.lambda_cls  * l_cls
                 + self.lambda_cot  * l_attn)

        return {
            "loss": total,
            "loss_mask": l_mask,
            "loss_cls": l_cls,
            "loss_attn": l_attn,
        }
