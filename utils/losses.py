"""
utils/losses.py — Combined loss functions for the sea ice segmentation pipeline.

  L_total = λ_mask * L_mask + λ_aux * L_aux + λ_cls * L_ce + λ_cot * L_attn

L_mask: SoftMaskLoss (BCE + soft Tversky + L1) when the raw `_scat` maps are
        the ground truth (mask_target_mode="soft_scat", default), or
        Focal + Tversky for the legacy Otsu-binary mode.
L_aux:              Auxiliary deep-supervision loss at ¼ resolution — +0.027 mIoU
L_cls:              6-class ice-type cross-entropy (weighted)
L_attn (λ_cot):     Attention-map regularisation — KL divergence on CLIP attention weights.
                    Named lambda_cot for checkpoint compatibility; this is NOT
                    chain-of-thought supervision — no language generation occurs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ICE_CLASS_WEIGHTS


# ─── Dice loss ────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-4):
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


# ─── Focal loss (handles extreme foreground/background imbalance) ─────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss on raw logits. Down-weights easy (background) pixels so
    the rare foreground dominates the gradient — essential when foreground is
    <5% of pixels (the case for this SAR scattering dataset).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * (1.0 - p_t).clamp(min=1e-6) ** self.gamma * ce
        return loss.mean()


# ─── Tversky loss (asymmetric Dice — penalises false negatives more) ──────────

class TverskyLoss(nn.Module):
    """
    Generalised Dice. beta > alpha penalises false negatives (missed foreground)
    harder than false positives, which counteracts the model's tendency to
    collapse to all-background on sparse masks.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten(1)
        target = target.flatten(1)
        tp = (pred * target).sum(dim=1)
        fp = (pred * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - pred) * target).sum(dim=1)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky).mean()


# ─── Combined mask loss (Focal + Tversky) ─────────────────────────────────────

class MaskLoss(nn.Module):
    """
    Segmentation loss = Focal + (balanced) Tversky/Dice.

    IMPORTANT — these `_scat` masks are NOT sparse: Otsu binarisation yields
    ~30%+ foreground. An asymmetric, false-negative-heavy loss (tversky_beta≫
    tversky_alpha) therefore makes "predict everything as foreground" a stable
    minimum: zero false negatives, and IoU pins at the mean foreground fraction
    (~0.32) for every image — exactly the flat-mIoU collapse we observed.

    Balanced settings (focal_alpha=0.5, tversky_alpha=tversky_beta=0.5, i.e.
    Dice) penalise false positives and false negatives equally, so all-foreground
    is no longer free and the network has to actually localise the region.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
        smooth: float = 1e-4,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
    ):
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta, smooth)
        self.focal_w = focal_weight
        self.tversky_w = tversky_weight

    def forward(
        self,
        mask_logits: torch.Tensor,  # (B, 1, H, W) — raw logits
        mask_target: torch.Tensor,  # (B, 1, H, W) — binary mask
    ) -> torch.Tensor:
        focal = self.focal(mask_logits, mask_target)
        prob = torch.sigmoid(mask_logits)
        tversky = self.tversky(prob, mask_target)
        return self.focal_w * focal + self.tversky_w * tversky


# ─── Soft-target mask loss (raw `_scat` maps as ground truth) ─────────────────

class SoftMaskLoss(nn.Module):
    """
    Mask loss for CONTINUOUS [0,1] targets (mask_target_mode="soft_scat",
    where the min-max-normalised `_scat` scattering map is the ground truth).

      L = w_bce * BCEWithLogits(logits, soft)        — pixel-wise regression in
                                                       probability space (BCE
                                                       accepts soft targets)
        + w_tversky * Tversky(sigmoid(logits), soft) — region-overlap signal;
                                                       the soft formulation
                                                       degrades gracefully to
                                                       Dice on binary targets
        + w_l1 * L1(sigmoid(logits), soft)           — sharpens the regression
                                                       toward exact scat values

    Focal loss is intentionally absent: its easy/hard weighting assumes hard
    labels and mis-weights mid-valued soft pixels.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        tversky_weight: float = 1.0,
        l1_weight: float = 0.5,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        smooth: float = 1e-4,
    ):
        super().__init__()
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta, smooth)
        self.bce_w = bce_weight
        self.tversky_w = tversky_weight
        self.l1_w = l1_weight

    def forward(
        self,
        mask_logits: torch.Tensor,  # (B, 1, H, W) — raw logits
        mask_target: torch.Tensor,  # (B, 1, H, W) — soft target in [0, 1]
    ) -> torch.Tensor:
        target = mask_target.clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(mask_logits, target)
        prob = torch.sigmoid(mask_logits)
        tversky = self.tversky(prob, target)
        l1 = F.l1_loss(prob, target)
        return self.bce_w * bce + self.tversky_w * tversky + self.l1_w * l1


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
        # label_smoothing=0.1 prevents logits from growing to FP16 overflow range
        return F.cross_entropy(logits, targets, weight=self.weights,
                               label_smoothing=0.1)


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
        mask_patches = F.adaptive_avg_pool2d(masks.float(), (side, side))  # (B,1,s,s)
        mask_flat = mask_patches.view(B, N)                                  # (B, N)

        # Skip batches where no foreground exists (all-zero mask → NaN distribution)
        foreground_sum = mask_flat.sum(dim=-1)  # (B,)
        valid = foreground_sum > 0              # (B,) bool

        if not valid.any():
            return attn_weights.new_tensor(0.0)

        attn_w = attn_weights[valid]
        mask_f = mask_flat[valid]

        # Normalise to probability distributions; clamp log input to prevent log(0)→NaN
        attn_dist = F.softmax(attn_w, dim=-1).clamp(min=1e-8)   # (B', N)
        mask_dist = mask_f / (mask_f.sum(dim=-1, keepdim=True) + 1e-8)

        # KL(mask_dist || attn_dist) — use log_target=False since mask_dist is a density
        kl = F.kl_div(
            attn_dist.log(), mask_dist,
            reduction="batchmean", log_target=False
        )

        # Guard against any remaining NaN/Inf (e.g. due to FP16 underflow)
        if not torch.isfinite(kl):
            return attn_weights.new_tensor(0.0)

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

        # Mask-loss selection follows the ground-truth definition:
        #   soft_scat — raw `_scat` maps as continuous GT → SoftMaskLoss
        #   binary    — legacy Otsu masks                 → Focal + Tversky
        target_mode = getattr(cfg.data, "mask_target_mode", "soft_scat")
        if target_mode == "soft_scat":
            self.mask_loss = SoftMaskLoss(
                bce_weight=getattr(train_cfg, "soft_bce_weight", 1.0),
                tversky_weight=getattr(train_cfg, "soft_tversky_weight", 1.0),
                l1_weight=getattr(train_cfg, "soft_l1_weight", 0.5),
                tversky_alpha=getattr(train_cfg, "tversky_alpha", 0.5),
                tversky_beta=getattr(train_cfg, "tversky_beta", 0.5),
            )
        else:
            # Loss shape is config-driven so over-segmentation can be tuned
            # without code edits: α>β in Tversky penalises false positives.
            self.mask_loss = MaskLoss(
                focal_alpha=getattr(train_cfg, "focal_alpha", 0.5),
                tversky_alpha=getattr(train_cfg, "tversky_alpha", 0.5),
                tversky_beta=getattr(train_cfg, "tversky_beta", 0.5),
            )
        self.cls_loss = WeightedClassificationLoss()
        self.attn_loss = AttentionGuidanceLoss()

        self.lambda_mask = train_cfg.lambda_mask
        self.lambda_cls = train_cfg.lambda_cls
        self.lambda_cot = train_cfg.lambda_cot
        self.lambda_aux = getattr(train_cfg, "lambda_aux", 0.4)
        self.lambda_presence = getattr(train_cfg, "lambda_presence", 1.0)

    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> dict:
        """
        Args:
            outputs: dict from pipeline forward(), optionally containing
                     "aux_logits" (B,1,H/4,W/4) for deep supervision
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
        is_pos = targets.get("is_positive")  # (B,) float in {0,1} or None

        # Resize prediction to match GT mask size (in case they differ)
        if mask_logits.shape[-2:] != gt_mask.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits, size=gt_mask.shape[-2:],
                mode="bilinear", align_corners=False
            )

        # ── Mask loss on POSITIVE samples only ────────────────────────────────
        # On negatives the target is empty; including them lets the trivial
        # "predict empty everywhere" solution dominate and the decoder collapses
        # (pos_mIoU → 0). The presence head (below) handles negatives instead, so
        # the decoder only ever learns to localise the referred ice.
        if is_pos is not None:
            pos = is_pos.to(mask_logits.device).view(-1) > 0.5
        else:
            pos = torch.ones(mask_logits.shape[0], dtype=torch.bool,
                             device=mask_logits.device)

        if pos.any():
            l_mask = self.mask_loss(mask_logits[pos], gt_mask[pos])
        else:
            l_mask = mask_logits.sum() * 0.0  # no positives in this batch

        l_cls = self.cls_loss(cls_logits_tc, gt_label)
        l_attn = self.attn_loss(attn[pos], gt_mask[pos]) if pos.any() \
            else mask_logits.sum() * 0.0

        total = (self.lambda_mask * l_mask
                 + self.lambda_cls  * l_cls
                 + self.lambda_cot  * l_attn)

        # ── Presence / reasoning loss (BCE on "is the queried ice present?") ──
        l_pres = mask_logits.new_tensor(0.0)
        if outputs.get("presence_logits") is not None and is_pos is not None:
            l_pres = F.binary_cross_entropy_with_logits(
                outputs["presence_logits"].view(-1),
                is_pos.to(mask_logits.device).view(-1).float(),
            )
            total = total + self.lambda_presence * l_pres

        # Deep supervision: aux mask head at 1/4 resolution (positives only)
        l_aux = mask_logits.new_tensor(0.0)
        if "aux_logits" in outputs and outputs["aux_logits"] is not None and pos.any():
            aux_logits = outputs["aux_logits"][pos]
            # Clamp before loss: aux_head is a bare Conv2d with no normalisation.
            aux_logits = aux_logits.clamp(-10.0, 10.0)
            gt_mask_ds = F.interpolate(gt_mask[pos], size=aux_logits.shape[-2:],
                                       mode="bilinear", align_corners=False)
            l_aux = self.mask_loss(aux_logits, gt_mask_ds)
            total = total + self.lambda_aux * self.lambda_mask * l_aux

        return {
            "loss": total,
            "loss_mask": l_mask,
            "loss_cls": l_cls,
            "loss_attn": l_attn,
            "loss_aux": l_aux,
            "loss_presence": l_pres,
        }
