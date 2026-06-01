"""
utils/metrics.py — Evaluation metrics for the sea ice segmentation pipeline.

Segmentation: mIoU, Dice, pixel accuracy
Classification: macro/weighted F1, per-class accuracy, confusion matrix
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    accuracy_score
)

from config import ICE_CLASSES


# ─── Segmentation metrics ─────────────────────────────────────────────────────

def compute_iou(
    pred: torch.Tensor,   # (B, 1, H, W) probability or binary
    target: torch.Tensor, # (B, 1, H, W) binary
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    pred_bin = (pred >= threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    pred_bin = (pred >= threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    denom = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return dice.mean().item()


def compute_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    pred_bin = (pred >= threshold).float()
    correct = (pred_bin == target).float().sum()
    total = target.numel()
    return (correct / total).item()


# ─── Reasoning-segmentation metrics (LISA-style cIoU / gIoU) ──────────────────
# In the reasoning-segmentation literature (e.g. LISA), two IoU variants are
# reported side by side:
#   gIoU — the MEAN of per-image IoU (every image weighted equally; sensitive
#          to small objects and empty masks).
#   cIoU — CUMULATIVE IoU = (Σ intersections) / (Σ unions) over the whole set
#          (pixel-weighted; large objects dominate). Reported as a single number.
# Both are computed from the same per-image intersection/union tallies.

def compute_intersection_union(
    pred: torch.Tensor,    # (B, 1, H, W) probability
    target: torch.Tensor,  # (B, 1, H, W) binary
    threshold: float = 0.5,
):
    """Return (total_intersection, total_union) summed over the batch — the
    raw tallies cIoU accumulates across the entire dataset."""
    pred_bin = (pred >= threshold).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return inter.item(), union.item()


def compute_giou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """gIoU = mean of per-image IoU (LISA generalized IoU)."""
    pred_bin = (pred >= threshold).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou.mean().item()


def compute_boundary_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    dilation: int = 2,
    smooth: float = 1e-6,
) -> float:
    """Boundary IoU — IoU computed only on the mask boundary band (width
    = `dilation`). Erosion is done with min-pooling (= -maxpool(-x)). Rewards
    sharp, well-localised edges, which pixel IoU can mask."""
    pred_bin = (pred >= threshold).float()

    def _boundary(m):
        k = dilation * 2 + 1
        eroded = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=dilation)
        return (m - eroded).clamp(0.0, 1.0)

    pb, tb = _boundary(pred_bin), _boundary(target)
    inter = (pb * tb).sum(dim=(1, 2, 3))
    union = pb.sum(dim=(1, 2, 3)) + tb.sum(dim=(1, 2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def compute_precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
):
    """Per-image pixel precision & recall, averaged over the batch."""
    pred_bin = (pred >= threshold).float()
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred_bin) * target).sum(dim=(1, 2, 3))
    precision = ((tp + smooth) / (tp + fp + smooth)).mean().item()
    recall = ((tp + smooth) / (tp + fn + smooth)).mean().item()
    return precision, recall


# ─── Classification metrics ───────────────────────────────────────────────────

class ClassificationMetrics:
    """Accumulates predictions and computes metrics at epoch end."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            preds:  (B,) int64 — argmax class predictions
            labels: (B,) int64 — ground truth class indices
        """
        self.all_preds.extend(preds.cpu().tolist())
        self.all_labels.extend(labels.cpu().tolist())

    def compute(self) -> Dict:
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        per_class_f1 = f1_score(
            labels, preds, average=None, zero_division=0,
            labels=list(range(len(ICE_CLASSES)))
        )
        cm = confusion_matrix(labels, preds, labels=list(range(len(ICE_CLASSES))))

        report = classification_report(
            labels, preds,
            target_names=ICE_CLASSES,
            labels=list(range(len(ICE_CLASSES))),
            zero_division=0,
        )

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_f1": {c: float(per_class_f1[i])
                             for i, c in enumerate(ICE_CLASSES)},
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }


# ─── Combined epoch metric accumulator ────────────────────────────────────────

class MetricAccumulator:
    """Tracks all metrics across an epoch."""

    def __init__(self):
        self.cls_metrics = ClassificationMetrics()
        self.seg_ious: List[float] = []
        self.seg_dices: List[float] = []
        self.seg_px_accs: List[float] = []
        self.seg_biou: List[float] = []
        self.seg_prec: List[float] = []
        self.seg_rec: List[float] = []
        # Cumulative tallies for cIoU (Σintersection / Σunion over the set)
        self.cum_inter: float = 0.0
        self.cum_union: float = 0.0
        self.losses: List[float] = []

    def reset(self):
        self.cls_metrics.reset()
        self.seg_ious.clear()
        self.seg_dices.clear()
        self.seg_px_accs.clear()
        self.seg_biou.clear()
        self.seg_prec.clear()
        self.seg_rec.clear()
        self.cum_inter = 0.0
        self.cum_union = 0.0
        self.losses.clear()

    def update(
        self,
        outputs: dict,
        targets: dict,
        loss: Optional[float] = None,
    ):
        # Segmentation
        masks = outputs.get("masks")
        gt_mask = targets.get("mask")
        if masks is not None and gt_mask is not None:
            if masks.shape[-2:] != gt_mask.shape[-2:]:
                masks = F.interpolate(masks, size=gt_mask.shape[-2:],
                                       mode="bilinear", align_corners=False)
            self.seg_ious.append(compute_iou(masks, gt_mask))
            self.seg_dices.append(compute_dice(masks, gt_mask))
            self.seg_px_accs.append(compute_pixel_accuracy(masks, gt_mask))
            self.seg_biou.append(compute_boundary_iou(masks, gt_mask))
            _p, _r = compute_precision_recall(masks, gt_mask)
            self.seg_prec.append(_p)
            self.seg_rec.append(_r)
            _i, _u = compute_intersection_union(masks, gt_mask)
            self.cum_inter += _i
            self.cum_union += _u

        # Classification
        pred_idx = outputs.get("pred_class_idx")
        gt_label = targets.get("label")
        if pred_idx is not None and gt_label is not None:
            self.cls_metrics.update(pred_idx, gt_label)

        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict:
        cls = self.cls_metrics.compute()
        # gIoU = mean per-image IoU (== mean_iou here); cIoU = cumulative
        ciou = (self.cum_inter / self.cum_union) if self.cum_union > 0 else 0.0
        return {
            "mean_iou": float(np.mean(self.seg_ious)) if self.seg_ious else 0.0,
            "giou": float(np.mean(self.seg_ious)) if self.seg_ious else 0.0,
            "ciou": float(ciou),
            "boundary_iou": float(np.mean(self.seg_biou)) if self.seg_biou else 0.0,
            "seg_precision": float(np.mean(self.seg_prec)) if self.seg_prec else 0.0,
            "seg_recall": float(np.mean(self.seg_rec)) if self.seg_rec else 0.0,
            "mean_dice": float(np.mean(self.seg_dices)) if self.seg_dices else 0.0,
            "pixel_accuracy": float(np.mean(self.seg_px_accs)) if self.seg_px_accs else 0.0,
            "mean_loss": float(np.mean(self.losses)) if self.losses else 0.0,
            **cls,
        }

    def summary_str(self) -> str:
        m = self.compute()
        lines = [
            f"Loss:         {m['mean_loss']:.4f}",
            f"mIoU/gIoU:    {m['mean_iou']:.4f}",
            f"cIoU:         {m['ciou']:.4f}",
            f"Boundary IoU: {m['boundary_iou']:.4f}",
            f"Seg P / R:    {m['seg_precision']:.4f} / {m['seg_recall']:.4f}",
            f"Dice:         {m['mean_dice']:.4f}",
            f"Pix Accuracy: {m['pixel_accuracy']:.4f}",
            f"Class Acc:    {m['accuracy']:.4f}",
            f"Macro F1:     {m['macro_f1']:.4f}",
            f"Weighted F1:  {m['weighted_f1']:.4f}",
            "Per-class F1:",
        ]
        for cls_name, f1 in m["per_class_f1"].items():
            lines.append(f"  {cls_name:25s} {f1:.4f}")
        return "\n".join(lines)
