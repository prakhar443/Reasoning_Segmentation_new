"""
evaluate.py — Comprehensive evaluation script for the Sea Ice Segmentation pipeline.

Evaluates on the test set and generates:
  - Segmentation metrics (mIoU, Dice, pixel accuracy)
  - Classification metrics (F1, precision, recall per class)
  - Confusion matrix
  - Per-image detailed results
  - Visualisations (masks, attention heatmaps)
  - Detailed JSON report

Usage:
    # Evaluate best model on test set
    python evaluate.py --checkpoint outputs/best_model.pth --output eval_results/

    # Full evaluation with visualisations
    python evaluate.py --checkpoint outputs/best_model.pth --output eval_results/ \
                       --save_visualizations --save_masks
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import cfg, ICE_CLASSES, IDX_TO_ICE_CLASS
from data.dataset import SeaIceDataset
from data.preprocessing import SARPreprocessor
from models.pipeline import SeaIceSegmentationPipeline
from utils.metrics import (
    compute_iou, compute_dice, compute_pixel_accuracy,
    ClassificationMetrics, MetricAccumulator
)
from inference import draw_mask_overlay, draw_attention_heatmap, COLORMAP


# ─── Evaluation utilities ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """Generate confusion matrix visualisation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved confusion matrix to {save_path}")

    return fig, ax


def plot_metrics_comparison(
    metrics_dict: Dict,
    save_path: Optional[Path] = None,
):
    """Plot per-class F1 scores and IoU."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-class F1
    f1_dict = metrics_dict.get("per_class_f1", {})
    classes = list(f1_dict.keys())
    f1_scores = list(f1_dict.values())
    axes[0].bar(range(len(classes)), f1_scores, color="steelblue")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(classes, rotation=45, ha="right")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Per-Class F1 Scores")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)

    # Segmentation metrics
    seg_names = ["mIoU", "Dice", "Pixel Acc"]
    seg_vals = [
        metrics_dict.get("mean_iou", 0),
        metrics_dict.get("mean_dice", 0),
        metrics_dict.get("pixel_accuracy", 0),
    ]
    axes[1].bar(seg_names, seg_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_ylabel("Score")
    axes[1].set_title("Segmentation Metrics")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved metrics plot to {save_path}")

    return fig, axes


def save_detailed_results(
    results: Dict,
    output_dir: Path,
):
    """Save detailed per-image results as JSON and CSV."""
    # JSON format
    json_path = output_dir / "detailed_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved detailed results to {json_path}")

    # CSV format (easier for spreadsheet review)
    csv_rows = []
    for fname, res in results.items():
        row = {
            "image": fname,
            "true_class": res["true_class"],
            "pred_class": res["pred_class"],
            "pred_confidence": res["pred_confidence"],
            "class_match": "✓" if res["true_class"] == res["pred_class"] else "✗",
            "iou": res.get("iou", None),
            "dice": res.get("dice", None),
            "pixel_acc": res.get("pixel_accuracy", None),
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    csv_path = output_dir / "detailed_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV results to {csv_path}")


# ─── Main evaluation ──────────────────────────────────────────────────────────

class Evaluator:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_sam: bool = True,
    ):
        self.device = device
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)

        self.model = SeaIceSegmentationPipeline(cfg.model, use_sam=use_sam).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print("✓ Model loaded")

        self.preprocessor = SARPreprocessor(cfg.data)

    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: Dict,
    ) -> Dict:
        """Evaluate a single batch."""
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)
        descriptions = batch["long_desc"]
        image_paths = batch["image_path"]

        # SAM requires uint8 numpy arrays
        images_np = [
            (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for img in images
        ]

        # Forward — sequence_id = ice-class folder name (.../<class>/images/<file>),
        # not the literal "images" dir that p.split('/')[-2] returned.
        outputs = self.model(
            images=images,
            descriptions=descriptions,
            images_np=images_np,
            sequence_ids=[str(Path(p).parent.parent.name) for p in image_paths],
        )

        # Compute metrics
        results = []
        B = images.shape[0]

        for i in range(B):
            fname = Path(image_paths[i]).name

            # Segmentation metrics
            pred_mask = outputs["masks"][i : i + 1, :, :, :]
            gt_mask = masks[i : i + 1, :, :, :]

            if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=gt_mask.shape[-2:],
                    mode="bilinear", align_corners=False
                )

            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)
            px_acc = compute_pixel_accuracy(pred_mask, gt_mask)

            # Classification
            pred_idx = outputs["pred_class_idx"][i].item()
            pred_name = IDX_TO_ICE_CLASS[pred_idx]
            true_idx = labels[i].item()
            true_name = IDX_TO_ICE_CLASS[true_idx]
            confidence = outputs["pred_probs"][i, pred_idx].item()

            results.append({
                "image_path": image_paths[i],
                "filename": fname,
                "true_class_idx": int(true_idx),
                "true_class": true_name,
                "pred_class_idx": int(pred_idx),
                "pred_class": pred_name,
                "pred_confidence": float(confidence),
                "iou": float(iou),
                "dice": float(dice),
                "pixel_accuracy": float(px_acc),
                "all_class_probs": {
                    c: float(outputs["pred_probs"][i, j].item())
                    for j, c in enumerate(ICE_CLASSES)
                },
            })

        return results, outputs


def evaluate(
    model_checkpoint: Path,
    data_cfg=None,
    output_dir: Optional[Path] = None,
    split: str = "test",
    save_visualizations: bool = False,
    save_masks: bool = False,
    use_sam: bool = True,
):
    """Run comprehensive evaluation on test/val set."""
    data_cfg = data_cfg or cfg.data
    output_dir = output_dir or Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.train.device

    # Load dataset
    print(f"Loading {split} dataset...")
    dataset = SeaIceDataset(
        data_cfg.data_root,
        split=split,
        data_cfg=data_cfg,
        use_augmentation=False,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: {k: [b[k] for b in x] if k in {"short_desc", "long_desc", "image_path"} else torch.stack([b[k] for b in x]) for k in x[0].keys()},
    )

    # Evaluator
    evaluator = Evaluator(model_checkpoint, device=device, use_sam=use_sam)

    # Metrics
    seg_metrics = MetricAccumulator()
    cls_metrics = ClassificationMetrics()
    all_detailed = {}

    # Run evaluation
    print(f"\nEvaluating on {split} set ({len(dataset)} images)...")
    for batch in tqdm(dataloader, desc="Evaluation"):
        results, outputs = evaluator.evaluate_batch(batch)

        for res in results:
            fname = res["filename"]
            all_detailed[fname] = res

            # Update metrics
            seg_metrics.update(
                outputs={
                    "masks": outputs["masks"],
                },
                targets={"mask": batch["mask"][0].unsqueeze(0)},
                loss=None,
            )
            cls_metrics.update(
                outputs["pred_class_idx"],
                batch["label"],
            )

            # Visualisations
            if save_visualizations or save_masks:
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(exist_ok=True)

                image_np = batch["image"][0].permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)

                pred_mask = outputs["masks"][0, 0].cpu().numpy()
                pred_class = res["pred_class"]
                confidence = res["pred_confidence"]

                # Mask
                if save_masks:
                    mask_uint8 = (pred_mask * 255).astype(np.uint8)
                    mask_path = vis_dir / f"{fname[:-4]}_mask.png"
                    cv2.imwrite(str(mask_path), mask_uint8)

                # Overlay
                if save_visualizations:
                    overlay = draw_mask_overlay(image_np, pred_mask, pred_class,
                                                 confidence, alpha=0.4)
                    overlay_path = vis_dir / f"{fname[:-4]}_overlay.png"
                    cv2.imwrite(str(overlay_path), overlay)

                    # Attention heatmap
                    attn = outputs["attn_weights"][0].cpu().numpy()
                    heatmap = draw_attention_heatmap(attn, image_np.shape[:2])
                    attn_path = vis_dir / f"{fname[:-4]}_attention.png"
                    cv2.imwrite(str(attn_path), heatmap)

    # Compute final metrics
    print("\n" + "=" * 60)
    print(f"Evaluation Results ({split} set)")
    print("=" * 60)

    seg_results = seg_metrics.compute()
    cls_results = cls_metrics.compute()

    # Print summary
    print("\n[SEGMENTATION METRICS]")
    print(f"  mIoU:         {seg_results['mean_iou']:.4f}")
    print(f"  Dice:         {seg_results['mean_dice']:.4f}")
    print(f"  Pixel Acc:    {seg_results['pixel_accuracy']:.4f}")

    print("\n[CLASSIFICATION METRICS]")
    print(f"  Accuracy:     {cls_results['accuracy']:.4f}")
    print(f"  Macro F1:     {cls_results['macro_f1']:.4f}")
    print(f"  Weighted F1:  {cls_results['weighted_f1']:.4f}")

    print("\n[PER-CLASS F1 SCORES]")
    for cls_name, f1 in cls_results["per_class_f1"].items():
        print(f"  {cls_name:25s} {f1:.4f}")

    print(f"\n{cls_results['classification_report']}")

    # Confusion matrix
    print("\n[CONFUSION MATRIX]")
    cm = np.array(cls_results["confusion_matrix"])
    print(cm)

    # Save results
    combined_metrics = {**seg_results, **cls_results}
    combined_metrics["confusion_matrix"] = cls_results["confusion_matrix"]

    # Save JSON report
    report_path = output_dir / f"{split}_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"\n✓ Saved evaluation report to {report_path}")

    # Save detailed results
    save_detailed_results(all_detailed, output_dir)

    # Plot confusion matrix
    cm_plot_path = output_dir / f"{split}_confusion_matrix.png"
    plot_confusion_matrix(cm, ICE_CLASSES, cm_plot_path)

    # Plot metrics
    metrics_plot_path = output_dir / f"{split}_metrics.png"
    plot_metrics_comparison(combined_metrics, metrics_plot_path)

    print("=" * 60 + "\n")

    return combined_metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sea ice segmentation model"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="Output directory for evaluation results")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save mask overlays and attention heatmaps")
    parser.add_argument("--save_masks", action="store_true",
                        help="Save predicted masks")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--use_sam", action="store_true",
                        help="Use SAM for mask decoding")
    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate(
        model_checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output),
        split=args.split,
        save_visualizations=args.save_visualizations,
        save_masks=args.save_masks,
        use_sam=args.use_sam,
    )

    print(f"\n📊 Evaluation complete!")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
