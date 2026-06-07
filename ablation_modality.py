"""
ablation_modality.py — Modality ablation for the Sea Ice pipeline.

Runs the SAME trained checkpoint in three inference-time configurations and
reports segmentation (mIoU, Dice) and classification (weighted F1) on the test
set, producing the modality-ablation table for the paper:

    Variant            mIoU     Dice     Weighted F1
    Full (Image+Text)  ...      ...      ...
    Image Only         ...      ...      ...
    Text Only          ...      ...      ...

The variants are produced WITHOUT retraining (no weights change):
  - Full        — image + text, the published model.
  - Image Only  — text signal zeroed at inference (ablate_text): the projected
                  text tokens (cross-attention K/V) and the sentence embedding
                  are set to zero, so only the SAR image drives the model.
  - Text Only   — image signal zeroed at inference (ablate_image): the visual
                  patch tokens, depth tokens and the image fed to the U-Net are
                  set to zero, so only the textual description drives the model.

This is an inference-time modality-reliance ablation: it measures how much each
output depends on each input modality for the trained model. It is described as
such in the paper (it is not a retrained-variant comparison).

Usage:
    python ablation_modality.py --checkpoint outputs/best_model.pth \
                                --output eval_results/modality_ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from data.dataset import SeaIceDataset
from models.pipeline import SeaIceSegmentationPipeline
from utils.metrics import MetricAccumulator


# Variant name → (ablate_text, ablate_image)
VARIANTS = {
    "Full (Image+Text)": (False, False),
    "Image Only":        (True,  False),
    "Text Only":         (False, True),
}


def _collate(batch):
    """Same collate as evaluate.py: keep text/path fields as lists, stack tensors."""
    list_keys = {"short_desc", "long_desc", "image_path"}
    return {
        k: ([b[k] for b in batch] if k in list_keys
            else torch.stack([b[k] for b in batch]))
        for k in batch[0].keys()
    }


@torch.no_grad()
def run_variant(
    model: SeaIceSegmentationPipeline,
    dataloader: DataLoader,
    device: str,
    ablate_text: bool,
    ablate_image: bool,
    desc: str,
) -> Dict:
    """Evaluate one modality configuration over the full split."""
    acc = MetricAccumulator()

    for batch in tqdm(dataloader, desc=desc):
        images = batch["image"].to(device)
        descriptions = batch["long_desc"]

        outputs = model(
            images=images,
            descriptions=descriptions,
            ablate_text=ablate_text,
            ablate_image=ablate_image,
        )

        acc.update(
            outputs={
                "masks": outputs["masks"],
                "pred_class_idx": outputs["pred_class_idx"],
            },
            targets={
                "mask": batch["mask"].to(device),
                "label": batch["label"].to(device),
            },
            loss=None,
        )

    return acc.compute()


def latex_table(rows: Dict[str, Dict]) -> str:
    """Render the modality-ablation results as an MDPI-style LaTeX table."""
    lines = [
        r"\begin{table}[H]",
        r"\caption{Inference-time modality ablation. The same trained model is "
        r"evaluated with each input modality zeroed at inference (no retraining); "
        r"the comparison quantifies how much each output relies on the SAR image "
        r"versus the textual description.}",
        r"\label{tab:modality_ablation}",
        r"\begin{tabularx}{\textwidth}{lCCC}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{mIoU} & \textbf{Dice} & \textbf{Weighted F1} \\",
        r"\midrule",
    ]
    for name, m in rows.items():
        lines.append(
            f"{name} & {m['mean_iou']:.3f} & {m['mean_dice']:.3f} "
            f"& {m['weighted_f1']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Modality ablation for sea ice pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="eval_results/modality_ablation",
                        help="Output directory for the ablation table/JSON")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading {args.split} dataset...")
    dataset = SeaIceDataset(
        cfg.data.data_root,
        split=args.split,
        data_cfg=cfg.data,
        use_augmentation=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=_collate,
    )

    # ── Model (load once; reused for every variant) ───────────────────────────
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SeaIceSegmentationPipeline(cfg.model, use_sam=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("✓ Model loaded\n")

    # ── Run all variants ──────────────────────────────────────────────────────
    results: Dict[str, Dict] = {}
    for name, (ab_text, ab_img) in VARIANTS.items():
        results[name] = run_variant(
            model, dataloader, device,
            ablate_text=ab_text, ablate_image=ab_img, desc=name,
        )

    # ── Report ────────────────────────────────────────────────────────────────
    header = f"{'Variant':<20}{'mIoU':>10}{'Dice':>10}{'Weighted F1':>14}"
    print("\n" + "=" * len(header))
    print(f"Modality Ablation ({args.split} set)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<20}{m['mean_iou']:>10.3f}{m['mean_dice']:>10.3f}"
              f"{m['weighted_f1']:>14.3f}")
    print("=" * len(header) + "\n")

    # Markdown (easy to paste into docs / GitHub)
    md_lines = [
        "| Variant | mIoU | Dice | Weighted F1 |",
        "|---|---|---|---|",
    ]
    for name, m in results.items():
        md_lines.append(
            f"| {name} | {m['mean_iou']:.3f} | {m['mean_dice']:.3f} "
            f"| {m['weighted_f1']:.3f} |"
        )
    md = "\n".join(md_lines)
    print(md + "\n")

    # ── Save artefacts ────────────────────────────────────────────────────────
    json_path = output_dir / "modality_ablation.json"
    with open(json_path, "w") as f:
        json.dump(
            {name: {"mIoU": m["mean_iou"], "Dice": m["mean_dice"],
                    "weighted_f1": m["weighted_f1"],
                    "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
             for name, m in results.items()},
            f, indent=2,
        )
    (output_dir / "modality_ablation.md").write_text(md + "\n")
    (output_dir / "modality_ablation.tex").write_text(latex_table(results) + "\n")
    print(f"✓ Saved JSON     → {json_path}")
    print(f"✓ Saved Markdown → {output_dir / 'modality_ablation.md'}")
    print(f"✓ Saved LaTeX    → {output_dir / 'modality_ablation.tex'}")


if __name__ == "__main__":
    main()
