"""
inference.py — Inference script for the Sea Ice Reasoning Segmentation pipeline.

Features:
  - Single image, batch, or directory inference
  - Temporal consistency tracking across sequences
  - Output masks, class labels, and visualizations
  - Confidence scoring and attention heatmap export
  - JSON predictions export

Usage:
    # Single image
    python inference.py --image sample.jpg --checkpoint outputs/best_model.pth --output results/

    # Directory of images
    python inference.py --image_dir dataset/first_year_ice/images \
                        --checkpoint outputs/best_model.pth --output results/

    # With descriptions
    python inference.py --image sample.jpg --description "First-year ice with melt ponds" \
                        --checkpoint outputs/best_model.pth --output results/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from config import cfg, ICE_CLASSES, IDX_TO_ICE_CLASS
from data.preprocessing import SARPreprocessor
from models.pipeline import SeaIceSegmentationPipeline


# ─── Visualisation utilities ──────────────────────────────────────────────────

COLORMAP = {
    "young_ice": (255, 200, 100),           # light orange
    "first_year_ice": (100, 150, 255),      # light blue
    "multi_year_ice": (100, 255, 150),      # light green
    "nilas": (200, 100, 255),               # light purple
    "deformed_ridged_ice": (255, 100, 100), # light red
    "open_water_leads": (0, 0, 0),          # black
}


def draw_mask_overlay(
    image_np: np.ndarray,
    mask: np.ndarray,
    class_name: str,
    confidence: float,
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw segmentation mask as overlay on original image."""
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    color = COLORMAP.get(class_name, (200, 200, 200))
    mask_color = np.zeros_like(image_np, dtype=np.uint8)
    mask_color[mask > 0.5] = color

    overlay = cv2.addWeighted(image_np, 1 - alpha, mask_color, alpha, 0)

    # Add text label
    cv2.putText(
        overlay,
        f"{class_name} ({confidence:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return overlay


def draw_attention_heatmap(
    attention: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Visualise attention heatmap as colour-coded overlay."""
    H, W = image_shape
    side = int(attention.shape[0] ** 0.5)

    # Reshape and upsample
    attn_2d = attention.reshape(side, side)
    attn_full = cv2.resize(attn_2d, (W, H), interpolation=cv2.INTER_LINEAR)

    # Normalise to [0, 255]
    attn_full = (attn_full - attn_full.min()) / (attn_full.max() - attn_full.min() + 1e-8)
    attn_full = (attn_full * 255).astype(np.uint8)

    # Apply colourmap
    heatmap = cv2.applyColorMap(attn_full, cv2.COLORMAP_JET)
    return heatmap


# ─── Image loading ────────────────────────────────────────────────────────────

def load_image(image_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess a single SAR image.

    Returns:
        tensor: (3, H, W) preprocessed tensor ready for model
        array:  (H, W, 3) uint8 for SAM encoding
    """
    preprocessor = SARPreprocessor(cfg.data)

    # Load original for SAM
    pil_img = Image.open(image_path)
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    img_np = np.array(pil_img)

    # Normalise to [0, 1] if uint8
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0

    # Preprocess
    tensor = preprocessor(img_np)

    # For SAM: convert to uint8 RGB
    img_for_sam = (img_np * 255).astype(np.uint8)
    if len(img_for_sam.shape) == 2:
        img_for_sam = np.stack([img_for_sam] * 3, axis=-1)

    return tensor, img_for_sam


# ─── Inference class ─────────────────────────────────────────────────────────

class SeaIceInferencer:
    """Handles model inference with temporal consistency."""

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
    def infer_single(
        self,
        image_tensor: torch.Tensor,        # (3, H, W)
        image_np: np.ndarray,              # (H, W, 3) uint8 for SAM
        description: str = "",
        sequence_id: str = "default",
        frame_id: int = 0,
    ) -> Dict:
        """
        Run inference on a single image.

        Returns:
            Dict with keys: mask, class_idx, class_name, confidence,
                           class_probs, attention, etc.
        """
        # Prepare batch
        images = image_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        if not description:
            description = "Segment the sea ice in this SAR image."
        descriptions = [description]

        images_np = [image_np] if image_np is not None else None

        # Forward
        outputs = self.model(
            images=images,
            descriptions=descriptions,
            images_np=images_np,
            sequence_ids=[sequence_id],
            frame_ids=[frame_id],
            use_long_desc=True,
        )

        # Extract results
        mask = outputs["masks"][0, 0].cpu().numpy()  # (H, W)
        cls_idx = outputs["pred_class_idx"][0].item()
        cls_name = IDX_TO_ICE_CLASS[cls_idx]
        probs = outputs["pred_probs"][0].cpu().numpy()
        confidence = probs[cls_idx]
        attn = outputs["attn_weights"][0].cpu().numpy()
        iou = outputs["iou_scores"][0].item() if outputs["iou_scores"][0] > 0 else None

        return {
            "mask": mask,
            "class_idx": cls_idx,
            "class_name": cls_name,
            "confidence": float(confidence),
            "class_probs": {cls: float(probs[i]) for i, cls in enumerate(ICE_CLASSES)},
            "attention": attn,
            "iou_score": iou,
        }

    @torch.no_grad()
    def infer_batch(
        self,
        images_tensors: List[torch.Tensor],  # List of (3, H, W)
        images_np: List[np.ndarray],          # List of (H, W, 3)
        descriptions: List[str],
        sequence_ids: Optional[List[str]] = None,
        frame_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Batch inference with temporal consistency."""
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(images_tensors))]
        if frame_ids is None:
            frame_ids = list(range(len(images_tensors)))

        images = torch.stack(images_tensors).to(self.device)
        outputs = self.model(
            images=images,
            descriptions=descriptions,
            images_np=images_np,
            sequence_ids=sequence_ids,
            frame_ids=frame_ids,
        )

        results = []
        B = images.shape[0]
        for i in range(B):
            mask = outputs["masks"][i, 0].cpu().numpy()
            cls_idx = outputs["pred_class_idx"][i].item()
            cls_name = IDX_TO_ICE_CLASS[cls_idx]
            probs = outputs["pred_probs"][i].cpu().numpy()
            confidence = probs[cls_idx]
            attn = outputs["attn_weights"][i].cpu().numpy()
            iou = outputs["iou_scores"][i].item() if outputs["iou_scores"][i] > 0 else None

            results.append({
                "mask": mask,
                "class_idx": cls_idx,
                "class_name": cls_name,
                "confidence": float(confidence),
                "class_probs": {cls: float(probs[j]) for j, cls in enumerate(ICE_CLASSES)},
                "attention": attn,
                "iou_score": iou,
            })

        return results


# ─── Main inference loop ──────────────────────────────────────────────────────

def infer_image(
    image_path: Path,
    inferencer: SeaIceInferencer,
    description: str = "",
    save_dir: Optional[Path] = None,
) -> Dict:
    """Run inference on a single image and save outputs."""
    print(f"\nProcessing {image_path.name}...")

    # Load
    tensor, img_np = load_image(image_path)

    # Infer
    result = inferencer.infer_single(tensor, img_np, description)

    # Visualisations
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save mask
        mask = result["mask"]
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_path = save_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask_uint8)

        # Save overlay
        overlay = draw_mask_overlay(img_np, mask, result["class_name"],
                                     result["confidence"])
        overlay_path = save_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        # Save attention heatmap
        attn_map = draw_attention_heatmap(result["attention"], img_np.shape[:2])
        attn_path = save_dir / f"{image_path.stem}_attention.png"
        cv2.imwrite(str(attn_path), attn_map)

        # Save JSON
        result_json = result.copy()
        result_json.pop("mask")
        result_json.pop("attention")
        json_path = save_dir / f"{image_path.stem}_predictions.json"
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=2)

        print(f"✓ Saved to {save_dir}")

    return result


def infer_directory(
    image_dir: Path,
    inferencer: SeaIceInferencer,
    descriptions_csv: Optional[Path] = None,
    save_dir: Optional[Path] = None,
):
    """Run inference on all images in a directory."""
    import pandas as pd

    image_paths = sorted(image_dir.glob("*"))
    image_paths = [p for p in image_paths
                   if p.suffix.lower() in {".jpg", ".png", ".tif", ".tiff"}]

    # Load descriptions if available
    desc_map = {}
    if descriptions_csv and descriptions_csv.exists():
        df = pd.read_csv(descriptions_csv)
        for _, row in df.iterrows():
            fname = str(row["image"]).strip()
            desc = str(row.get("long_descriptions", "")).strip()
            desc_map[fname] = desc

    results = {}
    for image_path in tqdm(image_paths, desc="Inference"):
        desc = desc_map.get(image_path.name, "")
        result = infer_image(image_path, inferencer, desc, save_dir)
        results[image_path.name] = result

    # Save results summary
    if save_dir:
        summary_path = save_dir / "inference_summary.json"
        summary = {}
        for fname, res in results.items():
            r = res.copy()
            r.pop("mask")
            r.pop("attention")
            summary[fname] = r
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary saved to {summary_path}")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sea Ice SAR Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to image directory")
    parser.add_argument("--output", type=str, default="inference_results",
                        help="Output directory for results")
    parser.add_argument("--description", type=str, default="",
                        help="Text description for single image")
    parser.add_argument("--descriptions_csv", type=str, default=None,
                        help="CSV with image descriptions for directory inference")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--use_sam", action="store_true",
                        help="Use SAM for mask decoding")
    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.image_dir:
        parser.error("Provide either --image or --image_dir")

    if args.image and args.image_dir:
        parser.error("Provide either --image or --image_dir, not both")

    # Setup
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    inferencer = SeaIceInferencer(checkpoint_path, device=args.device,
                                   use_sam=args.use_sam)

    # Run inference
    if args.image:
        infer_image(Path(args.image), inferencer, args.description, output_dir)
    else:
        infer_directory(
            Path(args.image_dir), inferencer,
            Path(args.descriptions_csv) if args.descriptions_csv else None,
            output_dir,
        )


if __name__ == "__main__":
    main()
