"""
models/prompt_generator.py — Converts CoT attention heatmaps into SAM-compatible prompts.

Process:
  1. Reshape flat attention weights → 2D spatial map
  2. Threshold + connected components → candidate regions
  3. Extract bounding boxes + foreground point clicks per region
  4. Return SAM-compatible prompt dicts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


class GeometricPromptGenerator(nn.Module):
    """
    Converts per-patch attention weights from the reasoning module into
    SAM bounding-box and point prompts.

    The attention map represents where the model is 'looking' in response
    to the ice-type description — high attention → likely target region.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.threshold = model_cfg.attn_threshold
        self.max_prompts = model_cfg.max_prompts_per_image
        self.min_area = model_cfg.prompt_min_area

    def forward(
        self,
        attn_weights: torch.Tensor,  # (B, N) flat patch attention
        image_size: Tuple[int, int], # (H, W) of the original image
    ) -> List[List[Dict]]:
        """
        Args:
            attn_weights: (B, N) — normalised attention weights per patch.
            image_size:   (H, W) of the image SAM will receive.

        Returns:
            List of length B. Each element is a list of prompt dicts:
            [{"boxes": Tensor(1,4), "point_coords": Tensor(1,K,2),
              "point_labels": Tensor(1,K)}, ...]
        """
        B = attn_weights.shape[0]
        results = []

        for b in range(B):
            prompts = self._generate_single(attn_weights[b], image_size)
            results.append(prompts)

        return results

    def _generate_single(
        self,
        attn: torch.Tensor,        # (N,)
        image_size: Tuple[int, int],
    ) -> List[Dict]:
        """Generate SAM prompts for a single image."""
        N = attn.shape[0]
        side = int(N ** 0.5)       # assume square patch grid
        H, W = image_size

        # ── 1. Normalise and reshape to 2D heatmap ────────────────────────────
        attn_np = attn.detach().cpu().float().numpy()
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        heatmap = attn_np.reshape(side, side)

        # ── 2. Upsample to image resolution ───────────────────────────────────
        heatmap_full = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

        # ── 3. Threshold → binary mask ────────────────────────────────────────
        thresh = (heatmap_full >= self.threshold).astype(np.uint8)

        # ── 4. Morphological cleanup ──────────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # ── 5. Connected components ───────────────────────────────────────────
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        prompts = []
        # Sort components by area (descending), skip background (label 0)
        components = sorted(
            range(1, n_labels),
            key=lambda i: stats[i, cv2.CC_STAT_AREA],
            reverse=True,
        )

        for comp_idx in components[: self.max_prompts]:
            area = stats[comp_idx, cv2.CC_STAT_AREA]
            if area < self.min_area:
                break

            # Bounding box
            x = stats[comp_idx, cv2.CC_STAT_LEFT]
            y = stats[comp_idx, cv2.CC_STAT_TOP]
            w = stats[comp_idx, cv2.CC_STAT_WIDTH]
            h = stats[comp_idx, cv2.CC_STAT_HEIGHT]

            box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)

            # Foreground points: centroid + top-3 attention peaks inside component
            comp_mask = (labels == comp_idx)
            fg_points = self._sample_fg_points(heatmap_full, comp_mask, k=3)
            fg_labels = torch.ones(1, len(fg_points), dtype=torch.int64)

            prompts.append({
                "boxes": box,                                            # (1, 4) xyxy
                "point_coords": torch.tensor([fg_points], dtype=torch.float32),  # (1, K, 2)
                "point_labels": fg_labels,                               # (1, K)
                "heatmap_region": heatmap_full * comp_mask.astype(np.float32),
            })

        # Fallback: if no region found, use full-image centre point
        if not prompts:
            cx, cy = W // 2, H // 2
            prompts.append({
                "boxes": torch.tensor([[0, 0, W, H]], dtype=torch.float32),
                "point_coords": torch.tensor([[[cx, cy]]], dtype=torch.float32),
                "point_labels": torch.ones(1, 1, dtype=torch.int64),
                "heatmap_region": heatmap_full,
            })

        return prompts

    @staticmethod
    def _sample_fg_points(
        heatmap: np.ndarray,
        mask: np.ndarray,
        k: int = 3,
    ) -> List[List[float]]:
        """Return the k highest-attention pixel coordinates inside the mask."""
        masked = heatmap * mask.astype(np.float32)
        flat = masked.flatten()
        top_k = np.argsort(flat)[::-1][:k]
        H, W = heatmap.shape
        points = [[float(idx % W), float(idx // W)] for idx in top_k]
        return points


# ─── Utility: visualise prompt on image ───────────────────────────────────────

def draw_prompts_on_image(
    image_np: np.ndarray,         # (H, W, 3) uint8
    prompts: List[Dict],
) -> np.ndarray:
    """Debug helper: overlay prompt boxes and points on an image."""
    vis = image_np.copy()
    for prompt in prompts:
        box = prompt["boxes"][0].long().numpy()     # (4,)
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        pts = prompt["point_coords"][0].numpy()     # (K, 2)
        for pt in pts:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), 6, (255, 0, 0), -1)

    return vis
