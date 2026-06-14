"""
models/ice_classifier.py — Multi-class sea ice type classification head.

Two input regimes (selected by model_cfg.cls_image_only):

  IMAGE-ONLY (default, cls_image_only=True) — HONEST classification.
    Inputs are derived from the SAR image alone: mask-pooled CLIP *patch*
    tokens ‖ CLIP *CLS* token. The per-image description never reaches this
    head, so the 6-class score cannot leak from the text channel (a detailed
    description like "crevassed tongue terminating in open water" otherwise
    identifies the class outright, driving F1 → 1.0 trivially). The text still
    navigates SEGMENTATION via the decoder; only CLASSIFICATION is image-only.

  FUSED (cls_image_only=False) — legacy/ablation. Inputs are the text-fused
    visual tokens ‖ sentence embedding. Reproduces the leaky behaviour.

Architecture:
  [masked_pool(visual_tokens) ‖ global_vec] → LayerNorm → MLP → 6-class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ICE_CLASSES


class IceTypeClassifier(nn.Module):
    """6-class sea ice type classification head (see module docstring)."""

    def __init__(self, model_cfg):
        super().__init__()
        hidden_dim = model_cfg.cls_hidden_dim  # 256
        dropout = model_cfg.cls_dropout        # 0.30
        n_cls = model_cfg.num_classes          # 6

        self.image_only = getattr(model_cfg, "cls_image_only", True)
        if self.image_only:
            # mask-pooled CLIP patch tokens ‖ CLIP CLS token (both image-only)
            visual_dim = model_cfg.clip_hidden_dim   # 1024
            global_dim = model_cfg.clip_hidden_dim   # 1024
        else:
            # text-fused tokens ‖ sentence embedding (leaky; ablation only)
            visual_dim = model_cfg.fusion_dim        # 768
            global_dim = model_cfg.fusion_dim        # 768
        in_dim = visual_dim + global_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_cls),
        )
        self.norm = nn.LayerNorm(in_dim)
        self.class_names = ICE_CLASSES

    def forward(
        self,
        visual_tokens: torch.Tensor,  # (B, N, visual_dim) — patch tokens (image-only) or fused tokens
        masks: torch.Tensor,          # (B, 1, H, W) — binary or probability mask
        global_vec: torch.Tensor,     # (B, global_dim) — CLS token (image-only) or sent_emb
    ) -> torch.Tensor:
        """
        Returns:
            logits (B, num_classes)
        """
        B, N, D = visual_tokens.shape
        side = int(N ** 0.5)

        # ── Pool visual tokens inside the mask region ──────────────────────────
        mask_small = F.adaptive_avg_pool2d(masks, (side, side))  # (B, 1, side, side)
        mask_flat = mask_small.view(B, 1, N)                      # (B, 1, N)
        mask_weight = mask_flat / (mask_flat.sum(dim=-1, keepdim=True) + 1e-8)
        visual_pool = (visual_tokens * mask_weight.permute(0, 2, 1)).sum(dim=1)  # (B, D)

        # ── Concatenate with the global image/text vector ─────────────────────
        combined = torch.cat([visual_pool, global_vec], dim=-1)
        combined = self.norm(combined)

        # ── Classify ──────────────────────────────────────────────────────────
        logits = self.mlp(combined)    # (B, num_classes)
        # Clamp prevents FP16 overflow (max finite: 65504) from poisoning the
        # temporal memory bank and turning cls_loss into NaN.
        return logits.clamp(-20.0, 20.0)

    def predict(self, logits: torch.Tensor):
        """
        Returns:
            class_idx  (B,) int64
            probs      (B, num_classes) float32
            class_name List[str]
        """
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        names = [self.class_names[i.item()] for i in idx]
        return idx, probs, names
