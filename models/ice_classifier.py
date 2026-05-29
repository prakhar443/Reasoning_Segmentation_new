"""
models/ice_classifier.py — Multi-class sea ice type classification head.

Takes pooled visual features (from inside the SAM mask region)
+ sentence-level text/CoT embedding → 6-class softmax prediction.

Architecture:
  [clip_masked_pool ‖ sent_emb] → MLP → 6-class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ICE_CLASSES


class IceTypeClassifier(nn.Module):
    """
    6-class sea ice type classification head.

    Input dimensions:
        visual_dim: dimension of masked-pooled CLIP features (fusion_dim)
        text_dim:   dimension of sentence embedding (fusion_dim)

    Together: 2 × fusion_dim → hidden → 6 classes
    """

    def __init__(self, model_cfg):
        super().__init__()
        fusion_dim = model_cfg.fusion_dim      # 768
        hidden_dim = model_cfg.cls_hidden_dim  # 512
        dropout = model_cfg.cls_dropout        # 0.10
        n_cls = model_cfg.num_classes          # 6

        in_dim = fusion_dim * 2  # [visual ‖ text]

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
        fused_tokens: torch.Tensor,   # (B, N, fusion_dim)
        masks: torch.Tensor,          # (B, 1, H, W) — binary or probability mask
        sent_emb: torch.Tensor,       # (B, fusion_dim)
    ) -> torch.Tensor:
        """
        Returns:
            logits (B, num_classes)
        """
        B, N, D = fused_tokens.shape
        side = int(N ** 0.5)

        # ── Pool visual tokens inside the mask region ──────────────────────────
        # Resize mask to patch grid resolution
        mask_small = F.adaptive_avg_pool2d(masks, (side, side))  # (B, 1, side, side)
        mask_flat = mask_small.view(B, 1, N)                      # (B, 1, N)
        mask_weight = mask_flat / (mask_flat.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted pool
        visual_pool = (fused_tokens * mask_weight.permute(0, 2, 1)).sum(dim=1)  # (B, D)

        # ── Concatenate with text embedding ───────────────────────────────────
        combined = torch.cat([visual_pool, sent_emb], dim=-1)  # (B, 2D)
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
