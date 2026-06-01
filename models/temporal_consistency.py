"""
models/temporal_consistency.py — Temporal consistency module for sequential sea ice images.

⚠️  NOT USED IN PUBLISHED RESULTS (temporal_mode=False by default). Evaluated
    and excluded: no measurable benefit on the single-scene evaluation used in
    the paper. Retained for future multi-temporal work; not a claimed
    contribution. See DOCUMENTATION/PAPER_SCOPE.md and DOCUMENTATION/CODE_MAP.md.

Maintains a feature memory bank of recent frames and enforces mask continuity
via cosine similarity gating. Prevents spurious frame-to-frame class flips.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Tuple


class TemporalMemoryBank:
    """
    Rolling feature memory bank storing the N most recent frame representations.
    Each entry stores:
        - visual_features: (N_patches, fusion_dim) — mean patch features
        - mask_embedding:  (fusion_dim,)           — mask-pooled embedding
        - cls_logits:      (num_classes,)           — class prediction
        - frame_id:        int                      — frame index
    """

    def __init__(self, bank_size: int = 5, fusion_dim: int = 768):
        self.bank_size = bank_size
        self.fusion_dim = fusion_dim
        self.bank: deque = deque(maxlen=bank_size)

    def push(self, mask_embedding: torch.Tensor, cls_logits: torch.Tensor,
             frame_id: int = -1):
        """Store current frame's mask embedding and prediction.
        Silently drops entries containing NaN/Inf to prevent bank poisoning.
        """
        if not torch.isfinite(cls_logits).all():
            return
        if not torch.isfinite(mask_embedding).all():
            return
        self.bank.append({
            "mask_embedding": mask_embedding.detach().cpu(),
            "cls_logits": cls_logits.detach().cpu(),
            "frame_id": frame_id,
        })

    def get_smoothed_logits(
        self,
        current_logits: torch.Tensor,   # (num_classes,)
        current_embedding: torch.Tensor, # (fusion_dim,)
        sim_threshold: float = 0.65,
        blend_alpha: float = 0.70,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compare current embedding to bank. If similarity is below threshold,
        blend current prediction with bank mean.

        Returns:
            smoothed_logits  (num_classes,)
            mean_similarity  float — for logging
        """
        if len(self.bank) == 0:
            return current_logits, 1.0

        # Cosine similarity with each bank entry; skip any corrupted entries
        sims = []
        prev_logits = []
        for entry in self.bank:
            emb = entry["mask_embedding"].to(current_embedding.device)
            lg  = entry["cls_logits"].to(current_logits.device)
            if not torch.isfinite(emb).all() or not torch.isfinite(lg).all():
                continue
            sim = F.cosine_similarity(
                current_embedding.unsqueeze(0), emb.unsqueeze(0)
            ).item()
            sims.append(sim)
            prev_logits.append(lg)
        if not sims:
            return current_logits, 1.0

        mean_sim = float(sum(sims) / len(sims))

        if mean_sim >= sim_threshold:
            return current_logits, mean_sim

        # Blend current with exponentially-weighted bank mean
        weights = torch.softmax(torch.tensor(sims), dim=0)
        bank_mean = sum(w * l for w, l in zip(weights, prev_logits))

        smoothed = blend_alpha * current_logits + (1 - blend_alpha) * bank_mean
        return smoothed, mean_sim

    def clear(self):
        self.bank.clear()

    def __len__(self):
        return len(self.bank)


class TemporalConsistencyModule(nn.Module):
    """
    Wraps the memory bank with a learnable gating module.

    At training time: uses ground-truth sequence ordering (if available).
    At inference time: processes image sequences in order.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.bank_size = model_cfg.memory_bank_size
        self.sim_threshold = model_cfg.temporal_sim_threshold
        self.blend_alpha = model_cfg.temporal_blend_alpha
        self.fusion_dim = model_cfg.fusion_dim

        # Learnable temporal gate: decides how much to blend
        self.gate = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Per-sequence memory banks (keyed by sequence_id or image path prefix)
        self._banks: dict = {}

    def get_bank(self, sequence_id: str) -> TemporalMemoryBank:
        if sequence_id not in self._banks:
            self._banks[sequence_id] = TemporalMemoryBank(
                self.bank_size, self.fusion_dim
            )
        return self._banks[sequence_id]

    def forward(
        self,
        cls_logits: torch.Tensor,      # (B, num_classes)
        mask_embeddings: torch.Tensor, # (B, fusion_dim)
        sequence_ids: list,            # List[str] — one per batch item
        frame_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Apply temporal smoothing per sequence.

        Returns:
            smoothed_logits (B, num_classes)
        """
        B = cls_logits.shape[0]
        smoothed = []

        for i in range(B):
            seq_id = sequence_ids[i]
            bank = self.get_bank(seq_id)
            frame_id = frame_ids[i] if frame_ids else -1

            cur_logits = cls_logits[i]          # (num_classes,)
            cur_emb = mask_embeddings[i]         # (fusion_dim,)

            # Learned gate if bank has valid entries
            last_valid = next(
                (e for e in reversed(bank.bank)
                 if torch.isfinite(e["cls_logits"]).all()
                 and torch.isfinite(e["mask_embedding"]).all()),
                None,
            )
            if last_valid is not None:
                prev_emb = last_valid["mask_embedding"].to(cur_emb.device)
                prev_logits = last_valid["cls_logits"].to(cur_logits.device)
                gate_input = torch.cat([cur_emb, prev_emb], dim=-1)
                gate_val = self.gate(gate_input.unsqueeze(0)).squeeze()
                smooth_logits = gate_val * cur_logits + (1 - gate_val) * prev_logits
            else:
                smooth_logits = cur_logits

            # Also apply cosine-sim gating as a safety check
            smooth_logits, sim = bank.get_smoothed_logits(
                smooth_logits, cur_emb, self.sim_threshold, self.blend_alpha
            )

            # Push to memory bank
            bank.push(cur_emb, smooth_logits, frame_id)
            smoothed.append(smooth_logits)

        return torch.stack(smoothed)   # (B, num_classes)

    def reset_sequence(self, sequence_id: str):
        """Call this when starting a new image sequence."""
        if sequence_id in self._banks:
            self._banks[sequence_id].clear()

    def reset_all(self):
        self._banks.clear()
