"""
models/pipeline.py — Full end-to-end sea ice reasoning segmentation pipeline.

Integrates all 8 modules in order:
  1. SARPreprocessor         (data/preprocessing.py)
  2. CLIPSAREncoder          (models/visual_encoder.py)
  3. DepthAnyV2Encoder       (models/depth_encoder.py)
  4. CrossAttentionReasoning (models/reasoning_module.py)
  5. GeometricPromptGen      (models/prompt_generator.py)
  6. SAMModule / LightweightDecoder (models/sam_module.py)
  7. IceTypeClassifier       (models/ice_classifier.py)
  8. TemporalConsistency     (models/temporal_consistency.py)

Forward returns a dict with all intermediate and final outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import cfg as default_cfg, ICE_CLASSES
from data.preprocessing import SARPreprocessor
from models.visual_encoder import CLIPSAREncoder
from models.depth_encoder import build_depth_encoder
from models.reasoning_module import build_reasoning_module
from models.prompt_generator import GeometricPromptGenerator
from models.sam_module import SAMModule, LightweightMaskDecoder
from models.ice_classifier import IceTypeClassifier
from models.temporal_consistency import TemporalConsistencyModule


class SeaIceSegmentationPipeline(nn.Module):
    """
    Full sea ice reasoning segmentation pipeline.

    Modes:
        use_sam=True   — use SAM for mask decoding (best accuracy, slower)
        use_sam=False  — use lightweight convolutional decoder (faster, less accurate)
    """

    def __init__(self, model_cfg=None, use_sam: bool = True):
        super().__init__()
        model_cfg = model_cfg or default_cfg.model
        self.model_cfg = model_cfg
        self.use_sam = use_sam

        print("=" * 60)
        print("Initialising Sea Ice Segmentation Pipeline")
        print("=" * 60)

        # ── Module 1 (not nn.Module — used in data pipeline) ──────────────────
        # SARPreprocessor is applied in the DataLoader; images arrive pre-processed

        # ── Module 2: Visual Encoder ──────────────────────────────────────────
        print("[2/8] Loading CLIP visual encoder + LoRA...")
        self.visual_encoder = CLIPSAREncoder(model_cfg)

        # ── Module 3: Depth Encoder ───────────────────────────────────────────
        print("[3/8] Loading DepthAnything V2 encoder...")
        self.depth_encoder = build_depth_encoder(model_cfg)

        # ── Module 4: Reasoning Module ────────────────────────────────────────
        print("[4/8] Building reasoning module...")
        self.reasoning_module = build_reasoning_module(model_cfg)

        # ── Module 5: Prompt Generator ────────────────────────────────────────
        print("[5/8] Building geometric prompt generator...")
        self.prompt_generator = GeometricPromptGenerator(model_cfg)

        # ── Module 6: SAM or lightweight decoder ──────────────────────────────
        if use_sam:
            print("[6/8] Loading SAM mask decoder...")
            try:
                self.mask_decoder = SAMModule(model_cfg)
            except (FileNotFoundError, RuntimeError, Exception) as e:
                # FileNotFoundError → checkpoint missing
                # RuntimeError      → checkpoint corrupt (bad/partial download)
                print(f"[WARN] Could not load SAM ({type(e).__name__}: {e})\n"
                      f"Falling back to lightweight convolutional decoder.")
                self.mask_decoder = LightweightMaskDecoder(
                    in_dim=model_cfg.fusion_dim,
                    image_size=default_cfg.data.image_size,
                )
                self.use_sam = False
        else:
            print("[6/8] Using lightweight convolutional mask decoder...")
            self.mask_decoder = LightweightMaskDecoder(
                in_dim=model_cfg.fusion_dim,
                image_size=default_cfg.data.image_size,
            )

        # ── Module 7: 6-class classification head ─────────────────────────────
        print("[7/8] Building ice type classification head...")
        self.classifier = IceTypeClassifier(model_cfg)

        # ── Module 8: Temporal consistency ────────────────────────────────────
        print("[8/8] Building temporal consistency module...")
        self.temporal = TemporalConsistencyModule(model_cfg)

        print("=" * 60)
        print(f"Pipeline ready | SAM={self.use_sam}")
        print(f"Trainable parameters: {self._count_trainable():,}")
        print("=" * 60)

    def forward(
        self,
        images: torch.Tensor,              # (B, 3, H, W) pre-processed SAR tensors
        descriptions: List[str],           # length B — short OR long descriptions
        images_np: Optional[List[np.ndarray]] = None,  # needed for SAM mode
        sequence_ids: Optional[List[str]] = None,
        frame_ids: Optional[List[int]] = None,
        use_long_desc: bool = True,        # toggle short/long at inference
    ) -> Dict:
        """
        Full pipeline forward pass.

        Returns dict with keys:
            mask_logits     (B, 1, H, W) — raw mask logits (pre-sigmoid)
            cls_logits      (B, 6) — raw classification logits
            cls_logits_tc   (B, 6) — temporally-smoothed classification logits
            attn_weights    (B, N) — reasoning attention heatmap
            sent_emb        (B, D) — text embedding
            fused_tokens    (B, N, D) — fused visual+text tokens
            pred_class_idx  (B,) — argmax class index
            pred_class_name List[str] — class names
            pred_probs      (B, 6) — softmax probabilities
            iou_scores      (B,) — SAM IoU confidence (if SAM mode)
        """
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        device = images.device

        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(B)]

        # ── Step 2: Visual encoding ────────────────────────────────────────────
        patch_tokens, cls_token, attn_weights = self.visual_encoder(images)
        # patch_tokens: (B, N, clip_dim)
        # attn_weights: (B, N)
        N = patch_tokens.shape[1]

        # ── Step 3: Depth encoding ────────────────────────────────────────────
        depth_tokens = self.depth_encoder(images, target_seq_len=N)
        # depth_tokens: (B, N, depth_dim)

        # ── Step 4: Reasoning + fusion ────────────────────────────────────────
        fused_tokens, reasoning_attn, sent_emb = self.reasoning_module(
            patch_tokens, depth_tokens, descriptions
        )
        # fused_tokens: (B, N, fusion_dim)
        # reasoning_attn: (B, N)
        # sent_emb: (B, fusion_dim)

        # Combine CLIP attention and reasoning attention for prompts
        if reasoning_attn is not None:
            combined_attn = 0.5 * attn_weights + 0.5 * reasoning_attn
        else:
            combined_attn = attn_weights

        # ── Step 5: Geometric prompt generation ───────────────────────────────
        # (only for SAM mode — prompts need pixel coordinates)
        if self.use_sam and images_np is not None:
            prompts_batch = self.prompt_generator(
                combined_attn, image_size=(H, W)
            )
        else:
            prompts_batch = None

        # ── Step 6: Mask decoding ─────────────────────────────────────────────
        iou_scores = torch.zeros(B, device=device)

        if self.use_sam and images_np is not None and prompts_batch is not None:
            # SAM path (returns float binary masks)
            masks_hw, iou_scores = self.mask_decoder(images_np, prompts_batch)
            masks_hw = masks_hw.to(device)
            mask_logits = masks_hw  # already binary; no sigmoid needed for loss
        else:
            # Lightweight decoder path (returns logits)
            mask_logits = self.mask_decoder(fused_tokens)  # (B, 1, H, W)
            masks_hw = torch.sigmoid(mask_logits)

        # ── Step 7: Ice type classification ───────────────────────────────────
        cls_logits = self.classifier(fused_tokens, masks_hw, sent_emb)
        # cls_logits: (B, 6)

        # ── Step 8: Temporal consistency ──────────────────────────────────────
        # Compute mask embedding for temporal comparison
        mask_emb = self._pool_mask_features(fused_tokens, masks_hw)  # (B, D)

        cls_logits_tc = self.temporal(
            cls_logits, mask_emb, sequence_ids, frame_ids
        )

        # ── Final predictions ─────────────────────────────────────────────────
        pred_idx, pred_probs, pred_names = self.classifier.predict(cls_logits_tc)

        return {
            "mask_logits": mask_logits,          # (B, 1, H, W)
            "masks": masks_hw,                    # (B, 1, H, W) — sigmoid applied
            "cls_logits": cls_logits,             # (B, 6) — pre-temporal
            "cls_logits_tc": cls_logits_tc,       # (B, 6) — post-temporal smoothing
            "attn_weights": combined_attn,        # (B, N)
            "sent_emb": sent_emb,                 # (B, D)
            "fused_tokens": fused_tokens,         # (B, N, D)
            "pred_class_idx": pred_idx,           # (B,)
            "pred_class_name": pred_names,        # List[str]
            "pred_probs": pred_probs,             # (B, 6)
            "iou_scores": iou_scores,             # (B,)
        }

    @staticmethod
    def _pool_mask_features(
        fused_tokens: torch.Tensor,   # (B, N, D)
        masks: torch.Tensor,          # (B, 1, H, W)
    ) -> torch.Tensor:
        """Mask-pooled embedding for temporal comparison."""
        B, N, D = fused_tokens.shape
        side = int(N ** 0.5)
        m = F.adaptive_avg_pool2d(masks, (side, side)).view(B, 1, N)
        w = m / (m.sum(dim=-1, keepdim=True) + 1e-8)
        return (fused_tokens * w.permute(0, 2, 1)).sum(dim=1)  # (B, D)

    def _count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_groups(self, model_cfg=None, train_cfg=None):
        """
        Return parameter groups with different learning rates for the optimizer.
        LoRA adapters use higher LR than the classification head.
        """
        model_cfg = model_cfg or self.model_cfg
        train_cfg = train_cfg or default_cfg.train

        lora_params = list(self.visual_encoder.get_trainable_params())
        cls_params = list(self.classifier.parameters())
        temporal_params = list(self.temporal.parameters())
        reasoning_params = [
            p for p in self.reasoning_module.parameters() if p.requires_grad
        ]
        decoder_params = [
            p for p in self.mask_decoder.parameters() if p.requires_grad
        ]

        return [
            {"params": lora_params,      "lr": train_cfg.lora_lr},
            {"params": cls_params,       "lr": train_cfg.cls_head_lr},
            {"params": temporal_params,  "lr": train_cfg.lr},
            {"params": reasoning_params, "lr": train_cfg.lr},
            {"params": decoder_params,   "lr": train_cfg.lr},
        ]
