"""
models/visual_encoder.py — CLIP ViT-L/14 visual encoder with LoRA SAR adaptation.

LoRA adapters are injected into Q, V, K, Out projection matrices.
The base CLIP weights stay frozen; only LoRA delta weights (≈2.5M params) are trained.
Output: patch-level feature tokens (B, N, D) + CLS token (B, D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPVisionConfig
from peft import LoraConfig, get_peft_model, TaskType


class CLIPSAREncoder(nn.Module):
    """
    CLIP ViT-L/14 visual encoder adapted for SAR imagery via LoRA.

    Forward input:  pixel_values (B, 3, H, W)  — already normalised
    Forward output:
        patch_tokens  (B, N, D)  — N = num patches, D = 1024
        cls_token     (B, D)     — global image representation
        attn_weights  (B, N)     — mean attention weight per patch (for prompt gen)
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        # ── Load pretrained CLIP vision encoder ───────────────────────────────
        print(f"Loading CLIP: {model_cfg.clip_model}")
        self.clip = CLIPVisionModel.from_pretrained(
            model_cfg.clip_model,
            output_attentions=True,
        )

        if model_cfg.clip_freeze:
            for param in self.clip.parameters():
                param.requires_grad = False

        # ── Inject LoRA adapters ───────────────────────────────────────────────
        lora_cfg = LoraConfig(
            r=model_cfg.lora_rank,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            target_modules=model_cfg.lora_target_modules,
            bias="none",
        )
        self.clip = get_peft_model(self.clip, lora_cfg)
        self.clip.print_trainable_parameters()

        self.hidden_dim = model_cfg.clip_hidden_dim  # 1024 for ViT-L/14

        # ── Lightweight feature projection ────────────────────────────────────
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

    # CLIP ViT-L/14 is trained at this fixed resolution
    CLIP_INPUT_SIZE: int = 224

    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: (B, 3, H, W)  — any spatial size (e.g. 512×512)

        Returns:
            patch_tokens: (B, N, D)
            cls_token:    (B, D)
            attn_weights: (B, N)  — last-layer CLS attention mean over heads
        """
        # CLIP ViT-L/14 requires exactly 224×224 input regardless of the
        # pipeline's working resolution. Resize here so the rest of the pipeline
        # can keep using higher-resolution feature maps.
        h, w = pixel_values.shape[-2], pixel_values.shape[-1]
        if h != self.CLIP_INPUT_SIZE or w != self.CLIP_INPUT_SIZE:
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.CLIP_INPUT_SIZE, self.CLIP_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        outputs = self.clip(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True,
        )

        # last_hidden_state: (B, 1+N, D)  — first token is CLS
        hidden = outputs.last_hidden_state
        cls_token = hidden[:, 0, :]           # (B, D)
        patch_tokens = hidden[:, 1:, :]       # (B, N, D)

        # Project
        patch_tokens = self.proj(patch_tokens)
        cls_token = self.proj(cls_token.unsqueeze(1)).squeeze(1)

        # Extract attention weights from last transformer layer, CLS row
        # attentions[-1]: (B, heads, 1+N, 1+N)
        last_attn = outputs.attentions[-1]         # (B, H, 1+N, 1+N)
        cls_attn = last_attn[:, :, 0, 1:]          # (B, H, N) — CLS attending patches
        attn_weights = cls_attn.mean(dim=1)        # (B, N) — average over heads

        return patch_tokens, cls_token, attn_weights

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
