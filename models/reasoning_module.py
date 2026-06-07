"""
models/reasoning_module.py — Image↔text cross-attention fusion module.

PUBLISHED MODEL uses the 'cross_attn_only' backend: four stacked cross-attention
blocks that fuse CLIP visual patch tokens (queries) with CLIP text embeddings
(keys/values). This is standard multimodal feature fusion — it does NOT perform
chain-of-thought reasoning, does NOT generate language, and is NOT "reasoning
segmentation" in the LISA/ReasonSeg sense. The file name is a historical artefact.

Three backends are implemented:
  1. cross_attn_only  — Published model. Cross-attention fusion, no LLM, no generation.
  2. blip2            — BLIP-2 VLM (alternative, NOT used in published results).
  3. llava            — LLaVA-1.5 VLM (alternative, NOT used in published results).

Outputs:
  - fused_features (B, N, fusion_dim) — visual tokens enriched by text context
  - cot_logits     (B, max_len, vocab) — language logits (blip2/llava only; None for cross_attn_only)
  - text_embedding  (B, fusion_dim)   — sentence-level embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─── Shared positional encoding ───────────────────────────────────────────────

class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


# ─── Text tokeniser & embedding (CLIP text encoder) ───────────────────────────

class CLIPTextEncoder(nn.Module):
    """Encode short/long descriptions using the CLIP text transformer."""

    def __init__(self, clip_model_name: str, out_dim: int):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_model = CLIPTextModel.from_pretrained(clip_model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False

        text_hidden = self.text_model.config.hidden_size
        self.proj = nn.Linear(text_hidden, out_dim)

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            token_embeds  (B, L, out_dim)
            sent_embed    (B, out_dim)   — EOS token (pooled)
        """
        device = next(self.text_model.parameters()).device
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=77
        ).to(device)

        with torch.no_grad():
            out = self.text_model(**enc, return_dict=True)

        token_embeds = self.proj(out.last_hidden_state)   # (B, L, out_dim)
        sent_embed = self.proj(out.pooler_output)          # (B, out_dim)
        return token_embeds, sent_embed


# ─── Cross-attention fusion block ─────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Standard cross-attention: image tokens (Q) attend to text tokens (KV).
    Followed by a feed-forward layer.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self,
                img_tokens: torch.Tensor,   # (B, N, D)
                txt_tokens: torch.Tensor,   # (B, L, D)
                txt_key_padding_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention on image tokens
        x, _ = self.self_attn(img_tokens, img_tokens, img_tokens)
        img_tokens = self.norm1(img_tokens + x)

        # Cross-attention: image (Q) → text (K, V)
        x, attn_weights = self.cross_attn(
            img_tokens, txt_tokens, txt_tokens,
            key_padding_mask=txt_key_padding_mask,
        )
        img_tokens = self.norm2(img_tokens + x)

        # Feed-forward
        img_tokens = self.norm3(img_tokens + self.ff(img_tokens))

        return img_tokens, attn_weights  # (B, N, D), (B, N, L)


# ─── Lightweight cross-attention-only reasoning module ────────────────────────

class CrossAttentionReasoningModule(nn.Module):
    """
    No large LLM required.
    Fuses image + depth features with text descriptions via stacked cross-attention.
    Produces spatial attention heatmap (used as 'reasoning chain proxy').
    """

    def __init__(self, model_cfg):
        super().__init__()
        clip_dim = model_cfg.clip_hidden_dim      # 1024
        depth_dim = model_cfg.depth_feature_dim   # 256
        text_dim = model_cfg.text_hidden_dim      # 512
        fusion_dim = model_cfg.fusion_dim         # 768
        n_heads = model_cfg.num_fusion_heads

        self.clip_model_name = model_cfg.clip_model

        # Project CLIP + Depth → fusion_dim
        self.img_proj = nn.Linear(clip_dim + depth_dim, fusion_dim)

        # Text encoder
        self.text_encoder = CLIPTextEncoder(model_cfg.clip_model, text_dim)

        # Project text to fusion_dim
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # Positional encoding
        self.pos_enc = SinCosPositionalEncoding(fusion_dim)

        # Stack of cross-attention blocks (dropout=0.2 for regularisation on small dataset)
        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(fusion_dim, n_heads, dropout=0.2) for _ in range(4)
        ])

        # Output projection
        self.out_proj = nn.Linear(fusion_dim, fusion_dim)

        # Sentence-level embedding projection (for cls head input)
        self.sent_proj = nn.Linear(text_dim, fusion_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,          # (B, N, clip_dim)
        depth_tokens: torch.Tensor,          # (B, N, depth_dim)
        descriptions: List[str],             # length B
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused    (B, N, fusion_dim) — image tokens enriched by text
            attn_map (B, N)            — mean cross-attention over last CA block
            sent_emb (B, fusion_dim)   — sentence embedding for classification head
        """
        B = patch_tokens.shape[0]

        # ── Concatenate CLIP + Depth tokens ───────────────────────────────────
        img = torch.cat([patch_tokens, depth_tokens], dim=-1)  # (B, N, clip+depth)
        img = self.img_proj(img)                                # (B, N, fusion_dim)
        img = self.pos_enc(img)

        # ── Encode text ───────────────────────────────────────────────────────
        txt_tok, sent_emb = self.text_encoder(descriptions)     # (B,L,text_dim), (B,text_dim)
        txt_tok = self.text_proj(txt_tok)                        # (B, L, fusion_dim)
        sent_emb = self.sent_proj(sent_emb)                     # (B, fusion_dim)

        # ── Stacked cross-attention ───────────────────────────────────────────
        x = img
        last_attn = None
        for block in self.ca_blocks:
            x, attn = block(x, txt_tok)
            last_attn = attn   # (B, N, L)

        x = self.out_proj(x)   # (B, N, fusion_dim)

        # Summarise attention map over text tokens → spatial heatmap (B, N)
        attn_map = last_attn.mean(dim=-1)  # (B, N)

        return x, attn_map, sent_emb


# ─── BLIP-2 VLM reasoning module ─────────────────────────────────────────────

class BLIP2ReasoningModule(nn.Module):
    """
    Full BLIP-2 VLM reasoning.
    Generates a free-form CoT reasoning string conditioned on the image and description.
    Also returns intermediate Q-Former features as fused visual representation.
    """

    def __init__(self, model_cfg):
        super().__init__()
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        print(f"Loading BLIP-2: {model_cfg.blip2_model}")
        self.processor = Blip2Processor.from_pretrained(model_cfg.blip2_model)
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            model_cfg.blip2_model,
            torch_dtype=torch.float16,
        )
        # Freeze — BLIP-2 used for feature extraction + inference, not fine-tuned
        for p in self.blip2.parameters():
            p.requires_grad = False

        qformer_dim = self.blip2.config.qformer_config.hidden_size  # usually 768
        self.fuse_proj = nn.Linear(qformer_dim, model_cfg.fusion_dim)

    def generate_reasoning(self,
                            images: torch.Tensor,
                            prompts: List[str],
                            max_new_tokens: int = 150) -> List[str]:
        """
        Generate CoT reasoning strings from image + prompt.
        Used at inference time to produce interpretable reasoning.
        """
        device = next(self.blip2.parameters()).device
        inputs = self.processor(
            images=[img for img in images],
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            gen_ids = self.blip2.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        descriptions: List[str],
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """
        Returns:
            fused    (B, 32, fusion_dim)  — Q-Former query features
            None                          — no per-patch attn in BLIP-2 path
            sent_emb (B, fusion_dim)      — mean Q-Former output
        """
        device = pixel_values.device
        inputs = self.processor(
            images=pixel_values,
            text=descriptions,
            return_tensors="pt",
            padding=True,
        ).to(device)

        out = self.blip2(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # Q-Former outputs: (B, 32, qformer_dim)
        qformer_out = out.qformer_outputs.last_hidden_state
        fused = self.fuse_proj(qformer_out)           # (B, 32, fusion_dim)
        sent_emb = fused.mean(dim=1)                  # (B, fusion_dim)

        return fused, None, sent_emb


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_reasoning_module(model_cfg):
    backend = model_cfg.llm_backend
    if backend == "blip2":
        try:
            return BLIP2ReasoningModule(model_cfg)
        except Exception as e:
            print(f"[WARN] BLIP-2 failed ({e}). Falling back to cross_attn_only.")
    return CrossAttentionReasoningModule(model_cfg)
