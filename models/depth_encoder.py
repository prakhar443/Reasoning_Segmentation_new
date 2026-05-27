"""
models/depth_encoder.py — DepthAnything V2 surface-topology feature extractor.

Provides complementary depth/surface-structure features alongside CLIP.
For sea ice: encodes pressure ridges, hummocking, and flat slab boundaries.
Output features are concatenated with CLIP patch tokens before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthAnyV2Encoder(nn.Module):
    """
    DepthAnything V2 wrapper that extracts intermediate feature maps.

    We use the encoder features (before the final head) rather than the
    raw depth prediction, since the features carry richer structural info.

    Forward input:  pixel_values (B, 3, H, W)
    Forward output: depth_features (B, N, D_depth)
        where N matches CLIP patch count (interpolated if needed)
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.out_dim = model_cfg.depth_feature_dim  # 256

        print(f"Loading DepthAnything V2: {model_cfg.depth_model}")
        self.processor = AutoImageProcessor.from_pretrained(model_cfg.depth_model)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            model_cfg.depth_model,
            output_hidden_states=True,
        )

        if model_cfg.depth_freeze:
            for param in self.depth_model.parameters():
                param.requires_grad = False

        # Infer encoder feature dim from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 518, 518)
            out = self.depth_model(dummy, output_hidden_states=True)
            # Use the last hidden state from the backbone
            enc_dim = out.hidden_states[-1].shape[-1]

        # Projection: encoder_dim → depth_feature_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(enc_dim, self.out_dim),
            nn.GELU(),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, pixel_values: torch.Tensor,
                target_seq_len: int = 256) -> torch.Tensor:
        """
        Args:
            pixel_values:   (B, 3, H, W)
            target_seq_len: number of patch tokens to match (from CLIP encoder)

        Returns:
            depth_features: (B, target_seq_len, D_depth)
        """
        with torch.set_grad_enabled(not self.model_cfg.depth_freeze):
            out = self.depth_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states[-1]: (B, patches, enc_dim) — use deepest features
        feats = out.hidden_states[-1]          # (B, N_depth, enc_dim)

        # Project to target dim
        feats = self.feature_proj(feats)       # (B, N_depth, D_depth)

        # Interpolate seq length to match CLIP patch count
        if feats.shape[1] != target_seq_len:
            feats = feats.permute(0, 2, 1)     # (B, D, N)
            feats = F.interpolate(
                feats, size=target_seq_len, mode="linear", align_corners=False
            )
            feats = feats.permute(0, 2, 1)     # (B, target_seq_len, D)

        return feats  # (B, target_seq_len, D_depth)


class DepthFallbackEncoder(nn.Module):
    """
    Lightweight depth-proxy encoder for when DepthAnything V2 is unavailable.
    Applies hand-crafted edge and gradient filters on the SAR image to extract
    structural cues similar to what depth estimation would provide.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim

        # Learnable 1x1 conv after Sobel/LoG feature stack
        self.proj = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_dim, 1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, pixel_values: torch.Tensor,
                target_seq_len: int = 256) -> torch.Tensor:
        # Average channels → single-channel SAR proxy
        gray = pixel_values.mean(dim=1, keepdim=True)  # (B,1,H,W)

        # Approximate Sobel edges
        sobel_x = F.conv2d(gray, self._sobel_x().to(gray.device), padding=1)
        sobel_y = F.conv2d(gray, self._sobel_y().to(gray.device), padding=1)
        magnitude = torch.sqrt(sobel_x ** 2 + sobel_y ** 2 + 1e-8)
        laplacian = F.conv2d(gray, self._laplacian().to(gray.device), padding=1)

        feats = torch.cat([gray, sobel_x, sobel_y, magnitude], dim=1)  # (B,4,H,W)

        # Pool to fixed size, then expand to sequence
        pooled = F.adaptive_avg_pool2d(feats, int(target_seq_len ** 0.5))
        B = pixel_values.shape[0]
        h = w = int(target_seq_len ** 0.5)
        pooled = pooled.view(B, 4, -1).permute(0, 2, 1)  # (B, N, 4)

        # Project to out_dim
        out = nn.Linear(4, self.out_dim).to(gray.device)(pooled)  # (B, N, out_dim)
        return out

    @staticmethod
    def _sobel_x():
        k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        return k.view(1, 1, 3, 3)

    @staticmethod
    def _sobel_y():
        k = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        return k.view(1, 1, 3, 3)

    @staticmethod
    def _laplacian():
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        return k.view(1, 1, 3, 3)


def build_depth_encoder(model_cfg):
    """Factory: try DepthAnything V2, fall back to gradient-based encoder."""
    try:
        return DepthAnyV2Encoder(model_cfg)
    except Exception as e:
        print(f"[WARN] DepthAnything V2 not available ({e}). Using fallback.")
        return DepthFallbackEncoder(out_dim=model_cfg.depth_feature_dim)
