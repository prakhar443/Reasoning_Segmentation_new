"""
config.py — Central configuration for the Sea Ice Reasoning Segmentation pipeline.
All paths, hyperparameters, and model settings live here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


# ─── Ice class definitions ────────────────────────────────────────────────────
# Names must exactly match the subdirectory names inside dataset/

ICE_CLASSES = [
    "Young Ice",
    "First Year Ice",
    "Floating Ice",
    "Glaciers",
    "Icebergs",
    "Old Ice",
]

ICE_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ICE_CLASSES)}
IDX_TO_ICE_CLASS = {i: cls for cls, i in ICE_CLASS_TO_IDX.items()}

# Class weights to handle imbalanced datasets (adjust per your distribution)
ICE_CLASS_WEIGHTS = [1.0, 1.0, 1.5, 1.5, 1.5, 1.0]


# ─── Dataset config ───────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    # Root folder containing 6 ice-type subdirectories
    data_root: str = "dataset"

    # Each subfolder must match an ICE_CLASSES name exactly
    # e.g.  dataset/young_ice/images/   dataset/young_ice/masks/
    image_subdir: str = "images"
    mask_subdir: str = "masks"

    # Subdirectory holding per-class description xlsx files
    # Each class folder contains:  descriptions/<class>_descriptions_appended.xlsx
    descriptions_subdir: str = "descriptions"

    # Image settings
    image_size: Tuple[int, int] = (512, 512)
    image_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # Mask handling — the '_scat' masks are continuous scattering maps, not
    # clean binary labels, and have a different aspect ratio than the images.
    mask_binarize: str = "otsu"          # "otsu" | "mean" | "fixed"
    # Images are 256×256 (square); the `_scat` masks are ~138×187 (portrait).
    # They cover the SAME scene at different sampling resolutions, so the mask
    # must be resized to the image's extent ("stretch") to stay spatially
    # aligned. "letterbox" pads image and mask independently — because their
    # aspect ratios differ, the mask foreground ends up offset from the image
    # content and the target becomes unlearnable (mIoU pins at the all-fg
    # fraction ~0.32). Keep "stretch" unless image and mask share an aspect ratio.
    mask_resize_mode: str = "stretch"    # "stretch" (aligned) | "letterbox"

    # When False, descriptions never name the ice class — prevents the
    # classification head from cheating via the text prompt (honest F1).
    use_class_name_in_prompt: bool = False

    # Train / val / test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # SAR preprocessing
    speckle_filter: str = "lee"          # options: "lee", "enhanced_lee", "none"
    speckle_window: int = 7
    speckle_damping: float = 1.0
    to_db: bool = True                   # convert linear to dB scale
    db_clip_min: float = -30.0
    db_clip_max: float = 5.0
    dual_pol: bool = False               # True if HH+HV available


# ─── Model config ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # ── Visual Encoder (CLIP + LoRA) ──────────────────────────────────────────
    clip_model: str = "openai/clip-vit-large-patch14"
    clip_freeze: bool = True             # freeze base weights; only LoRA trains
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.10           # ↑ from 0.05 — curb overfitting on ~420 imgs
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"]
    )

    # ── DepthAnything V2 ──────────────────────────────────────────────────────
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_freeze: bool = True
    depth_feature_dim: int = 256

    # ── Reasoning / CoT module (cross-attention + LLM head) ───────────────────
    # Using BLIP-2 or LLaVA for the language side; set llm_backend accordingly
    llm_backend: str = "blip2"           # options: "blip2", "llava", "cross_attn_only"
    blip2_model: str = "Salesforce/blip2-opt-2.7b"
    llava_model: str = "llava-hf/llava-1.5-7b-hf"
    use_short_description: bool = True
    use_long_description: bool = True

    # ── Multimodal fusion dims ────────────────────────────────────────────────
    clip_hidden_dim: int = 1024          # ViT-L/14 output dim
    text_hidden_dim: int = 512
    fusion_dim: int = 768
    num_fusion_heads: int = 8

    # ── SAM ───────────────────────────────────────────────────────────────────
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"        # options: vit_h, vit_l, vit_b
    sam_freeze: bool = True

    # ── Lightweight (non-SAM) decoder ─────────────────────────────────────────
    # "unet"  — full-resolution U-Net on the SAR image (sharp masks; recommended,
    #           the `_scat` targets are fine texture maps the 16x16 token grid
    #           cannot represent).
    # "token" — coarse CLIP-token → upsample decoder (lower VRAM, much blurrier).
    decoder_type: str = "unet"
    decoder_base_channels: int = 32      # U-Net width; drop to 16 if VRAM-tight

    # ── Prompt generator ──────────────────────────────────────────────────────
    attn_threshold: float = 0.70
    max_prompts_per_image: int = 3
    prompt_min_area: int = 500           # min connected component area (pixels)

    # ── 6-class classification head ───────────────────────────────────────────
    # Smaller head + heavier dropout: train F1 was hitting 1.0 (pure memorization)
    # while val F1 collapsed — classic over-capacity for a ~420-image dataset.
    cls_hidden_dim: int = 256            # ↓ from 512
    cls_dropout: float = 0.30            # ↑ from 0.10
    num_classes: int = len(ICE_CLASSES)

    # ── Temporal consistency ──────────────────────────────────────────────────
    memory_bank_size: int = 5
    temporal_sim_threshold: float = 0.65
    temporal_blend_alpha: float = 0.70   # weight for current prediction


# ─── Training config ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    epochs: int = 50
    batch_size: int = 4
    grad_accum_steps: int = 4            # effective batch = 16

    # Optimiser
    optimizer: str = "adamw"
    lr: float = 5e-5
    lora_lr: float = 2e-4               # LoRA adapters can use higher LR
    cls_head_lr: float = 1e-4
    decoder_lr: float = 2e-4            # lower now that real mask signal exists (was 3e-4)
    weight_decay: float = 0.05          # ↑ from 0.01 — stronger L2 vs overfitting
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05

    # Loss weights
    # mask loss stays primary (segmentation is the goal) but NOT so extreme
    # that "predict all foreground" becomes a stable minimum.
    lambda_mask: float = 2.0            # Focal + Dice for segmentation mask
    lambda_cls: float = 0.5             # CrossEntropy for 6-class head
    lambda_cot: float = 0.05            # CoT reasoning supervision (if available)
    dice_smooth: float = 1e-4
    lambda_aux: float = 0.2             # ↓ from 0.4: aux was running ≥1.0 and driving overflow
    grad_clip_norm: float = 1.0         # global grad-norm clip (0.5 starved the U-Net)

    # Mask-loss shape. Diagnostics showed the model over-segments (precision
    # ≈0.42, recall ≈0.84): it floods foreground to catch every ice pixel.
    # In Tversky = TP/(TP + α·FP + β·FN), raising α penalises false positives
    # (over-prediction) harder than false negatives, trading excess recall for
    # precision. α=0.6,β=0.4 is a modest nudge from balanced Dice (0.5/0.5).
    focal_alpha: float = 0.5
    tversky_alpha: float = 0.6          # ↑ penalise false positives (curb over-seg)
    tversky_beta: float = 0.4

    # Mixed precision
    fp16: bool = True
    bf16: bool = False

    # Early stopping on val mIoU. Raised to 12 because mask alignment was the
    # blocker until epoch 13 of the prior run; F1 was still trending up at stop.
    early_stop_patience: int = 12       # stop if val mIoU doesn't improve for N evals

    # Logging & checkpointing
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 500
    keep_last_n: int = 3
    use_wandb: bool = False
    wandb_project: str = "sea-ice-seg"

    # Hardware
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ─── Inference config ─────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    model_checkpoint: str = "outputs/best_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    use_tta: bool = False                # test-time augmentation
    conf_threshold: float = 0.50
    temporal_mode: bool = True           # enable temporal consistency at inference
    output_dir: str = "results"
    save_masks: bool = True
    save_overlay: bool = True
    save_json: bool = True               # save predictions as JSON


# ─── Master config ────────────────────────────────────────────────────────────

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# Singleton default config
cfg = Config()
