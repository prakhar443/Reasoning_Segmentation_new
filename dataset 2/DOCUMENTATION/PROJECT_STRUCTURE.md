"""
PROJECT_STRUCTURE.md — Complete guide to the Sea Ice Segmentation Pipeline codebase.
"""

# Complete Project Structure & File Guide

## Directory Layout

```
sea_ice_seg/
├── config.py                          ← Central configuration (ALL hyperparameters)
├── train.py                           ← Training script
├── inference.py                       ← Inference on single/batch images
├── evaluate.py                        ← Test set evaluation with metrics
├── quickstart.py                      ← Setup verification script
├── requirements.txt                   ← Python dependencies
├── README.md                          ← User guide
├── PROJECT_STRUCTURE.md               ← This file
│
├── data/                              ← Data loading & preprocessing
│   ├── __init__.py
│   ├── preprocessing.py               ← SAR preprocessing (Lee filter, dB, stacking)
│   └── dataset.py                     ← PyTorch Dataset + DataLoaders
│
├── models/                            ← Model components & pipeline
│   ├── __init__.py
│   ├── visual_encoder.py              ← CLIP ViT-L/14 + LoRA
│   ├── depth_encoder.py               ← DepthAnything V2 surface topology
│   ├── reasoning_module.py            ← Cross-attention CoT reasoning
│   ├── prompt_generator.py            ← Attention-guided SAM prompts
│   ├── sam_module.py                  ← SAM mask decoder wrapper
│   ├── ice_classifier.py              ← 6-class ice type MLP head
│   ├── temporal_consistency.py        ← Memory bank + temporal smoothing
│   └── pipeline.py                    ← Full 8-module end-to-end pipeline
│
├── utils/                             ← Utilities
│   ├── __init__.py
│   ├── losses.py                      ← Combined loss (mask + cls + CoT)
│   └── metrics.py                     ← Evaluation metrics (IoU, F1, etc.)
│
├── checkpoints/                       ← Model checkpoints (after download)
│   ├── sam_vit_h_4b8939.pth          ← SAM weights (download separately)
│   ├── best_model.pth                ← Trained model (after training)
│   └── checkpoint_step_*.pth         ← Intermediate checkpoints
│
└── dataset/                           ← Dataset root (user-provided)
    ├── first_year_ice/
    │   ├── images/                   ← Original SAR images
    │   ├── masks/                    ← Binary segmentation masks
    │   └── descriptions.csv          ← Image descriptions & queries
    ├── young_ice/
    ├── multi_year_ice/
    ├── nilas/
    ├── deformed_ridged_ice/
    └── open_water_leads/
```

---

## File Descriptions

### Core Configuration & Entry Points

#### `config.py` (≈500 lines)
Central configuration file containing all hyperparameters organized into dataclasses:

- **ICE_CLASSES**: List of 6 ice types (young_ice, first_year_ice, etc.)
- **DataConfig**: Dataset paths, image size, augmentation, preprocessing
- **ModelConfig**: CLIP, Depth, reasoning, SAM, classifier dimensions & models
- **TrainConfig**: Learning rates, epochs, batch size, loss weights, optimiser
- **InferenceConfig**: Checkpoint paths, output options, batch size
- **Global `cfg`**: Singleton instance of all configs

```python
from config import cfg
print(cfg.model.clip_model)      # "openai/clip-vit-large-patch14"
print(cfg.train.epochs)          # 50
```

#### `train.py` (≈450 lines)
Full training pipeline with:
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Learning rate scheduling (cosine warmup)
- Checkpoint saving & resumption
- WandB logging
- Validation & best model tracking

**Usage:**
```bash
python train.py --data_root dataset --output_dir outputs/exp1 --epochs 50 --use_sam
```

**Key functions:**
- `train()` — Main training loop
- `validate()` — Validation step
- `save_checkpoint()` / `load_checkpoint()` — Serialisation
- `get_lr_scheduler()` — LR schedule with warmup

#### `inference.py` (≈450 lines)
Single or batch inference with temporal consistency:
- Load trained checkpoint
- Preprocess SAR images
- Run forward pass
- Generate visualisations (masks, attention, overlays)
- Save JSON predictions

**Usage:**
```bash
# Single image
python inference.py --checkpoint outputs/exp1/best_model.pth --image sample.jpg

# Directory batch
python inference.py --checkpoint outputs/exp1/best_model.pth --image_dir dataset/images
```

**Key classes:**
- `SeaIceInferencer` — Model wrapper for inference
- Functions: `infer_image()`, `infer_directory()`, `draw_mask_overlay()`, `draw_attention_heatmap()`

#### `evaluate.py` (≈550 lines)
Comprehensive test set evaluation with metrics reporting:
- Segmentation: mIoU, Dice, pixel accuracy
- Classification: F1 (macro/weighted), per-class accuracy, confusion matrix
- Visualisations: confusion matrix plot, metric bars
- Detailed results: per-image CSV + JSON

**Usage:**
```bash
python evaluate.py --checkpoint outputs/exp1/best_model.pth --split test --save_visualizations
```

**Key classes:**
- `Evaluator` — Evaluation engine
- Functions: `evaluate()`, `plot_confusion_matrix()`, `plot_metrics_comparison()`

#### `quickstart.py` (≈350 lines)
Setup verification script that checks:
- Dependencies installed
- Directory structure
- Configuration loadable
- All model modules importable
- SAR preprocessing functional
- Dataset loading functional

**Usage:**
```bash
python quickstart.py
```

---

### Data Handling

#### `data/preprocessing.py` (≈350 lines)
SAR-specific image preprocessing:
- **Enhanced Lee speckle filter** — Reduces multiplicative noise while preserving edges
- **dB conversion** — Linear backscatter → dB scale (10 * log10)
- **Channel stacking** — Single-pol pseudo-RGB or dual-pol HH/HV/difference
- **Normalisation & resize** — Matches CLIP/ViT input requirements

**Key classes:**
- `SARPreprocessor` — Main preprocessing pipeline
- Functions: `lee_filter()`, `linear_to_db()`, `stack_to_rgb()`

**Example:**
```python
from config import cfg
from data.preprocessing import SARPreprocessor
preprocessor = SARPreprocessor(cfg.data)
tensor = preprocessor(sar_image)  # (3, 512, 512)
```

#### `data/dataset.py` (≈350 lines)
PyTorch Dataset for loading ice class folders:
- Loads images, masks, descriptions from 6 subdirectories
- Applies augmentations (crops, flips, elastic transforms, noise)
- Balanced sampling via WeightedRandomSampler
- Collate function handling mixed data types (tensors + strings)

**Key classes:**
- `SeaIceDataset` — Main dataset
- Functions: `build_dataloaders()`, `collate_fn()`

**Example:**
```python
from data.dataset import build_dataloaders
train_loader, val_loader, test_loader = build_dataloaders(cfg.data, cfg.train)
```

---

### Model Components

#### `models/visual_encoder.py` (≈150 lines)
CLIP ViT-L/14 visual encoder with LoRA domain adaptation:
- Frozen CLIP backbone (300M params)
- LoRA adapters on Q,V,K,Out (≈2.5M trainable params)
- Extracts patch-level features + CLS token + attention weights

**Key classes:**
- `CLIPSAREncoder` — Main encoder with LoRA

**Example:**
```python
encoder = CLIPSAREncoder(cfg.model)
patches, cls_tok, attn = encoder(images)  # (B,N,1024), (B,1024), (B,N)
```

#### `models/depth_encoder.py` (≈250 lines)
DepthAnything V2 for surface topology features:
- Primary: Full DepthAnything V2 (large pretrained model)
- Fallback: Lightweight Sobel+Laplacian hand-crafted edges
- Output: Depth feature tokens interpolated to match CLIP patch count

**Key classes:**
- `DepthAnyV2Encoder` — Full DepthAnything pipeline
- `DepthFallbackEncoder` — Gradient-based fallback
- Function: `build_depth_encoder()` — Factory with automatic fallback

**Example:**
```python
depth_enc = build_depth_encoder(cfg.model)
depth_feats = depth_enc(images, target_seq_len=256)  # (B, 256, 256)
```

#### `models/reasoning_module.py` (≈400 lines)
Multimodal CoT reasoning with text + image fusion:
- **CrossAttentionReasoningModule** — Lightweight (stacked cross-attention)
- **BLIP2ReasoningModule** — Full VLM (requires 24GB VRAM)
- Text encoder: CLIP text transformer
- Positional encoding for patches
- Stacked cross-attention blocks
- Output: Fused tokens + attention heatmap + sentence embedding

**Key classes:**
- `CLIPTextEncoder` — Text description encoder
- `CrossAttentionBlock` — Self + cross-attention block
- `CrossAttentionReasoningModule` — Lightweight pipeline
- `BLIP2ReasoningModule` — Full VLM pipeline
- Function: `build_reasoning_module()` — Factory with fallback

**Example:**
```python
reasoning = build_reasoning_module(cfg.model)
fused, attn_map, sent_emb = reasoning(patches, depth_feats, descriptions)
# (B, N, 768), (B, N), (B, 768)
```

#### `models/prompt_generator.py` (≈250 lines)
Converts attention heatmaps into SAM bounding-box + point prompts:
- Reshape flat attention → 2D spatial map
- Threshold + morphological cleanup
- Connected component analysis
- Extract bounding boxes + foreground point clicks
- Output: SAM-compatible prompt dicts

**Key classes:**
- `GeometricPromptGenerator` — Main prompt generator
- Function: `draw_prompts_on_image()` — Visualisation

**Example:**
```python
prompt_gen = GeometricPromptGenerator(cfg.model)
prompts = prompt_gen(attn_weights, image_size=(512, 512))
# List[List[Dict]] with boxes, point_coords, point_labels
```

#### `models/sam_module.py` (≈250 lines)
SAM mask decoder wrapper + lightweight alternative:
- **SAMModule** — Official SAM (vit_h/l/b) for high accuracy
- **LightweightMaskDecoder** — Convolutional upsampler (no pretrained)
- Handles batch processing and SAM compatibility

**Key classes:**
- `SAMModule` — SAM wrapper
- `LightweightMaskDecoder` — Learnable decoder
- Functions: `encode_image()`, `forward()`

**Example:**
```python
# Option A: SAM
decoder = SAMModule(cfg.model)
masks, iou = decoder(images_np, prompts)  # (B,1,H,W), (B,)

# Option B: Lightweight
decoder = LightweightMaskDecoder(in_dim=768)
masks = torch.sigmoid(decoder(fused_tokens))  # (B,1,H,W)
```

#### `models/ice_classifier.py` (≈100 lines)
6-class ice type classification head:
- Input: Masked-pooled visual features (B, 768) + sentence embedding (B, 768)
- Architecture: 2-layer MLP with GELU + Dropout
- Output: 6-class logits or softmax probabilities

**Key classes:**
- `IceTypeClassifier` — Main classifier

**Example:**
```python
classifier = IceTypeClassifier(cfg.model)
logits = classifier(fused_tokens, masks, sent_emb)  # (B, 6)
idx, probs, names = classifier.predict(logits)
```

#### `models/temporal_consistency.py` (≈250 lines)
Feature memory bank + temporal smoothing:
- Tracks embeddings of last N frames
- Cosine similarity gating
- Learnable blend gate for current vs. previous predictions
- Prevents spurious frame-to-frame class flips

**Key classes:**
- `TemporalMemoryBank` — Per-sequence memory
- `TemporalConsistencyModule` — Main module
- Methods: `forward()`, `reset_sequence()`, `reset_all()`

**Example:**
```python
temporal = TemporalConsistencyModule(cfg.model)
smooth_logits = temporal(cls_logits, mask_emb, sequence_ids, frame_ids)
```

#### `models/pipeline.py` (≈400 lines)
Full 8-module end-to-end pipeline:
- Orchestrates all modules in sequence
- Handles data flow between modules
- Returns comprehensive output dict
- Supports parametrised module selection (SAM vs lightweight)

**Key classes:**
- `SeaIceSegmentationPipeline` — Main pipeline

**Example:**
```python
model = SeaIceSegmentationPipeline(cfg.model, use_sam=True)
outputs = model(images, descriptions, images_np=images_np, sequence_ids=seq_ids)
# Dict with mask, cls_logits, attention, probs, etc.
```

---

### Utilities

#### `utils/losses.py` (≈250 lines)
Combined loss functions for training:
- **MaskLoss** — BCE + Dice for segmentation
- **WeightedClassificationLoss** — Balanced cross-entropy for 6 classes
- **AttentionGuidanceLoss** — KL divergence for CoT attention regulation
- **SeaIceLoss** — Master loss combining all three

**Key classes:**
- `DiceLoss` — Dice coefficient loss
- `MaskLoss` — Segmentation loss (BCE + Dice)
- `WeightedClassificationLoss` — Class-weighted CE
- `AttentionGuidanceLoss` — Attention regularisation
- `SeaIceLoss` — Combined loss

**Example:**
```python
criterion = SeaIceLoss(cfg.train)
loss_dict = criterion(outputs, targets={"mask": gt_mask, "label": gt_label})
# {"loss": total, "loss_mask": ..., "loss_cls": ..., "loss_attn": ...}
```

#### `utils/metrics.py` (≈350 lines)
Segmentation + classification metrics:
- **Segmentation:** mIoU, Dice, pixel accuracy
- **Classification:** F1 (macro/weighted), per-class F1, confusion matrix
- **Tracking:** MetricAccumulator for batch-wise evaluation

**Key classes:**
- `ClassificationMetrics` — Per-batch classification tracking
- `MetricAccumulator` — Epoch-level metric aggregation
- Functions: `compute_iou()`, `compute_dice()`, `compute_pixel_accuracy()`

**Example:**
```python
metrics = MetricAccumulator()
for batch in dataloader:
    outputs = model(batch)
    metrics.update(outputs, targets, loss)
summary = metrics.compute()
print(f"mIoU: {summary['mean_iou']:.4f}, F1: {summary['weighted_f1']:.4f}")
```

---

## Module Dependency Graph

```
train.py / inference.py / evaluate.py
    ↓
config.py ←──┐
    ↓        │
data/preprocessing.py (SARPreprocessor)
    ↓        │
data/dataset.py (SeaIceDataset)
    ↓        │
models/pipeline.py (SeaIceSegmentationPipeline)
    │        ├─→ models/visual_encoder.py (CLIPSAREncoder + LoRA)
    │        ├─→ models/depth_encoder.py (DepthAnyV2)
    │        ├─→ models/reasoning_module.py (CrossAttention/BLIP2)
    │        ├─→ models/prompt_generator.py (GeometricPromptGenerator)
    │        ├─→ models/sam_module.py (SAM/LightweightDecoder)
    │        ├─→ models/ice_classifier.py (6-class MLP)
    │        └─→ models/temporal_consistency.py (MemoryBank)
    ↓        │
utils/losses.py (SeaIceLoss)
    ↓        │
utils/metrics.py (MetricAccumulator)
    └────────┘
```

---

## Data Flow During Training

```
Dataset (image, mask, label, description)
    ↓
SARPreprocessor
    ↓ (3, H, W) normalised tensor
    ↓
[Batch loading & augmentation]
    ↓
CLIPSAREncoder + LoRA
    ↓ (B, N, 1024) patches
    ↓
DepthAnyV2Encoder
    ↓ (B, N, 256) depth features
    ↓
CrossAttentionReasoningModule
    ↓ (B, N, 768) fused tokens + (B, N) attention
    ↓
├─→ GeometricPromptGenerator → SAMModule
│                                ↓
│                            (B, 1, H, W) masks
│
├─→ IceTypeClassifier
│       (B, 768) sentence embedding + masked features
│       ↓
│   (B, 6) class logits
│
└─→ TemporalConsistencyModule
        ↓
    (B, 6) smoothed class logits

[Combined via SeaIceLoss]
    ↓
loss_mask + loss_cls + loss_attn = total_loss
    ↓
Backprop → optimizer.step()
```

---

## Data Flow During Inference

```
Raw SAR Image (GeoTIFF/JPEG)
    ↓
SARPreprocessor
    ↓ (3, H, W) tensor
    ↓
SeaIceSegmentationPipeline.forward()
    ├─→ Visual encoder → (B, N, 1024) patches
    ├─→ Depth encoder → (B, N, 256) features
    ├─→ Reasoning → (B, N, 768) fused + (B, N) attention
    ├─→ Prompt generator → SAM prompts
    ├─→ SAM decoder → (B, 1, H, W) binary mask
    ├─→ Classifier → (B, 6) class logits
    └─→ Temporal smoothing → (B, 6) final logits
    ↓
[JSON output + PNG masks + PNG attention heatmap]
```

---

## Training Command Examples

```bash
# Minimal (lightweight decoder, no SAM)
python train.py --data_root dataset --output_dir outputs/exp1

# Full setup with SAM
python train.py --data_root dataset --output_dir outputs/exp1 --use_sam --epochs 50 --batch_size 4

# Resume from checkpoint
python train.py --data_root dataset --output_dir outputs/exp1 --resume outputs/exp1/checkpoint_step_5000.pth

# With logging
python train.py --data_root dataset --output_dir outputs/exp1 --use_wandb --wandb_project sea-ice-seg

# Custom hyperparameters
python train.py --data_root dataset --output_dir outputs/exp1 --lr 1e-4 --epochs 100 --batch_size 2
```

---

## Inference Command Examples

```bash
# Single image
python inference.py --checkpoint outputs/exp1/best_model.pth --image sample.jpg --output results/

# Single image with custom description
python inference.py --checkpoint outputs/exp1/best_model.pth --image sample.jpg \
    --description "First-year ice with melt ponds" --output results/

# Directory batch
python inference.py --checkpoint outputs/exp1/best_model.pth \
    --image_dir dataset/first_year_ice/images \
    --descriptions_csv dataset/first_year_ice/descriptions.csv \
    --output results/

# With lightweight decoder (faster, less VRAM)
python inference.py --checkpoint outputs/exp1/best_model.pth --image sample.jpg --output results/
```

---

## Evaluation Command Examples

```bash
# Test set with visualisations
python evaluate.py --checkpoint outputs/exp1/best_model.pth --split test \
    --save_visualizations --save_masks --output eval_results/

# Validation set only
python evaluate.py --checkpoint outputs/exp1/best_model.pth --split val --output eval_results/

# Lightweight evaluation (no visualisations)
python evaluate.py --checkpoint outputs/exp1/best_model.pth --split test --output eval_results/
```

---

## Size & Complexity Reference

| Component | Lines | Params | VRAM (GB) | Notes |
|-----------|-------|--------|-----------|-------|
| CLIP ViT-L/14 | — | 300M | 2.0 | Frozen base + 2.5M LoRA |
| DepthAnything V2 | — | 36M | 1.2 | Lightweight backbone |
| BLIP-2 (optional) | — | 3B | 8.0 | For full VLM (else cross-attn) |
| SAM ViT-H | — | 640M | 2.0 | Optional (else lightweight) |
| **Total trainable** | — | **≈10M** | **≤1.2** | LoRA + classifier + decoder |
| **Full model inference** | — | **≈1B** | **≤12** | If using SAM + DepthAnything |

---

## File Statistics

```
Total lines of code:  ≈6,500
├── Models:           ≈2,800 lines
├── Data:               ≈700 lines
├── Training:         ≈1,000 lines
├── Inference:          ≈450 lines
├── Evaluation:         ≈550 lines
└── Utilities:          ≈600 lines

Python files:  20
Config files:  1
Documentation: 3 (README, PROJECT_STRUCTURE, This file)
```

---

## Adding Custom Modules

To extend the pipeline:

1. **New preprocessing** → `data/preprocessing.py` + `SARPreprocessor`
2. **New encoder** → `models/visual_encoder.py` or `models/depth_encoder.py`
3. **New reasoning backend** → `models/reasoning_module.py` + factory
4. **New loss function** → `utils/losses.py` + `SeaIceLoss.forward()`
5. **New metrics** → `utils/metrics.py` + `MetricAccumulator`

Update `models/pipeline.py` to wire in new modules.

---

**Last updated:** May 2026
