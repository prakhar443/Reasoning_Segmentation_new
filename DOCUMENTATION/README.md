# Sea Ice SAR Segmentation & Classification — First Proof of Concept

**The first end-to-end deep-learning pipeline for simultaneous pixel-level
segmentation and 6-class ice-type classification from SAR sea ice satellite
imagery.**  No prior deep-learning baseline existed for this combined task.
The goal is not to claim state-of-the-art metrics but to demonstrate the task
is tractable and establish reproducible baselines for future work.

## Test-set results (stratified 15-sample-per-class hold-out)

| Task | Metric | Value |
|------|--------|-------|
| Segmentation | mIoU | 0.351 |
| Segmentation | cIoU | 0.456 |
| Segmentation | Dice | 0.442 |
| Segmentation | Pixel Accuracy | 0.656 |
| Classification | Weighted F1 | 0.778 |
| Classification | Accuracy | 0.833 |
| Classification | Macro F1 | 0.778 |

5 of 6 ice classes reach F1 ≥ 0.667 (4 reach F1 = 1.0).
Old Ice F1 = 0.000 — see Limitations.

---

## Pipeline — What Actually Works

The effective pipeline has **5 modules**.  Components in the codebase that were
tested and found not to contribute are listed separately below.

```
SAR image (single-band grayscale → pseudo-RGB)
        │
        ▼
[1] SAR Preprocessing
    Lee speckle filter → dB conversion → 3-channel pseudo-RGB tensor (512×512)
        │
        ▼
[2] CLIP ViT-L/14 + LoRA
    Frozen CLIP vision backbone + 4 rank-8 LoRA adapters on Q/K/V/Out
    ~1.57 M trainable of 304 M total (0.52 %)
    → patch tokens (B, 1024, N)  +  attention weights (B, N)
        │
        ▼
[3] Cross-Attention Fusion
    4 stacked cross-attention blocks fuse visual patch tokens with
    CLIP text encodings of per-image descriptions
    → fused tokens (B, N, 768)  +  sentence embedding (B, 768)
        │
    ┌───┴────────────────────┐
    ▼                        ▼
[4] U-Net Mask Decoder    [5] 6-Class Ice Classifier
    Image-conditioned          2-layer MLP on masked-pooled
    U-Net, 4 scales,           fused features + sentence emb.
    Tversky loss               Cross-entropy loss
    (α=0.6, β=0.4) +
    aux deep supervision       → ice-type label (one of 6)
    at ¼ resolution
    → binary mask (B,1,H,W)
```

### Components tested but not contributing (in codebase, not claimed)

| Component | Finding | Status |
|-----------|---------|--------|
| DepthAnything V2 depth encoder | Ablation bit-identical to full model across all metrics in two independent runs | In codebase, always effectively disabled |
| Temporal consistency module | No measurable benefit; disabled before final runs | In codebase, disabled |
| SAM mask decoder | Replaced by the lighter U-Net decoder; not evaluated in final model | In codebase, optional |
| Free-form text generation | `cross_attn_only` backend does not generate language; output is class-conditioned template | Never claimed |

---

## Ice Classes

```
0  Young Ice         — thin, smooth, low backscatter
1  First Year Ice    — intermediate thickness
2  Floating Ice      — buoyant ice fragments
3  Glaciers          — land-based ice, distinct spectral signature
4  Icebergs          — calved glacier fragments
5  Old Ice           — multi-year ice; FAILS classification (see Limitations)
```

---

## Limitations

### Old Ice classification (F1 = 0.000)
Old (multi-year) ice is physically the *opposite* of Young/First-Year ice —
thicker, rougher, with higher SAR volume backscatter — and is separable in
calibrated or multi-polarization SAR.  The failure here is a **modality
limitation**: the dataset provides single-channel grayscale intensity JPEGs.
On that one band, Old Ice (mean brightness 141.5, 90th pct 180.5), Young Ice
(140.8 / 178.4) and First Year Ice (143.3 / 175.4) are statistically
indistinguishable.  Glaciers (157.7 / 194.7) stands clearly apart and scores
F1 = 1.0.  **Fix:** use calibrated dual-polarization imagery or add an explicit
roughness / thickness channel.

### Over-segmentation
Precision 0.444, Recall 0.640.  The model favours ice recall at the cost of
precision.  Boundary IoU ≈ 0.014 — masks capture region extent, not fine edges.
Consistent with Otsu-binarized ground truth used for supervision.

### Ablation training-budget mismatch
Full model: 50 epochs.  Ablated variants: ≤15 epochs (early stop, patience 5).
Tversky-Only beats the full model on mIoU (0.389 > 0.351) and Dice (0.493 >
0.442) — reported honestly; the full model was not the best segmentation row.
The combination (Focal + Tversky) was retained for a more balanced
precision/recall operating point, not for raw mIoU.

### Single-band input ceiling
Train mIoU ≈ val mIoU throughout training — the performance ceiling is a
property of the supervision signal (Otsu-binarized scattering maps), not of
the optimizer.  Better masks require better labels.

---

## Dataset

```
dataset/
├── Young Ice/          100 images  (70 train / 15 val / 15 test)
├── First Year Ice/     100 images  (70 train / 15 val / 15 test)
├── Floating Ice/       100 images  (70 train / 15 val / 15 test)
├── Glaciers/           100 images  (70 train / 15 val / 15 test)
├── Icebergs/           100 images  (70 train / 15 val / 15 test)
└── Old Ice/            100 images  (70 train / 15 val / 15 test)
```

Each folder: `images/` (SAR .jpg), `masks/` (binary .jpg), `descriptions/` (.csv or .xlsx).
Split is **stratified and deterministic** (per-class shuffle with fixed seed).

---

## Installation

```bash
git clone https://github.com/prakhar443/sea_ice_seg.git
cd sea_ice_seg
pip install -r DOCUMENTATION/requirements.txt
```

---

## Training

```bash
# Full training (50 epochs, U-Net decoder, Tversky loss)
python train.py \
    --data_root dataset \
    --output_dir outputs/exp1 \
    --epochs 50 \
    --batch_size 4

# Resume from checkpoint
python train.py --data_root dataset --output_dir outputs/exp1 \
    --resume outputs/exp1/best_model.pth
```

Key config values (`config.py`):
```python
cfg.model.decoder_type    = 'unet'
cfg.model.llm_backend     = 'cross_attn_only'
cfg.train.tversky_alpha   = 0.6   # penalises false negatives more
cfg.train.tversky_beta    = 0.4
cfg.train.epochs          = 50
cfg.train.early_stop_patience = 12
```

---

## Evaluation / Reproducing Results

Open `sea_ice_colab_training.ipynb` in Google Colab.

**To reproduce all metrics and figures in one step:**  
Run **Cell 19 — Complete Results & Figures**.  
Prerequisites: `best_model.pth` in `outputs/` or Google Drive (run Cell 18b to train).  
For the ablation figure also run Ablation Cells A and B first.

Outputs saved to `outputs/figures/`:

| Figure | Content |
|--------|---------|
| `fig0_training_curves.png` | Loss / mIoU / F1 vs epoch |
| `fig1_per_class_f1.png` | Per-class F1 bar chart |
| `fig2_segmentation_overview.png` | Segmentation metrics + per-class IoU |
| `fig3_confusion_matrix.png` | 6×6 classification confusion matrix |
| `fig4_threshold_sensitivity.png` | mIoU & precision/recall vs threshold |
| `fig5_sample_predictions.png` | 6-panel grid (1 sample per class) |
| `fig6_ablation.png` | mIoU / F1 across ablation variants |

---

## Ablation Summary

| Variant | mIoU | Dice | Weighted F1 | Note |
|---------|------|------|-------------|------|
| **Full Model** | 0.351 | 0.442 | 0.778 | Focal+Tversky, LoRA, Aux loss |
| w/o LoRA | 0.365 | 0.469 | 0.534 | LoRA hurts seg., crucial for F1 |
| Token Decoder | 0.344 | 0.455 | 1.000 | Comparable seg., perfect F1 (15 samples) |
| w/o Aux Loss | 0.324 | 0.405 | 0.801 | Aux loss helps mIoU |
| w/o Depth | 0.351 | 0.442 | 0.778 | **Identical** — depth contributes nothing |
| Focal Only | 0.252 | 0.310 | 1.000 | Collapses segmentation |
| **Tversky Only** | **0.389** | **0.493** | 0.989 | **Best segmentation** |
| BCE + Dice | 0.332 | 0.421 | 0.822 | Baseline |

Bold marks true column-best (not always the full model — reported honestly).

---

## References

- CLIP: Radford et al., 2021 — *Learning Transferable Visual Models from Natural Language Supervision*
- LoRA: Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*
- Tversky loss: Salehi et al., 2017 — *Tversky loss function for image segmentation*
- DepthAnything V2: Yang et al., 2024 (in codebase; not a claimed contribution)
- SAM: Kirillov et al., 2023 (in codebase; not used in final model)

---

**Status:** Research prototype — first proof of concept for combined SAR sea ice segmentation + classification  
**Last updated:** June 2026
