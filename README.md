# Sea Ice SAR: Reasoning Segmentation & Ice-Type Classification

Reasoning-segmentation pipeline for **joint** pixel-level sea-ice segmentation
and six-class ice-type classification from single-band SAR imagery, where the
**text and the image jointly navigate the segmentation**: each image's
annotator description conditions the mask decoder, not just the classifier.

## What the model is

The reasoning-segmentation pipeline (defaults in `config.py`) is:

1. **SAR preprocessing** — Lee filter (7×7) → dB conversion → pseudo-RGB
   (`data/preprocessing.py`)
2. **CLIP ViT-L/14 + rank-8 LoRA** on Q/K/V/O projections — 1.57 M trainable
   of 304 M (`models/visual_encoder.py`)
3. **Cross-attention text–visual fusion** — the per-image description's CLIP
   text embeddings as K/V, visual patch tokens as Q
   (`models/reasoning_module.py`)
4. **Text-guided U-Net decoder** (`models/sam_module.py::ImageUNetDecoder`,
   `text_guided_decoder=True`): the text-fused tokens are injected at the
   bottleneck alongside the CLIP patch tokens, the description's sentence
   embedding FiLM-modulates the bottleneck (per-channel γ/β), and a
   **text–pixel cosine-similarity map** is concatenated before the mask head.
   The description therefore steers *where* the mask goes.
5. **6-class MLP classifier** (`models/ice_classifier.py`)

BLIP-2 / LLaVA backends, a SAM decoder, a DepthAnything V2 branch, and a
temporal-consistency module exist in the codebase as evaluated-and-excluded
alternatives (kept for ablation reproducibility).

### Ground truth: raw `_scat` maps (not Otsu masks)

The continuous `_scat` scattering maps **are** the ground truth
(`mask_target_mode = "soft_scat"` in `config.py`): each map is per-image
min–max normalised to [0, 1] and regressed directly as a soft target
(BCE + soft Tversky + L1, `utils/losses.py::SoftMaskLoss`). Otsu-binarised
masks are no longer used as labels (the legacy behaviour remains available
via `mask_target_mode = "binary"`). Binary metrics (IoU / Dice / F1) binarise
the soft GT at 0.5 of its normalised range (`utils/metrics.py`).

### Leakage control

The per-image descriptions name the ice type ("glacier", "first-year sea
ice", …), which would let the classifier read its answer from the text
channel. By default (`scrub_class_names = True`) all class names and their
lexical variants are scrubbed from the descriptions and replaced with the
neutral phrase "ice region" before they reach the model, so the descriptions
keep their spatial/textural cues (the part that navigates segmentation) while
classification F1 stays leakage-free. Set `scrub_class_names = False` to feed
the unredacted text (higher F1, but no longer leakage-controlled).

## Previous published results (90-image held-out test split, seed 42)

These were measured by the earlier constant-prompt model **against the old
Otsu-binary GT** — they are the baseline this reasoning-segmentation model is
trained to beat:

| Task | Metric | Previous model | Floor |
|---|---|---|---|
| Segmentation | mIoU | 0.351 | 0.330 (all-foreground, scat GT) |
| Segmentation | cIoU | 0.456 | 0.330 |
| Segmentation | Dice | 0.442 | 0.411 |
| Classification | Accuracy | 0.833 | 0.167 (chance) |
| Classification | Weighted F1 | 0.778 | 0.167 |

Reproduce the degenerate floor under the new scat GT with:

```bash
python baseline_allforeground.py --split test --gt scat   # no GPU / torch needed
python baseline_allforeground.py --split test --gt otsu   # legacy Otsu floor (0.383)
```

## Repository layout

```
config.py                     # single source of truth for all hyperparameters
train.py                      # end-to-end training
evaluate.py                   # test-set evaluation: metrics, confusion matrix, JSON report
inference.py                  # single-image inference + visualisation
baseline_allforeground.py     # degenerate all-foreground baseline (numpy/PIL/cv2 only)
quickstart.py                 # smoke test of the pipeline
data/                         # dataset + SAR preprocessing code
models/                       # pipeline modules (see "What the model is")
utils/                        # losses, metrics
dataset/                      # 600 images: <Class>/{images,masks,descriptions}
reasoning_seg_colab_training.ipynb  # ★ Colab: reasoning-seg training + evaluation + demos
sea_ice_colab_training.ipynb  # legacy Colab notebook (constant-prompt, Otsu-GT model)
comparison_baselines_colab.ipynb  # zero-shot LISA/CLIPSeg/GeoPixel comparison harness
PAPER/                        # LaTeX source of the manuscript
DOCUMENTATION/                # scope/claims documentation (PAPER_SCOPE.md etc.)
```

## Setup

```bash
git clone https://github.com/prakhar443/sea_ice_seg
cd sea_ice_seg
pip install -r requirements.txt
```

Python ≥ 3.10. A CUDA GPU with ≥ 16 GB VRAM is recommended for training
(the published run used a single A100 40 GB, ~45 min for 50 epochs).

## Data layout

`dataset/` contains one folder per class (`Young Ice`, `First Year Ice`,
`Floating Ice`, `Glaciers`, `Icebergs`, `Old Ice`), each with:

- `images/` — native 256×256 grayscale JPEGs (e.g. `1_1003_.jpg`)
- `masks/` — continuous-valued scattering maps, ~139×187 portrait
  (e.g. `1_1003_scat.jpg`). **These raw maps are the ground truth**: at load
  time each is per-image min–max normalised to [0, 1] (before any resize) and
  bilinear-stretched to the image frame (`mask_target_mode="soft_scat"`,
  `mask_resize_mode="stretch"` in `config.py`). The image/mask aspect-ratio
  mismatch introduces geometric label noise that is acknowledged in the paper.
- `descriptions/` — per-image text descriptions (xlsx/csv). **Used by the
  reasoning-segmentation model** as the text that navigates segmentation,
  after class-name scrubbing (see Leakage control above).

## Reproducing

| Artifact | Command |
|---|---|
| Train the reasoning-seg model | `python train.py` (defaults in `config.py`) |
| Main results table | `python evaluate.py --checkpoint outputs/best_model.pth --output eval_results/` |
| All-foreground floor (scat GT) | `python baseline_allforeground.py --split test --gt scat` |
| Full Colab run (train + eval + demos) | notebook `reasoning_seg_colab_training.ipynb` |
| Zero-shot comparison (LISA/CLIPSeg/GeoPixel) | notebook `comparison_baselines_colab.ipynb` |

The train/val/test split is deterministic (stratified per class,
`random.Random(42 + class_index)`, 70/15/15) and is reproduced identically by
`data/dataset.py` and `baseline_allforeground.py`.

## Trained checkpoint

The published checkpoint (`best_model.pth`) is distributed via this
repository's **GitHub Releases** page. After downloading, place it at
`outputs/best_model.pth` and run `evaluate.py` as above.

## Citation

Citation entry will be added upon publication.
