# Sea Ice SAR: Joint Segmentation & Ice-Type Classification

Language-conditioned multimodal baseline for **joint** pixel-level sea-ice
segmentation and six-class ice-type classification from single-band SAR
imagery. Companion code for the MDPI Remote Sensing submission
*"Towards Language-Conditioned Sea-Ice Understanding: Joint Segmentation and
Ice-Type Classification from SAR Imagery"*.

## What the published model is (and is not)

The published pipeline (`llm_backend = "cross_attn_only"` in `config.py`) is:

1. **SAR preprocessing** — Lee filter (7×7) → dB conversion → pseudo-RGB
   (`data/preprocessing.py`)
2. **CLIP ViT-L/14 + rank-8 LoRA** on Q/K/V/O projections — 1.57 M trainable
   of 304 M (`models/visual_encoder.py`)
3. **Cross-attention text–visual fusion** — CLIP text embeddings as K/V,
   visual patch tokens as Q (`models/reasoning_module.py`)
4. **Image-conditioned U-Net decoder** with auxiliary deep supervision
   (`models/sam_module.py::ImageUNetDecoder`)
5. **6-class MLP classifier** (`models/ice_classifier.py`)

It contains **no LLM and generates no language**. BLIP-2 / LLaVA backends, a
SAM decoder, a DepthAnything V2 branch, and a temporal-consistency module
exist in the codebase as evaluated-and-excluded alternatives (kept for
ablation reproducibility); none contributes to any reported number.

### Leakage control

The published run uses a **fixed, class-agnostic text prompt for every image**
(`use_class_name_in_prompt = False` in `config.py`; see
`data/dataset.py` lines ~330–349). The text input is identical across all
images and classes, so classification scores cannot arise from label leakage
through the text channel. The per-image annotator descriptions shipped in
`dataset/*/descriptions/` are **not** used by the published model.

## Headline results (90-image held-out test split, seed 42)

| Task | Metric | Model | Degenerate floor |
|---|---|---|---|
| Segmentation | mIoU | 0.351 | **0.383** (all-foreground) |
| Segmentation | cIoU | **0.456** | 0.383 |
| Segmentation | Dice | 0.442 | **0.480** |
| Segmentation | Pixel accuracy | **0.656** | 0.383 |
| Classification | Accuracy | **0.833** | 0.167 (chance) |
| Classification | Weighted F1 | **0.778** | 0.167 |

**Read honestly:** the classifier clearly beats chance with a leakage-free
constant prompt; the segmentation does **not** exceed the all-foreground floor
on per-image mIoU/Dice (it does on cIoU, pixel accuracy and precision). The
floor exists because the Otsu-binarised `_scat` scattering maps average ~38 %
foreground; see the paper's "Proximity to the All-Foreground Floor" section.
Reproduce the floor with:

```bash
python baseline_allforeground.py --split test   # no GPU / torch needed
```

## Repository layout

```
config.py                     # single source of truth for all hyperparameters
train.py                      # end-to-end training (reproduces the published run)
evaluate.py                   # test-set evaluation: metrics, confusion matrix, JSON report
inference.py                  # single-image inference + visualisation
baseline_allforeground.py     # degenerate all-foreground baseline (numpy/PIL/cv2 only)
quickstart.py                 # smoke test of the pipeline
data/                         # dataset + SAR preprocessing code
models/                       # pipeline modules (see "What the published model is")
utils/                        # losses, metrics
dataset/                      # 600 images: <Class>/{images,masks,descriptions}
sea_ice_colab_training.ipynb  # Colab notebook: training + ablations + figures
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
  (e.g. `1_1003_scat.jpg`). Binary masks are derived at load time:
  per-image Otsu threshold **before** any resize, then nearest-neighbour
  stretch to the image frame (`mask_binarize="otsu"`,
  `mask_resize_mode="stretch"` in `config.py`). These are **derived labels**,
  not expert annotations; the image/mask aspect-ratio mismatch introduces
  geometric label noise that is acknowledged in the paper.
- `descriptions/` — per-image text descriptions (xlsx/csv). **Unused by the
  published model** (see Leakage control above).

## Reproducing the paper

| Artifact | Command |
|---|---|
| Train the published model | `python train.py` (defaults in `config.py` are the published configuration) |
| Main results table | `python evaluate.py --checkpoint outputs/best_model.pth --output eval_results/` |
| All-foreground floor | `python baseline_allforeground.py --split test` |
| Ablation table | notebook `sea_ice_colab_training.ipynb`, section 18 |
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
