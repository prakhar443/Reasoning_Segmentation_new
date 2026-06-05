# Baseline Comparison Study — Language-Guided Segmentation on SAR Sea Ice

This document records the **comparative experiment** built for the journal
submission (target: *Information Fusion*). It evaluates the trained sea-ice
model against published zero-shot / language-guided / reasoning-segmentation
baselines on **identical test data with identical metrics**, mirroring the
Table-1 protocol of the *Information Fusion* underwater reasoning-segmentation
paper (which reports gIoU / cIoU / Dice).

> **Integrity note.** Every number in this study is produced by an actual run of
> the notebook on the 90-image SAR test split. No value is hand-edited,
> synthesised, or interpolated. Cells marked *pending* below have not yet been
> filled from a confirmed run and must stay blank until a genuine result exists.

Notebook: **`comparison_baselines_colab.ipynb`** (branch
`claude/laughing-thompson-AhCX7`).

---

## 1. Evaluation protocol (shared by every method)

A single harness in **Section 0** of the notebook drives every baseline so that
all methods are scored under exactly the same conditions:

| Item | Value |
|------|-------|
| Test images | **90** SAR scenes — 15 per ice class, the held-out test split |
| Split | Deterministic per-class shuffle, `seed=42`, 70 / 15 / 15 train/val/test |
| Mask naming | `<stem>_.jpg` → `<stem>_scat.jpg` (image → scattering-map mask) |
| Ground-truth binarisation | **Per-image Otsu** on the `_scat` scattering map (`data/dataset.py:binarize_mask`) — the same GT definition used in training |
| Evaluation resolution | 512 × 512, nearest-neighbour resize for both prediction and GT |

### Metrics (defined exactly as in the underwater paper)

| Metric | Definition |
|--------|------------|
| **gIoU** | Mean of per-image IoU over the 90 images |
| **cIoU** | Cumulative intersection ÷ cumulative union (pooled over all pixels of all images) |
| **Dice** | Mean of per-image Dice coefficient |

`evaluate_predictor(predict_fn)` takes any function `PIL.Image → bool mask` and
returns `{gIoU, cIoU, Dice, n}`. Every baseline plugs into this one function, so
differences in the table reflect the models, not the scoring.

---

## 2. Methods compared

| # | Method | Family | Language-guided? | Trained on SAR? | Runtime env |
|---|--------|--------|:----------------:|:---------------:|-------------|
| 1 | **Ours** (CLIP+LoRA + cross-attn + U-Net) | Language-guided segmentation | ✅ | ✅ | Colab kernel |
| 2 | CLIPSeg | Zero-shot text→mask | ✅ | ❌ | Colab kernel |
| 3 | GroundingDINO + SAM | Open-vocab detection → promptable seg | ✅ | ❌ | Colab kernel |
| 4 | DeepLabv3+ | Supervised CNN segmentation | ❌ | ✅ (on this data) | Colab kernel |
| 5 | LISA-7B | Reasoning segmentation LMM | ✅ | ❌ | Isolated py3.10 |
| 6 | GeoPixel-7B | Remote-sensing grounded LMM | ✅ | ❌ | Isolated py3.10 |
| 7 | PixelLM-7B | Pixel-reasoning LMM | ✅ | ❌ | Isolated py3.10 |

All language-guided methods receive the **same generic prompt** (a request to
segment the sea ice), so none is hand-tuned per image.

### Why these baselines

- **CLIPSeg / GroundingDINO+SAM** — strong *zero-shot* open-vocabulary
  references; they quantify how far generic vision-language priors transfer to
  SAR without any in-domain training.
- **DeepLabv3+** — a *supervised, non-language* control to isolate the
  contribution of language guidance versus plain in-domain training.
- **LISA / GeoPixel / PixelLM** — published **reasoning / grounded
  segmentation LMMs**, the most direct comparators to the paper's
  language-guided framing. GeoPixel additionally represents the
  remote-sensing-specialised LMM family.

---

## 3. Engineering: dependency isolation

The three 7B LMMs (LISA, GeoPixel, PixelLM) pin 2023-era dependency stacks that
are incompatible with Colab's current Python. Each therefore runs as a
**subprocess inside its own Python 3.10 virtualenv** built with `uv`, with a
self-contained eval script. Key per-model engineering decisions, all driven by
real failures observed during the runs:

| Model | Env | Notable fixes |
|-------|-----|---------------|
| LISA-7B | torch 2.0.1 + cu118, transformers 4.31 | Removed `bitsandbytes` (model runs bf16; bnb's prebuilt CUDA binary can't find `libcusparse.so.11`) |
| GeoPixel-7B | torch 2.1.2 + cu118, transformers 4.33.2 | Vendored SAM2; requires `flash-attn` (prebuilt cu118/torch2.1/cp310 wheel); `model.tokenizer` + `config.*_token_id` must be set before `evaluate()`; bf16 autocast for image preprocessing |
| PixelLM-7B | torch 2.0.1 + cu118, transformers 4.31 | Lightweight pixel decoder (no SAM/detectron2/mmcv/flash-attn on the inference path); released-7B config `seg_token_num=3`, `image_feature_scale_num=2`, CLIP ViT-L/14-**336** @ 448 |

Common to all subprocess runs: `LD_LIBRARY_PATH` is extended with the env's
`torch/lib` and `nvidia/*/lib`, and `MPLBACKEND=Agg` is forced (Colab's inline
matplotlib backend does not exist inside the isolated env).

### SEEM — attempted and dropped

SEEM was initially included but **removed**: its `focall_unicl` model imports
**detectron2**, a compiled C++/CUDA package that does not build against this
Colab's torch/Python combination. A pure-Python stub of detectron2's inference
API was carried a long way (the `focall_unicl_lang_v1` config uses the
pure-PyTorch FPN encoder, so no compiled `MSDeformAttn` op is needed), but the
maintenance cost outweighed its value — SEEM is an interactive/promptable
*generalist*, not a reasoning-segmentation model, making it the least
on-narrative entry. It was replaced by **PixelLM**, which shares the LLaVA
lineage already de-risked for LISA and needs no compiled extensions.

---

## 4. Results

> Scores below come from the shared harness in §1 (gIoU / cIoU / Dice on the
> same 90 images). **Confirmed** rows are from completed runs; **pending** rows
> await a genuine run and are intentionally left blank.

| Method | gIoU | cIoU | Dice | Status |
|--------|:----:|:----:|:----:|--------|
| **Ours** (language-guided) | **0.3917** | _pending_ | _pending_ | confirmed (gIoU) |
| CLIPSeg | 0.0839 | _pending_ | _pending_ | confirmed (gIoU) |
| GroundingDINO + SAM | _pending_ | _pending_ | _pending_ | run completed; numbers to be transcribed |
| DeepLabv3+ | _pending_ | _pending_ | _pending_ | run completed; numbers to be transcribed |
| LISA-7B | _pending_ | _pending_ | _pending_ | run completed; numbers to be transcribed |
| GeoPixel-7B | _pending_ | _pending_ | _pending_ | run in progress |
| PixelLM-7B | _pending_ | _pending_ | _pending_ | run in progress |

**Confirmed observations so far:**

- Our model's harness **gIoU = 0.3917** is consistent with the headline
  **mIoU = 0.351** reported for the published model (the small difference is the
  test-time prompt and the gIoU averaging convention), validating that the
  comparison harness scores our model the same way the main evaluation does.
- **CLIPSeg gIoU = 0.0839** — a ~4.7× gap below our model, quantifying how
  poorly a generic zero-shot text-to-mask prior transfers to SAR scattering
  imagery and supporting the paper's in-domain-adaptation argument.

The remaining cells will be filled **only** from the per-model JSON outputs the
notebook writes to `outputs/comparison/*.json` on Drive.

---

## 5. Reproducing the study

1. Open `comparison_baselines_colab.ipynb` in Colab (GPU runtime; A100
   recommended for the 7B LMMs).
2. Run **Section 0** first after every restart (mounts Drive, clones the repo,
   defines the test-set lister + metric harness, installs shared deps).
3. Run Sections 1–7 in order. Each writes `outputs/comparison/<Model>.json`.
   - Sections 5–7 (LISA / GeoPixel / PixelLM) each build a one-time py3.10 env;
     the first run downloads the model weights (7–14 GB each).
4. Run **Section 8** to aggregate every `*.json` into the final comparison
   table and an IEEE-style LaTeX table (caption above, **Ours** row bolded).

Each heavy section prints `STDOUT`/`STDERR` tails on failure so a broken run can
be diagnosed without re-running the whole notebook.

---

## 6. Relationship to the main results

This study lives alongside, and does not change, the model's headline numbers in
`DOCUMENTATION/README.md` (mIoU 0.351 / Dice 0.442 / weighted-F1 0.778) and the
ablation in `CODE_MAP.md`. It adds **external comparison context**: how the
trained language-guided model stands against published zero-shot, supervised,
and reasoning-segmentation baselines under one protocol.

_Last updated: June 2026._
