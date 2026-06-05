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
| 3 | DeepLabv3+ | Supervised CNN segmentation | ❌ | ✅ (on this data) | Colab kernel |
| 4 | LISA-7B | Reasoning segmentation LMM | ✅ | ❌ | Isolated py3.10 |
| 5 | GeoPixel-7B | Remote-sensing grounded LMM | ✅ | ❌ | Isolated py3.10 |
| 6 | PixelLM-7B | Pixel-reasoning LMM | ✅ | ❌ | Isolated py3.10 |

All language-guided methods receive the **same generic prompt** (a request to
segment the sea ice), so none is hand-tuned per image.

### Why these baselines

- **CLIPSeg** — a strong *zero-shot* open-vocabulary reference; quantifies how
  far a generic text-to-mask vision-language prior transfers to SAR without any
  in-domain training.
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

All scores below come from the shared harness in §1 (gIoU / cIoU / Dice on the
same 90 images), transcribed verbatim from the per-model JSON the notebook wrote
to `outputs/comparison/*.json`. Values are raw [0, 1] (multiply by 100 for the
percentages used in the LaTeX table).

| Method | gIoU | cIoU | Dice | n | Notes |
|--------|:----:|:----:|:----:|:-:|-------|
| **Ours** (language-guided) | **0.3917** | **0.5105** | **0.4853** | 90 | confirmed |
| LISA-7B | 0.2549 | 0.2904 | 0.3490 | 90 | confirmed |
| CLIPSeg | 0.0839 | 0.1226 | 0.1268 | 90 | confirmed |
| GeoPixel-7B | 0.0316 | 0.0350 | 0.0597 | 90 | confirmed |
| DeepLabv3+ | 0.0008 | 0.0010 | 0.0016 | 90 | ⚠️ **invalid — training bug, see finding (a)** |
| PixelLM-7B | _pending_ | _pending_ | _pending_ | — | not yet run (replaced SEEM) |

> A GroundingDINO+SAM (Grounded-SAM) baseline will be added separately by the
> authors later; it is intentionally **not** included here.

### Findings (reported honestly)

**(a) The DeepLabv3+ row is NOT a valid baseline and must not be reported as
one.** Its training `Dataset` (`_SegDS`) loads masks as `masks/<image_name>.jpg`
instead of the real `masks/<stem>_scat.jpg`, so every training mask resolved to
an all-zero array. DeepLab therefore trained to predict empty masks and scores
≈0. This is a **data-loader bug in the baseline harness**, not a property of
DeepLabv3+. It needs the `_scat` mask path (and Otsu binarisation) wired into
`_SegDS`, then a re-run, before it can stand as the supervised, no-language
control. Until then this row is excluded from any claim.

### Ranking of the valid confirmed runs (by gIoU)

1. **Ours** — 0.3917  (also **#1 on cIoU**, 0.5105, and Dice, 0.4853)
2. LISA-7B — 0.2549
3. CLIPSeg — 0.0839
4. GeoPixel-7B — 0.0316

**Reading the LMM baselines:** LISA (general reasoning-seg) transfers best of the
three 7B LMMs; **GeoPixel underperforms despite being remote-sensing-specialised**
(0.0316) — its optical-RS pretraining does not transfer to SAR scattering maps,
a useful point for the discussion. PixelLM is pending.

> No number in this table is hand-edited. The DeepLab row is shown only so the
> bug is on the record; it is flagged invalid rather than silently dropped.

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
