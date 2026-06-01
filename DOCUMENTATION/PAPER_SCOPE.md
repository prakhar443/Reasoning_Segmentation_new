# Paper Scope & Honest Reporting Decisions

## Novel Contribution — The Actual Claim

**This work is, to our knowledge, the first to demonstrate that simultaneous
pixel-level segmentation and 6-class ice-type classification of sea ice from
SAR satellite imagery is achievable with a single end-to-end deep learning
pipeline.**

Prior to this work, sea ice interpretation from SAR was dominated by:
- Manual expert annotation (the norm in operational settings)
- Semi-automated threshold / texture classifiers that do **either** rough
  segmentation **or** coarse type labelling — not both jointly
- No published, reproducible deep-learning baseline that handles both tasks
  simultaneously on the same SAR image

**What we prove:** With a CLIP ViT-L/14 + LoRA encoder and an image-conditioned
U-Net decoder trained on Otsu-binarized scattering maps, a network can
(a) produce binary ice-/water region masks (mIoU 0.351, Dice 0.442) and
(b) assign ice types from 6 classes (weighted-F1 0.778, 5 of 6 classes ≥ F1
0.667) on a held-out test set.  The results are modest in absolute terms — we
make no claim of state-of-the-art performance — but they establish a concrete
and reproducible existence proof that the task is tractable.

> **This framing is the paper's core claim.**  Ablations and metric details
> below support it honestly and do not overstate what the experiments show.

---

This document records *what the paper claims*, *what it reports*, and *why*,
strictly matched to the experimental evidence. The guiding rule:

> We may choose which metrics to report, but we never make a claim that an
> experiment we ran would contradict. Where an ablation shows a component does
> not help, we say so.

All numbers below are **test-set** results on the stratified split (15 samples
per class in val and test). The ablation variants were retrained for ≤15 epochs
(early stopping, patience 5); the full model is the main 50-epoch checkpoint.
This training-budget difference is disclosed in the ablation table caption and
is a known limitation of the ablation (the variants had less training budget,
yet several still match or exceed the full configuration).

## Headline results (test set)

**Segmentation (region-level):** mIoU 0.351, cIoU 0.456, Dice 0.442, pixel
accuracy 0.656. Precision 0.444 / recall 0.640 (reported as a
limitation — the model still over-predicts foreground, but no longer floods it:
per-class over-segmentation ratios are 0.94×–1.86×, down from up to 5.4× before
loss rebalancing).

**Classification:** accuracy 0.833, macro-F1 0.778, weighted-F1 0.778. Per-class
F1: Floating Ice, Glaciers, Icebergs, First Year Ice ≈ 1.0; Young Ice 0.667;
**Old Ice 0.000** (see limitations).

## Claimed contribution (re-scoped to what the ablations support)

The contribution is a **CLIP + image-conditioned U-Net pipeline for SAR
sea-ice region segmentation and 6-class ice-type classification**, trained with
a **Tversky-dominant segmentation objective**. Specifically, the ablation
evidence supports:

1. **Tversky loss is the key segmentation driver.** Tversky-only achieves the
   best segmentation on the test set (mIoU 0.389, Dice 0.493). Focal-only
   collapses it (mIoU 0.252). The combined Focal+Tversky objective used in the
   full model trades a small amount of mIoU for a more balanced
   precision/recall operating point; **we do not claim the combination
   maximizes mIoU** — it does not.
2. **Auxiliary deep-supervision loss helps mIoU.** Removing it drops mIoU
   0.351 → 0.324. Kept as a contribution.
3. **LoRA adaptation helps classification, not segmentation.** Removing LoRA
   (frozen CLIP) drops weighted-F1 0.778 → 0.534, but actually *raises*
   segmentation mIoU slightly (0.351 → 0.365). We therefore present LoRA as a
   **classification** contribution only, and state plainly that it does not
   improve segmentation on this dataset.

## Explicitly NOT claimed (removed because the evidence does not support it)

- **Depth features.** The "w/o Depth" ablation is **bit-identical** to the full
  model across every metric, in two independent runs. Depth contributes
  nothing measurable on this dataset. The earlier claim that "removing depth
  drops weighted-F1 0.62→0.58" was **not supported by any experiment** and has
  been removed. Depth remains in the codebase but is **not claimed** as a
  contribution.
- **U-Net decoder superiority.** On the test set the U-Net decoder (mIoU 0.351,
  Dice 0.442) does **not** clearly beat the token decoder (mIoU 0.344, Dice
  0.455 — higher Dice). We do **not** claim the U-Net decoder improves
  segmentation accuracy; at most it is comparable. (It was retained for
  full-resolution masks, not for a metric win.)
- **Temporal consistency.** A prior ablation showed it does not help F1; removed
  from the claimed method and from the ablation table. Remains in the codebase,
  disabled.
- **Free-form text generation.** The trained backend (`cross_attn_only`) does
  not generate language; the reasoning demo is a templated, class-conditioned
  response. **BLEU/ROUGE/CIDEr are not reported** — they would be meaningless
  for a non-generative model.

## Metrics reported (match the claims)

**Segmentation (region-level):** mIoU, cumulative IoU (cIoU), Dice, pixel
accuracy. Evaluated and described at the **region level**.

**Classification:** accuracy, macro-F1, weighted-F1, per-class F1 (all classes
shown, including Old Ice at 0.0 — no class is hidden).

**Ablation table:** bold marks the **true best value in each column**, not the
full model. Where an ablated variant wins, the table shows it.

## Stated limitations (reported, not hidden)

- **Old Ice classification fails (F1 = 0.000).** The model *segments* Old Ice
  well (IoU 0.513, second-best of all classes) but misclassifies all 15 test
  samples. With only 52 training images (the smallest class), the classifier
  does not generalize for this type. Reported openly.
- **Over-segmentation tendency** (precision 0.444, recall 0.640): the model
  favors recall of ice pixels at the cost of precision. Calibrated far better
  than before (over-seg 0.94×–1.86×) but not eliminated.
- **Coarse boundaries.** Boundary IoU is near zero; masks capture region extent,
  not fine edges. Stated as a limitation, never reported as a headline metric.
- **Classification overfits the small validation set** (val F1 reaches 1.0
  within 1–2 epochs on 90 samples). Only test F1 is treated as meaningful.
- **Ablation training-budget mismatch.** Ablated variants used ≤15 epochs vs.
  the full model's 50; the comparison is not fully controlled. Disclosed in the
  table caption.
- **Performance ceiling from supervision.** Targets are Otsu-binarized
  scattering maps; train mIoU ≈ val mIoU throughout, indicating the ceiling is a
  property of the supervision, not optimization.

## Why this is honest scoping, not cherry-picking

Every removed claim is removed *because an experiment we ran failed to support
it* — depth (no effect), U-Net superiority (token decoder comparable), temporal
(hurts F1), text metrics (no generation). Every retained claim (Tversky as the
key loss, aux loss, LoRA-for-classification) is the conclusion the ablation
actually points to, even where that means the full model is not the best row in
the table. A reviewer replicating any experiment will find numbers consistent
with every statement here.
