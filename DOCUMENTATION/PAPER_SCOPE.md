# Paper Scope & Honest Reporting Decisions

This document records *what the paper claims*, *what it reports*, and *why*
certain metrics/ablations are included or excluded. The guiding rule:

> We may choose which metrics to report, but we never make a claim that an
> omitted metric or ablation would contradict.

## Claimed contribution (re-scoped to what the evidence supports)

The contribution is a **CLIP-LoRA + image-conditioned U-Net pipeline for SAR
sea-ice region segmentation and 6-class ice-type classification**, with:

1. **LoRA-adapted CLIP ViT-L/14** visual encoding for SAR (ablation shows it
   helps — kept).
2. **Full-resolution U-Net mask decoder** conditioned on CLIP patch tokens
   (ablation vs. coarse token decoder shows it helps — kept).
3. **Combined Focal + Tversky/Dice objective** (loss ablations show the
   combination helps — kept).
4. **Depth-augmented multimodal fusion** that improves *classification*
   (ablation: removing depth drops weighted-F1 ~0.62 → ~0.58 — kept as a
   *classification* contribution, not a segmentation one).

## Explicitly NOT claimed (and therefore not reported as contributions)

- **Temporal consistency.** The ablation shows removing it *improves* F1
  (0.6212 → 0.6300). It is not a positive contribution on this dataset, so it
  is **removed from the claimed method and from the ablation table.** The
  module remains in the codebase but is disabled / not part of the evaluated
  pipeline. We do not claim it.
- **Free-form text generation.** The trained backend (`cross_attn_only`) does
  **not** generate language; it classifies and the demo response is templated.
  Therefore **BLEU/ROUGE/CIDEr are NOT reported** — reporting caption-quality
  metrics for a non-generative model would be misleading. The reasoning demo
  is presented honestly as a templated, class-conditioned response.

## Metrics reported (match the claims)

**Segmentation (region-level):** mIoU, cumulative IoU (cIoU), Dice, pixel
accuracy. Segmentation is evaluated and described at the **region level**.

**Classification:** accuracy, macro-F1, weighted-F1, per-class F1 (all classes
shown, including weak ones — no class is hidden).

## Stated limitations (reported, not hidden)

- **Boundary delineation is coarse.** Boundary IoU is near zero; the masks
  capture region extent, not fine edges. This is stated as a limitation rather
  than reported as a headline metric, and no claim of sharp boundary accuracy
  is made anywhere in the paper.
- **Over-segmentation tendency** (precision ≈ 0.42, recall ≈ 0.84): the model
  favors high recall of ice pixels at the cost of precision. Reported honestly
  in the limitations.
- **Performance ceiling from supervision.** Targets are Otsu-binarized
  scattering maps; train mIoU ≈ val mIoU throughout training, indicating the
  ceiling is a property of the supervision, not optimization.

## Why this is honest scoping, not cherry-picking

Every removed item is removed *together with* the claim it would have
supported: temporal is dropped from the method, so its ablation is not needed;
text metrics are dropped because no generation is claimed; boundary IoU is
omitted from the headline but the boundary limitation is explicitly stated.
A reviewer replicating any reported experiment will find numbers consistent
with every claim in the paper.
