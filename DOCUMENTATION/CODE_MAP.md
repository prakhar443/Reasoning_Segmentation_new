# Code Map — Paper ↔ Repository

This maps each part of the paper to the code that implements it, and states
clearly which modules are part of the **published 5-module model** versus which
are **alternatives we evaluated and excluded** (retained for reproducibility,
not claimed as contributions).

Running `python train.py --data_root dataset --output_dir outputs/exp1` with no
extra flags reproduces the published model (the `config.py` defaults are set to
the published configuration).

---

## Published model — the 5 claimed modules

| Paper section | Module | File(s) | Notes |
|---------------|--------|---------|-------|
| §3.1 SAR Preprocessing | Lee filter, dB, pseudo-RGB | `data/preprocessing.py`, `data/dataset.py` | Infrastructure |
| §3.2 Visual Encoder | CLIP ViT-L/14 + LoRA | `models/visual_encoder.py` | **Claimed (LoRA for classification)** |
| §3.3 Cross-Attention Fusion | Image↔text fusion (`cross_attn_only`) | `models/reasoning_module.py` | Fusion only; no text generation |
| §3.4 Segmentation Decoder | Image-conditioned U-Net + Tversky + aux | `models/sam_module.py` (`ImageUNetDecoder` / `LightweightMaskDecoder`), `utils/losses.py` | **Claimed (Tversky + aux loss)** |
| §3.5 Ice-Type Classifier | 6-class MLP | `models/ice_classifier.py` | **Claimed (joint with seg.)** |
| §3.6 Training | Loop, optimiser, schedule | `train.py`, `config.py` | — |
| §4 Metrics | mIoU/cIoU/Dice/F1 | `utils/metrics.py` | — |
| Orchestration | Full forward pass | `models/pipeline.py` | Wires the 5 modules |

## Alternatives evaluated and excluded (in codebase, NOT claimed)

| Component | File | Why excluded | Default state |
|-----------|------|--------------|---------------|
| DepthAnything V2 encoder | `models/depth_encoder.py` | Adding it produced no measurable change (bit-identical metrics, two runs) | Off |
| Temporal consistency | `models/temporal_consistency.py` | No benefit on single-scene evaluation | `temporal_mode=False` |
| SAM mask decoder | `models/sam_module.py` (`SAMModule`) | Replaced by lighter U-Net decoder; not evaluated in final model | `--use_sam` opt-in only |
| BLIP-2 / LLaVA backends | `models/reasoning_module.py` | Not used; published model uses `cross_attn_only` | `llm_backend='cross_attn_only'` |
| Free-form text generation | — | `cross_attn_only` does not generate language; never implemented as a claim | N/A |

> **Why are these still here?** Several appear as *controls* in the ablation
> study (Type B ablations — "add component X, show it doesn't help"), which is
> the evidence that justifies excluding them. Removing the code would make those
> ablations non-reproducible. They are clearly banner-marked at the top of each
> file and disabled by default.

---

## Reproducing the paper's tables and figures

| Output | How |
|--------|-----|
| Headline results (Table 1) | `sea_ice_colab_training.ipynb` Cell 19, or `python evaluate.py --checkpoint outputs/best_model.pth --split test` |
| Ablation study (Table 2) | Notebook Ablation Cells A + B + C |
| All figures | Notebook Cell 19 → `outputs/figures/` |
| Training curves | Notebook Cell 13 or Cell 19 (needs `train_history.json` from Cell 18b) |

All reported numbers come from `best_model.pth` (val mIoU = 0.4025, 50 epochs).
