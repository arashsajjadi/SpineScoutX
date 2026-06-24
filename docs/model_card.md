# Model card — SpineScoutX

> **Research-only. Not diagnostic, not clinically validated, not for medical
> decision-making.** Not a medical device.

## Overview
SpineScoutX is a research prototype for **disc-level lumbar degenerative finding
grading** with anatomy-grounded evidence. It contains three models plus a
deterministic finding-graph/reporting layer.

| Model | Task | Backbone | Trained on |
|---|---|---|---|
| **E0** image-only classifier | severity grade (normal_mild/moderate/severe) per (level, condition) | ConvNeXt-Tiny (timm) on 2.5D 224² crops + level/condition embeddings | RSNA crops |
| **E4** anatomy segmenter | 4-class anatomy (bg/vertebra/disc/canal) | 2D U-Net | SPIDER slices |
| **E1** anatomy-guided classifier | same grading task as E0, + anatomy-prior channels (disc/canal/vertebra) | image encoder + anatomy encoder + embeddings, **concat fusion** | RSNA crops + SPIDER→RSNA priors |
| **E2** anatomy-forced classifier (v0.5) | same grading task | feature map + **masked region pooling** over anatomy masks + condition→target-region attention + global-feature dropout | RSNA crops + SPIDER→RSNA priors |

## Intended use / out of scope
- **Intended:** research on anatomy-grounded grading, evidence consistency,
  calibration, and finding-graph generation using public research datasets.
- **Out of scope:** any clinical or diagnostic use, treatment decisions, screening,
  or deployment. Findings are limited to the five RSNA label types; no disease
  outside those labels is produced.

## Training
Patient/study-level splits (no leakage); class-weighted CE for the skewed severity
distribution (severe ≈ 6%); frozen-backbone warmup then **gentle fine-tuning**
(backbone lr ×0.2) to avoid an unfreeze shock; AMP; early stopping on val weighted
log loss; deterministic seeds. SPIDER uses its official split.

## Performance (real, held-out val) — see `docs/results.md`
- E0: weighted log loss **0.462**, macro F1 0.706, severe recall **0.751**, severe
  AUROC **0.971**, ECE 0.027.
- E1: weighted log loss 0.458, macro F1 0.717, severe recall 0.711, ECE 0.034.
- E4: mean Dice **0.884** (canal 0.902).
- **Ablation finding:** E1 (concat) barely uses the anatomy prior (zero ≈ shuffle ≈
  correct, |Δwll| < 0.001). **E2 (anatomy-forced) measurably uses it** (zero/noise/
  wrong-region Δwll 0.014–0.020, ~20× E1; `target_region_only` best) and reaches a
  higher severe-recall frontier (sevR 0.855 vs E0 0.751), but is not a free aggregate
  win and does not yet improve AEC. See `docs/results.md` / `run_logs/e2_ablation_results.md`.
- **The numbers above are oracle-crop UPPER BOUNDS (GT localizer coordinates).** On the
  real **auto-localized** canal distribution the deployed E0 severe recall is 0.644.
  The oracle→auto collapse is entirely **in-plane crop-centre** error (2×2: in-plane
  −0.184 decisive; slice +0.011 n.s.). **Robust auto-inference** — training the grader
  on auto-localized crops — recovers it: auto severe recall **0.793 [0.696, 0.881]**, a
  paired **+0.149 [+0.050, +0.243]** over E0 (decisive; McNemar p=0.007), ~81% of the
  gap, with better auto log loss. **Safety Mode** (auto): recall@FAR≤10% 0.851 [0.766,
  0.924]; 90% severe recall at FAR 0.153 or ~20% review burden. See
  `run_logs/robust_auto_experiments.md`, `run_logs/safety_mode.md`.

## Evidence & calibration
Grad-CAM heatmaps → Anatomical Evidence Consistency (AEC; mean ≈ 0.10, flat across
ablation modes). Canal-stenosis evidence uses the real canal mask (`anatomy`);
foraminal/subarticular regions are **approximate** (SPIDER has no such labels) and
flagged. ECE + reliability + optional temperature scaling; per-finding uncertainty
flags.

## Limitations & ethics
Single-crop-per-localizer modeling (no full-study context); anatomy priors are
anatomy (not pathology) masks; approximate foraminal/lateral-recess regions; no
external or clinical validation; results are from one site's public data under
non-commercial terms. No PHI; no identifiers in figures; no data/weights committed.
Optional LLM (Ollama) report wording is fail-closed and may only rephrase the
deterministic finding graph.
