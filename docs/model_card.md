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
- **v1-track LOCKED TEST (splits_v1):** the canal auto result is confirmed out-of-val —
  auto severe recall **0.830 [0.725, 0.929]** (auto-trained) vs **0.434** (oracle-trained
  control), paired **+0.396 [+0.268, +0.529]**, ~96% of the oracle ceiling. Multi-condition
  locked-test **oracle** baselines (upper bounds): canal 0.925, L/R foraminal 0.769/0.811,
  L/R subarticular 0.790/0.854 severe recall. **Auto-localization exists only for canal**
  (sagittal-T2); foraminal (sagittal-T1) and subarticular (axial-T2) need view-specific
  localizers (documented frontier). **Default deployable research grader = the canal
  auto-trained robust E0**; non-canal conditions are oracle-bound until their localizers
  exist. See `run_logs/canal_locked_test.md`, `run_logs/multicondition_robust_results.md`,
  `run_logs/safety_mode_v2.md`.
- **v0.13–v0.15 five-finding auto (LOCKED TEST): coverage 1/5 → 3/5.** Foraminal L/R now
  have a real auto route (sagittal-T1 side-aware localizer, median 2.2 px, crop-hit 0.999).
  Deployable foraminal auto severe recall: left **0.788 [0.673, 0.892]**, right
  **0.660 [0.524, 0.788]** — and the deployable foraminal grader is **oracle-trained**
  (robust auto-training hurt foraminal, because its clean localizer makes the oracle→auto
  gap small — opposite of canal). **Default deployable graders: canal = auto-trained robust;
  foraminal = oracle-trained.** Subarticular L/R remain a **measured blocker** (axial
  z-level-matching only 27.5% within ±1 slice). Safety Mode v3 covers all 3 auto conditions.
  See `run_logs/foraminal_auto_results.md`, `run_logs/safety_mode_v3.md`, `run_logs/report_v3.md`.
- **v0.16–v1.0 five-finding auto (LOCKED TEST): coverage 3/5 → 5/5.** Subarticular UNLOCKED
  via a coordinate-supervised axial level scorer (±1 slice-hit 0.43 vs geometry 0.275) +
  fixed in-plane offset + robust auto-training. Deployable subarticular auto severe recall:
  left **0.746 [0.674, 0.815]**, right **0.737 [0.667, 0.807]** (auto-trained robust; the
  oracle-trained grader collapses to 0.25/0.37 — paired +0.50/+0.37, McNemar p<1e-12). **All
  five findings now have real auto locked-test results.** Default deployable graders (router):
  canal & subarticular = auto-trained robust; foraminal = oracle-trained. Safety Mode v4 +
  router cover 5/5. Honest caveats: right-foraminal weakest; subarticular relies on the grader
  tolerating an imperfect level scorer. See `run_logs/subarticular_auto_results.md`,
  `run_logs/safety_mode_v4.md`, `results.md`.

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
