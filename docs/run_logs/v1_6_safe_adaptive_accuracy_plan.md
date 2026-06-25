# v1.6 — Safe Adaptive Accuracy Offensive: plan

> **Research-only · non-commercial · not diagnostic · not clinically validated · not for medical
> decision-making.** Held-out reference = RSNA graded severity. Protocol: `splits_v1`
> (train 1382 / dev 296 / test 296 studies). **Dev/train select; RSNA locked-test read once per
> final selected system.** No held-out/test labels as model input. Data + weights gitignored.

## Goal

Raise **raw severe grading** on the weak routes — especially **right-foraminal severe recall**
and **foraminal macro** — by changing strategy away from internal-only tricks toward **external
data + representation learning + anatomy priors + stronger route-specific graders**, switching
plans adaptively until a real improvement lands or every valid path is executed/hard-blocked with
proof.

Baseline (locked-test auto severe recall): canal 0.830 · L-for 0.788 · **R-for 0.660** · L-sub
0.746 · R-sub 0.737 · **macro 0.752**.

## Why v1.5 did not improve grading (and why internal MIL/localizer-only is deprioritized)

- **Candidate-bag MIL** (R-for + subarticular): executed negative. R-for *won dev* recall@FAR10
  (+0.125) but did not generalize (test severe recall 0.660→0.453) — thin severe (n≈50) → dev
  overfit. Subarticular collapsed (→0.000). More candidate crops do not beat the robust graders.
- **BiGRU axial localization**: decisive localization win (±1-hit 0.487→0.616, medAE 2→1) that
  **did not propagate to grading** (recrop→regrade test Δ−0.040 n.s.). The grader is
  leveling-invariant.
- **Conclusion (v1.4+v1.5, convergent):** the weak routes are **representation / data / grader-
  capacity limited**, NOT crop/localizer limited. Repeating internal MIL/localizer-only is
  therefore deprioritized; it only re-enters combined with new external data, pretraining, or a
  stronger representation.

## Plan ladder (adaptive A→B→C→D; dev decides each switch)

**Plan A — External foraminal data (LSS-MRI AISSLab).** First target because the proven
bottleneck is foraminal severe *data/representation*. LSS = 500 patients / 8,500 sagittal slices /
3,885 L1–S1 foraminal boxes with side (LFS/RFS), level, and grade (Normal/Mild/Moderate/Severe);
CC BY 4.0 (non-commercial research). **Key caveat discovered up-front:** only **47 Severe** boxes
total — so LSS is a **foraminal representation / encoder-pretraining** resource (rich Normal/Mild/
Moderate morphology + precise boxes), **not** a severe-label windfall. Use modes: (a) supervised
LSS foraminal pretraining → RSNA fine-tune; (b) joint LSS+RSNA with a dataset embedding; (c) LSS
encoder-only pretraining. Success = RSNA dev right-foraminal / foraminal-macro severe recall (or
recall@FAR≤10%) up → confirm on locked-test once.

**Plan B — Self-supervised spine-MRI pretraining.** If LSS is blocked or LSS-transfer fails. Learn
a foraminal/subarticular representation from legal unlabelled data (RSNA train/dev sagittal-T1 /
axial-T2 + SPIDER + LSS slices if present) via contrastive + route-prediction + level-position
auxiliary tasks (RSNA locked-test excluded from SSL for headline metrics), then fine-tune the
route-specific graders. Rationale: representation is the named bottleneck; SSL is the
data-efficient lever when labelled severe is scarce.

**Plan C — Anatomy-prior grader (SPIDER).** If SSL doesn't move grading. Make the grader
*anatomically smarter*: generate vertebra/disc/canal anatomy-prior channels (SPIDER-derived, no
test labels in generation) for RSNA foraminal/subarticular crops; train image+prior vs image-only
vs shuffled/zero-prior controls (ablation gap = does it USE the prior). Earlier anatomy work
(v0.4/v0.5) targeted canal; foraminal/subarticular anatomy-prior grading is the new angle.

**Plan D — Stronger route-specific grader + severe curriculum.** If A–C fail. Stronger backbone
(timm ConvNeXt/EfficientNet/Swin — `timm` available), freeze/unfreeze schedule, severe curriculum
(normal/mod/severe → hard severe-vs-nonsevere), label-smoothing/ordinal/focal/class-balanced/
severe-FN-weighted losses with FAR guardrails, optionally warm-started from the Plan-B SSL encoder
(so it is *not* internal-only). Dev selects; locked-test once.

**Adaptive controller** records every experiment (id, train/dev/test usage, metrics, decision,
next action) and never hides a failed run.

## Safety & rollback policy

- Commit after every safe milestone. Local rollback tags (pushed only if needed):
  `v1.6-pre-external-data` (set), `v1.6-pre-pretraining`, `v1.6-pre-final-eval`.
- **Never commit** DICOM/NIfTI/masks/raw images/checkpoints/weights/runs/outputs/caches/large
  artifacts; `data/external/` is gitignored. No patient identifiers. Repo private.
- Allowed terms only (finding, severity estimate, P(severe), held-out reference, severe recall,
  recall@FAR, false negative, uncertainty, review_required, non-diagnostic). No diagnosis/treatment/
  clinical-deployment claims.

## Locked-test policy

Dev/train for ALL model selection and plan switching. RSNA locked-test evaluated **once per final
selected system**; every locked-test read is counted and reported. LSS has its own train/dev; LSS
test is never used to select an RSNA model.

## Success thresholds (≥1 required, else executed autopsy)

1. R-for severe recall **+≥0.05** abs; or 2. foraminal macro **+≥0.03**; or 3. overall macro
**+≥0.03**; or 4. high-confidence severe FN **−≥20%**; or 5. recall@FAR≤10% materially up for
R-for / foraminal macro; or 6. all paths executed/hard-blocked with exact evidence + next-data
requirements (rigorous autopsy).

## Tagging

`v1.6.0-external-foraminal-accuracy-upgrade` (LSS transfer wins) · `…-representation-accuracy-
upgrade` (SSL wins) · `…-anatomy-prior-upgrade` (anatomy wins) · `…-adaptive-accuracy-negative-
result` (all execute/block, no raw gain). **No accuracy-upgrade tag without a real metric gain.**
