# v1.8c — REAL MedSAM2 + VisionServeX Spine Morphometry Offensive: plan

> **Research-only · non-commercial · not diagnostic · not clinically validated.** Protocol:
> `splits_v1`; **dev/train select**; **RSNA locked-test read once** for the final frozen candidate.
> HF/Kaggle tokens used locally only — **never printed/committed**. All weights/data/masks/features
> **gitignored**.

## Why this sprint (correction after v1.8b)

v1.8b **downloaded** MedSAM2 but the `sam2` package was missing, so it ran **generic Transformers
SAM2.1** and called the morphometry pipeline on those masks. That was **not** a real MedSAM2
experiment. v1.8c **must execute real MedSAM2** (official `sam2` stack + the MedSAM2 checkpoint)
**first**, then judge whether medical segmentation-morphometry helps grading.

## Hard requirements

- **Run real MedSAM2** (real model code + checkpoint, proven by module path + checkpoint hash in the
  smoke report). `sam2` is pip-installable (v1.1.0) and the MedSAM2 checkpoint is already local
  (`data/models/medsam2/MedSAM2_latest.pt`, gitignored).
- **VisionServeX** (local, v3.23.0, importable) is inspected and **used if it already serves
  MedSAM2/SAM2** — no duplicate serving code if VisionServeX solves it.
- **SAM2.1 / SAM3 are baselines/comparators only**, never substitutes for MedSAM2.
- Do **not** stop at "sam2 missing" — install/vendor/wrap it or document a genuine hard blocker with
  exact logs.

## Plan

P1 audit VisionServeX + MedSAM2 capability → P2 install real MedSAM2 (`sam2` + checkpoint) → P3 smoke
test proving real MedSAM2 ran → P4 spine prompt tuning (train/dev only) → P5 segmentation cache →
P6 MedSAM2 vs SAM2.1 vs SAM3 QC comparison → P7 MedSAM2 morphometry → P8 morphometry-only signal
(**go/no-go**: if not complementary to v1.8b, don't waste locked-test) → P9 fusion → P10 triage →
P11 VisionServeX integration → P12 locked-test once → P13 decision → P14 gates/merge.

## Locked-test policy

All prompt tuning, QC, morphometry selection, and fusion/router selection use **train/dev only**.
Locked-test is read **once** for the final dev-frozen candidate; every read counted.

## Artifact / secret safety

Commit **only** code, configs, docs, summary metrics, safe metadata. Never commit token files,
`kaggle.json`, model weights, the official MedSAM2/SAM2 repo, DICOM/NIfTI, panels, masks,
segmentation caches, morphometry/feature/logit parquet, checkpoints, runs, or outputs. Generated
artifacts live under gitignored `data/`, `external/`, `outputs/`, `runs/`.

## Success thresholds (≥1, else executed negative proving real MedSAM2 is redundant/bounded)

1. R-for severe recall **+≥0.04**; 2. foraminal macro **+≥0.03**; 3. subarticular macro **+≥0.03**;
4. five-route macro **+≥0.02**; 5. high-conf severe-FN **−≥20%** without raw loss; 6. morphometry
triage beats v1.7/v1.8b severe-FN capture at equal budget; 7. else honest executed negative
(**real MedSAM2 ran**, not SAM2.1 fallback).

## Tagging

`v1.8c.0-real-medsam2-accuracy-upgrade` (raw severe recall up) · `…-real-medsam2-triage-upgrade`
(triage up, raw flat) · `…-visionservex-medsam2-integration` (operational integration the deliverable)
· `…-real-medsam2-negative-result` (real MedSAM2 executed, nothing improves). No accuracy-upgrade tag
without a locked-test raw gain.
