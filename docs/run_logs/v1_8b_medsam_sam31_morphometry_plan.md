# v1.8b — MedSAM2 / SAM3.1 Segmentation-Morphometry Accuracy Offensive: plan

> **Research-only · non-commercial · not diagnostic · not clinically validated · not for medical
> decision-making.** Held-out reference = RSNA graded severity. Protocol: `splits_v1`
> (train 1382 / dev 296 / test 296). **Dev/train select; RSNA locked-test read once** per final
> selected candidate. Never tune on locked-test; never use test labels for training/selection.

## Hypothesis (the new lever)

v1.4–v1.7 proved the weak-route severe ceiling is not moved by classifier-only tricks, external
classification data, SSL, anatomy-prior concat, bigger backbones, MIL, distillation, or algorithmic
label cleaning. **Untested:** explicit **segmentation-derived anatomy + morphometry** as new
evidence for the grader — foraminal opening geometry, canal area / AP diameter, disc height /
alignment, lateral-recess / subarticular narrowing proxies, and segmentation-quality-aware route
confidence. This sprint turns strong segmentation **foundation models** into a real
segmentation→morphometry→fusion pipeline and measures whether that signal raises raw severe recall.

## Plan ladder

- **Plan A — MedSAM2** (medical SAM2): box/point prompts + slice-sequence propagation → canal /
  disc / foramen / lateral-recess masks on RSNA.
- **Plan B — SAM 3.1** (open-vocabulary; text + visual prompts) as comparator / fallback;
  **SAM 2.1** if SAM 3.1 access is blocked.
- **Plan C — spine-specific fallback** (SPINEPS / nnU-Net / light U-Net on SPIDER + LSS masks) if
  the foundation models fail QC.
- **Plan D — morphometric fusion grading**: extract morphometry from the QC-selected masks →
  morphometry-only signal check → image+morphometry fusion grader → morphometry-informed triage.

## Credentials (local, never exposed)

Hugging Face + Kaggle tokens are read **locally only** from the user's `~/Documents/api_*.txt` by
`scripts/private_load_tokens_v1_8b.py`; secret **values are never printed, logged, committed, or
pasted into docs**. Any generated `kaggle.json` (chmod 600) and all downloaded weights/data/features
live under **gitignored** folders (`data/models/`, `data/cache/v1_8b_*`, `data/external/`).

## Forbidden-artifact policy

Commit **only** code, configs, docs, summary metrics, and pixel-free / synthetic examples. Never
commit token files, model weights, DICOM/NIfTI, PNG/JPG panels, masks, segmentations, crops,
morphometry/feature/logit parquet dumps, caches, checkpoints, runs, outputs, or large files.

## Locked-test policy

All segmentation QC, morphometry selection, and grader/router selection use **train/dev only**. RSNA
**locked-test is read once** for the final dev-selected candidates; every read is counted.

## Success thresholds (≥1, else rigorous executed negative)

1. R-for severe recall **+≥0.04**; 2. foraminal macro **+≥0.03**; 3. subarticular macro **+≥0.03**;
4. five-route macro **+≥0.02**; 5. high-confidence severe-FN **−≥20%** without raw-recall loss;
6. recall@FAR≤10% materially up for R-for **and** another weak route; 7. else prove whether
segmentation/morphometry **is or is not** the missing signal (executed negative).

## Tagging

`v1.8b.0-medsam2-morphometry-accuracy-upgrade` / `…-sam31-…` / `…-spine-specific-morphometry-upgrade`
(raw severe recall improves) · `…-morphometry-triage-upgrade` (raw flat, triage improves) ·
`…-morphometry-negative-result` (all paths executed, nothing improves). **No accuracy-upgrade tag
without a real locked-test raw metric improvement.**
