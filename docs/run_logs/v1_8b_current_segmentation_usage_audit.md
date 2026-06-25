# v1.8b — current foundation-segmentation usage audit (Phase 1)

> Research-only · not diagnostic. Reproduce: `audit_segmentation_foundation_usage_v1_8b.py`.

## Are MedSAM2 / SAM2 / SAM3.1 / SPINEPS / nnU-Net used? — NO

A repo-wide search (`medsam|sam2|sam3|segment.?anything|spineps|nnunet`) finds **zero** references
in `src/` or `scripts/`. **No segmentation foundation model has ever been used** in SpineScoutX —
this is the genuinely new lever for v1.8b.

## What segmentation / morphometry infra already exists

| component | what it is | status |
|---|---|---|
| `models/anatomy_segmenter.py` + `runs/e4_segmentation_spider_real` | U-Net trained on **SPIDER** (canal/disc/vertebra) | trained, Dice ≈0.884 (v0.x) — **reusable spine-specific seg (Plan C)** |
| `training/train_segmenter.py` | SPIDER segmenter trainer | available |
| `evaluation/segmentation_metrics.py` | Dice / IoU | available |
| `features/morphology.py` | 14-feature morphometry engine (**canal-centric**: canal_disc_ratio, etc.) | used in v0.x E3 (canal); showed signal (canal_disc_ratio 5.9 normal→2.3 severe) but E3 did **not** beat E2 |
| `data/anatomy_priors.py`, `models/anatomy_guided_classifier.py`, `anatomy_forced_classifier.py` | v0.4/v0.5 anatomy-prior grading | executed; no grading uplift (v1.6 Plan C) |

## Implications for v1.8b

- Foundation segmentation (**MedSAM2 / SAM3.1 / SAM2.1**) is **new** → Plans A/B.
- The existing **SPIDER U-Net (e4)** is a reliable spine-specific fallback (Plan C) and the QC
  reference for the foundation models on canal/disc/vertebra.
- Existing morphometry is **canal-centric**; **foraminal opening geometry** + **subarticular /
  lateral-recess** morphometry, and **fusion for the weak routes**, are new (Plan D).
- No deployed grader uses any segmentation/morphometry signal today.
