# Kaggle Competition Data Availability — SpineScoutX v1.9

> Research-only · non-commercial · not diagnostic.

## Status: LOCALLY AVAILABLE (previously downloaded)

Competition: `rsna-2024-lumbar-spine-degenerative-classification`

## Files available locally at `data/raw/rsna/`

| File | Status |
|---|---|
| `sample_submission.csv` | ✅ present (2545 bytes) |
| `test_series_descriptions.csv` | ✅ present (136 bytes) |
| `test_images/` | ✅ present |
| `train.csv` | ✅ present (training only) |
| `train_images/` | ✅ present (training only) |

## Test set structure

- **1 test study** visible: `44036939`
- 3 MRI series:
  - `2828203845` — Sagittal T1 (→ foraminal inference)
  - `3481971518` — Axial T2 (→ subarticular inference)
  - `3844393089` — Sagittal T2/STIR (→ canal inference)
- **25 prediction rows** (5 conditions × 5 levels)

## Kaggle private test set

The Kaggle private test set is NOT downloadable. It is evaluated server-side.
Our submission covers all 25 visible rows. Kaggle will score this against the
private test portion as well.

Note: This is a late submission (competition deadline: 2024-10-08).
Kaggle accepts late CSV submissions and provides public/private scores.

## Safety checks

- No competition DICOM data committed to git (`/data/` is gitignored).
- No test images committed.
- `sample_submission.csv` referenced only in code; not committed.
- Kaggle data directory: `data/raw/rsna/` (gitignored via `/data/`).
