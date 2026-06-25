# Data handling and privacy — SpineScoutX

> Research-only. Non-commercial. Not diagnostic. No patient data committed to this repository.

## Datasets used

### RSNA 2024 Lumbar Spine Degenerative Classification

- **Source:** Kaggle competition hosted by RSNA.
- **Licence:** CC BY-NC-SA 4.0 — **non-commercial research only**.
- **Access:** Kaggle API (`kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification`). Requires accepting the Kaggle competition terms.
- **Status:** NOT redistributed. Raw DICOMs and labels are gitignored (`/data/`).
- **Content used:** sagittal T1/T2 and axial T2 MRI series; grading labels for 5 degenerative findings (normal/mild/moderate/severe). Series metadata (instance numbers, series descriptions) used for routing.
- **Studies:** 1975 unique patients → splits_v1: train 1382 / dev 296 / test 296 (patient-level, no leakage).

### SPIDER — Lumbar Spine Segmentation Dataset

- **Source:** Zenodo record 10159290 (`https://zenodo.org/record/10159290`).
- **Licence:** CC BY 4.0 — redistribution permitted with attribution.
- **Status:** NOT redistributed here (≈GB of NIfTI); downloaded locally to `data/raw/spider/` (gitignored).
- **Content used:** segmentation masks (vertebra, intervertebral disc, spinal canal) for anatomy-prior generation. Not used for grading labels.
- **Official splits:** used as-is (SPIDER's train/val/test).

## What IS committed to this repository

- Source code, configs, pyproject.toml, tests.
- Safe metric metadata: JSON/markdown summaries of aggregate metrics (recall, AUROC, CIs) — no patient-level data.
- Documentation and run logs.
- Anonymized derived example panels (`docs/assets/v1_9/real_cases/`): small PNG panels showing a single central slice crop + model predictions. Case IDs are **SHA-1 hashes** of `study_id|level|condition` — not reversible to patient identity. No EXIF or DICOM metadata. No full series. No patient names, dates, or identifiers.
- Chart and gallery index files.

## What is NOT committed

| Category | Examples | Why |
|---|---|---|
| Raw imaging data | DICOMs, NIfTI, PNG image dumps | Data licence (CC BY-NC-SA); privacy |
| Processed caches | Crops (`.npy`), anatomy priors, segmentation outputs | Large, reproducible from code; gitignored under `/data/` |
| Model weights | `best.pt`, `last.pt` checkpoints | Large (> 50 MiB each); published via GitHub Release asset only |
| Features / logits | Parquet files of morphometry features, probability logits | Derived; gitignored under `/outputs/` |
| API tokens | HF token, Kaggle token | Never committed; loaded from local files outside repo |
| Review packs | `review_packs/` (704-case radiology review images) | Full-resolution local-only per RSNA licence |
| Segmentation caches | SAM2.1/MedSAM2 mask caches | Derived; gitignored |

## Panel anonymization details

Each panel in `docs/assets/v1_9/real_cases/` uses a case ID of the form `case_XXXXXXXX` where
`XXXXXXXX` is the first 8 hex characters of `SHA-1(study_id|level|condition)`. This hash:

- Is deterministic (same case always gets the same ID)
- Cannot be reversed to the original RSNA study ID without a brute-force lookup table
- Contains no patient name, date of birth, scan date, or other PHI

The panels show only the model's central slice crop (not the full MRI series), probability
estimates, and labels. No DICOM metadata is embedded in the PNG files.

## Compliance summary

- **Non-commercial use only** (both datasets; this repository is a non-commercial research project).
- **No redistribution** of raw imaging data or labels.
- **No PHI** committed (anonymized hash IDs only; no names, dates, or identifiers).
- **Private repository** (GitHub private; no public redistribution of RSNA data).
- Panels committed under the CC BY-NC-SA 4.0 derived-work exception for non-commercial research in a private repository.
