# Data setup

> **Research-only.** SpineScoutX is not diagnostic and not for medical
> decision-making. No dataset files are distributed with this repository. You must
> obtain the datasets yourself under their respective licenses.

SpineScoutX never downloads data automatically. All commands take explicit
`--root`/`--out` paths and fail with a precise message listing what is missing.

## Datasets

| Dataset | Role | License / use | Acquire from |
|---|---|---|---|
| RSNA 2024 Lumbar Spine Degenerative Classification | Disc-level finding **grading** (primary) | Non-commercial research use only | Kaggle competition page |
| SPIDER Lumbar Spine Segmentation | **Anatomy** segmentation priors (vertebra/disc/canal) | CC BY 4.0 (attribution required) | Zenodo |

Neither dataset is committed, cached publicly, or redistributed here.

## Expected layout

### RSNA (`--rsna-root`)
```
<rsna-root>/
  train.csv                          # wide grades: <condition>_<level> -> Normal/Mild|Moderate|Severe
  train_label_coordinates.csv        # study_id, series_id, instance_number, condition, level, x, y
  train_series_descriptions.csv      # study_id, series_id, series_description
  train_images/<study_id>/<series_id>/<instance>.dcm
```
Build the cache:
```
spinescoutx prepare-rsna --rsna-root data/raw/rsna --out data/cache/rsna
```

### SPIDER (`--spider-root`)
```
<spider-root>/
  images/<subject>.mha      (or .nii.gz)
  masks/<subject>.mha       (or .nii.gz)
```
Build the cache:
```
spinescoutx prepare-spider --spider-root data/raw/spider --out data/cache/spider
```

## Optional readers (lazy)

The core install does **not** require imaging readers. Install only what you need:

- `pip install spinescoutx[dicom]` — `pydicom`, to decode RSNA DICOMs.
- `pip install SimpleITK` **or** `pip install nibabel` — to read SPIDER `.mha`/`.nii.gz` volumes.
- `pip install spinescoutx[seg]` — optional MONAI segmentation backbone.
- `pip install spinescoutx[parquet]` — Parquet crop manifests (CSV fallback otherwise).

`spinescoutx doctor` reports which of these are present.

## Anatomy mask remapping (SPIDER → 4 classes)

SPIDER ships per-vertebra and per-disc label ids plus a spinal-canal label. We
collapse them into 4 semantic classes — `background / vertebra / disc /
spinal_canal` — via `spinescoutx.data.spider_index.remap_spider_labels`. The
mapping convention is documented in that function and is **approximate**; SPIDER
does **not** label neural foramina or lateral recesses, so evidence regions for
foraminal/subarticular findings are flagged `evidence_region_source="approximate"`.

## Splits & leakage

Splits are **patient/study-level only** (`spinescoutx.data.splits`), never
image-level. `check_no_leakage` raises if any study appears in more than one
split. Split files record the seed, a caller-supplied timestamp, and per-split
counts.

## No data required for tests

The full test suite and synthetic smoke runs use in-memory synthetic fixtures
(`spinescoutx.data.synthetic`). You can develop and run CI without either dataset.
