# Data status & acquisition blocker

> **Research-only.** No dataset files are committed or redistributed. RSNA is
> non-commercial research use; SPIDER is CC BY 4.0 (attribution).

**Status as of 2026-06-23 (branch `feature/spinescoutx-real-data-experiments`):**
the real-data experiments are **BLOCKED**. Neither dataset is present, and RSNA
cannot be obtained automatically because Kaggle credentials are missing. No real
metrics have been produced; no synthetic output is presented as real.

Check readiness any time with:
```
spinescoutx doctor --data --rsna-root data/raw/rsna --spider-root data/raw/spider
```

## RSNA 2024 Lumbar Spine Degenerative Classification — BLOCKED (credentials)

| Item | Value |
|---|---|
| Expected root | `data/raw/rsna` |
| Present? | **No** |
| Missing files | `train.csv`, `train_label_coordinates.csv`, `train_series_descriptions.csv`, `train_images/` |
| Kaggle CLI installed? | **No** |
| `~/.kaggle/kaggle.json`? | **No** |
| `KAGGLE_USERNAME` env? | **No** |
| Command attempted | `spinescoutx prepare-rsna --rsna-root data/raw/rsna --out data/cache/rsna` → fails with explicit "missing files" report |
| License | non-commercial research use only (RSNA/Kaggle competition terms) |

**Exact user action required (only you can do this):**
1. Create/sign in to a Kaggle account.
2. On the competition page, **accept the competition rules** (data is gated until you do).
3. Create an API token: Kaggle → *Account* → *Create New API Token* → download `kaggle.json`.
4. Install it: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
5. `pip install kaggle`
6. Download + unzip:
   ```
   kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification -p data/raw/rsna
   cd data/raw/rsna && unzip -q '*.zip'
   ```
7. `pip install "spinescoutx[dicom]"` (pydicom is needed to decode DICOMs).
8. Re-run `spinescoutx doctor --data` and then `spinescoutx prepare-rsna ...`.

> The maintainer/agent must **not** download RSNA from unofficial mirrors and must
> not commit any DICOMs, CSV labels, crops, or caches.

## SPIDER Lumbar Spine Segmentation — DOWNLOADED; E4 trained ✅

Real E4 anatomy segmentation has been **run** on this dataset: mean Dice **0.884**
(canal 0.902, vertebra 0.903, disc 0.846) on SPIDER's **official** validation split
(218 patients / 447 volumes / 10,338 cached slices). See `technical_report.md`
§9.1. Raw data lives under `data/raw/spider` (gitignored, not redistributed).

| Item | Value |
|---|---|
| Expected root | `data/raw/spider` |
| Present? | **Yes** (downloaded from the official Zenodo record) |
| Official source | Zenodo record **10159290**, DOI `10.5281/zenodo.10159290` (verified via Zenodo API) |
| Title | *SPIDER — Lumbar spine segmentation in MR images: a dataset and a public benchmark* (van der Graaf et al.) |
| License | **CC BY 4.0** (attribution required) |
| Files | `images.zip` (~3.70 GB), `masks.zip` (~58 MB), `radiological_gradings.csv`, `overview.csv` (~3.76 GB total) |
| Auth needed? | No (public) |

**Official acquisition (no credentials needed):**
```
mkdir -p data/raw/spider && cd data/raw/spider
curl -L -O "https://zenodo.org/records/10159290/files/images.zip?download=1"
curl -L -O "https://zenodo.org/records/10159290/files/masks.zip?download=1"
curl -L -O "https://zenodo.org/records/10159290/files/overview.csv?download=1"
unzip -q images.zip && unzip -q masks.zip
pip install SimpleITK     # or nibabel, to read the .mha volumes
```
**Scope note:** SPIDER alone enables only **E4** (anatomy segmentation) and the
SPIDER-side AEC overlays. The headline anatomy-prior experiments (**E0/E1/E2/E3**,
RSNA AEC) require RSNA, so they remain blocked until RSNA is available. The SPIDER
download (~3.76 GB) is **not** started automatically; it is a heavy operation that
should be explicitly requested.

## What runs without data (regression-safe)

The full test suite and synthetic smoke run with **no** datasets
(`spinescoutx.data.synthetic`); see `technical_report.md` §0/§9 — those numbers are
**synthetic smoke only**, never real results, and synthetic figures are watermarked
`SYNTHETIC SMOKE — not a real RSNA/SPIDER result`.

## No redistribution

Do not commit or redistribute RSNA or SPIDER data, masks, DICOMs, NIfTI/MHA
volumes, crops, caches, or model weights. `.gitignore` enforces this; the
`tests/test_no_data_committed.py` guard fails the build if any slips through.
