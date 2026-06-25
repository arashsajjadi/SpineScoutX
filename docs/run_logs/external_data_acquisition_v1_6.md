# v1.6 external data acquisition (Plan A)

> Research-only · non-commercial · not diagnostic. External data is **gitignored** and never
> committed; only derived numbers + tooling live in the repo.

## Acquired: LSS-MRI AISSLab Dataset

| field | value |
|---|---|
| title | LSS MRI AISSLab Dataset (sagittal lumbar-spine MRI, foraminal stenosis) |
| DOI | 10.17632/rgb77xm3jf.4 (Mendeley Data) |
| GitHub | https://github.com/AISSLab2025/LSS-MRI-AISSLab-Dataset (code/figures only) |
| paper | Nature Scientific Data s41597-026-07138-x |
| license | **CC BY 4.0** (non-commercial research use; attribution) |
| archive | `LSS MRI AISSLab Dataset_V0.2.zip`, 2,035,587,816 bytes |
| sha256 | `592a294f93d575a16bccc2681c793eb1cfc6679fa2746ac50cbc8970f806b4b1` (**verified**) |
| download | Mendeley public-files API URL (no login) — see `scripts/prepare_lss_mri_aisslab.py` |

**Status: acquired, checksum-verified, extracted.** Network was available; the public Mendeley
file URL downloaded without authentication. (Manual fallback documented in
`lss_manual_download_instructions.md` for environments without network.)

## Archive structure

- `DICOM/<patient>/IM*.dcm` — raw sagittal series (6680 slices).
- `Foramina_Detection/<patient>/IM*.png` + `IM*.xml` — rendered 512×512 grayscale slices +
  **PASCAL-VOC** foraminal-stenosis boxes (1474 XML carrying 2979 boxes).
- `Segmentation/Masks/*.xml` — 6-class anatomical-mask annotations (500 XML; not used by Plan A).

## Annotation schema (foraminal)

Object `<name>` = `{RFS|LFS}{grade}` (Right/Left Foraminal Stenosis + grade 0=Normal, 1=Mild,
2=Moderate, 3=Severe); `<level>` = L1-L2..L5-S1; `<bndbox>` in PNG pixels. Mapped to the RSNA
3-class scheme: **Normal/Mild→0, Moderate→1, Severe→2** (`grade_to_rsna_severity_index`).

## What was used

Only `Foramina_Detection/` (PNGs + XML) → 2.5D foraminal crops (3,224,224) compatible with the
RSNA sagittal-T1 foraminal grader. Tooling: `scripts/prepare_lss_mri_aisslab.py` (download/verify/
extract), `src/spinescoutx/data/lss_aisslab.py` (adapter), `scripts/prepare_lss_foraminal_v1_6.py`
(crop cache), `scripts/audit_external_data_v1_6.py` (audit). Config:
`configs/data/external_datasets_v1_6.yaml`.
