# v1.8b — extra spine dataset audit (Phase 3)

> Research-only · not diagnostic. Downloaded with the user's local Kaggle token (never printed) into
> **gitignored** `data/external/`. Reproduce: `download_extra_spine_datasets_v1_8b.py` then
> `audit_extra_spine_datasets_v1_8b.py`.

## AxonData — Foraminal Stenosis MRI Dataset (`axondata/foraminal-stenosis-mri-dataset`)

| field | value |
|---|---|
| size | 11.6 MB (a **sample**) |
| patients | **3** (Set_1 / Set_2 / Set_3) |
| modality | **axial T2 frFSE** (DICOM) |
| masks | 20 NRRD, 4-class (labels 0–3) segmentations |
| markups | 38 3D-Slicer JSON markups with **mm measurements** |
| license | not surfaced via the Kaggle API (usability 0.94); treated as research-sample, **not redistributed**, kept gitignored |

**RSNA compatibility: LOW.** Only **3 patients** (a teaser sample), and **axial T2** — whereas RSNA
foraminal grading is on **sagittal T1**. Too few cases for training or even morphometric calibration.

**Decision: audited → rejected for direct use.** The one useful takeaway is the **mm-measurement
schema** (it confirms which foraminal-opening geometry to measure — vertical/horizontal opening,
area), which informs the v1.8b morphometry feature design. Nothing committed.

## Other datasets

SPIDER (lumbar vertebra/disc/canal masks) is already local and is the segmentation-QC reference /
spine-specific fallback (Plan C). LSS-MRI AISSLab (foraminal boxes/grades) is already local from
v1.6. No additional license-safe lumbar segmentation dataset was adopted this run.
