# v1.9 — real-image gallery release gate

> Research-only · not diagnostic. RSNA dataset: CC BY-NC-SA 4.0 for non-commercial research.

**Decision: PASS — panels may be committed**

## Gate conditions

| Condition | Status |
|---|---|
| repo is private | ✅ PASS |
| no dicom or nifti in docs | ✅ PASS |
| no oversized panels | ✅ PASS |
| panels are derived not raw series | ✅ PASS |
| no phi in filenames | ✅ PASS |
| non commercial licence noted | ✅ PASS |

## RSNA licence note

RSNA 2024 Lumbar Spine Degenerative Classification dataset is licensed **CC BY-NC-SA 4.0 for non-commercial research** (RSNA terms). Derived visual panels (not raw DICOMs; anonymized case IDs; small crops) may be included in a private non-commercial repository under this licence, provided no full-resolution images or patient identifiers are committed.

## Regenerate gallery locally

```bash
python scripts/generate_readme_assets_v1_9.py  # runs on local RSNA data
```

The generated panels live under gitignored `local_reports/v1_9_real_case_gallery/`.
