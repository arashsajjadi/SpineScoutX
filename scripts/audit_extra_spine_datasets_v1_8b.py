#!/usr/bin/env python3
"""Audit extra spine datasets for usefulness (v1.8b Phase 3). Reads only; emits a JSON summary."""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
AXON = ROOT / "data/external/axondata"
OUT = ROOT / "outputs/real/v1_8b_extra_dataset_audit.json"


def main() -> int:
    import SimpleITK as sitk

    sets = sorted({p.parent.parent.name for p in AXON.rglob("*") if p.is_file()})
    dicoms = [p for p in AXON.rglob("*") if p.is_file() and p.suffix == "" and p.name.isdigit()]
    nrrd = glob.glob(str(AXON / "**/*.nrrd"), recursive=True)
    js = glob.glob(str(AXON / "**/*.json"), recursive=True)
    labels = Counter()
    for f in nrrd[:20]:
        import numpy as np

        labels.update(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(f))).tolist())
    audit = {
        "axondata": {
            "patient_sets": sets,
            "n_patient_sets": len([s for s in sets if s.startswith("Set")]),
            "modality": "axial T2 frFSE (DICOM)",
            "n_dicom": len(dicoms),
            "n_nrrd_masks": len(nrrd),
            "n_json_markups": len(js),
            "mask_labels": {int(k): int(v) for k, v in labels.items()},
            "markup_format": "3D Slicer markups (mm measurements)",
            "rsna_compat": "LOW — 3-patient sample; axial T2 (RSNA foraminal is sagittal T1); "
            "too few for training or calibration",
            "useful_for": "mm-measurement schema reference only (what foraminal morphometry "
            "measures); NOT for training/grading",
            "decision": "audited -> rejected for direct use (sample-only); kept gitignored",
        }
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
