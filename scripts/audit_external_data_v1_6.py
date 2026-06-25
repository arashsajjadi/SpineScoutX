#!/usr/bin/env python3
"""Audit the LSS external data + its RSNA compatibility (v1.6, Plan A).

Reports the LSS foraminal label distribution (from the crop cache) and the LSS-vs-RSNA crop
compatibility (shape, intensity) used to choose the transfer mode. Reads only cached crops; emits
a JSON summary (gitignored) and prints a table. Research-only. Not diagnostic.
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
LSS = ROOT / "data/cache/lss_foraminal"
RSNA = ROOT / "data/cache/rsna_auto_foraminal"
OUT = ROOT / "outputs/real/external_data_audit_v1_6.json"


def _intensity(paths, n=400):
    arr = np.stack([np.load(p).astype(np.float32) for p in paths[:n]])
    return {
        "shape": list(arr.shape[1:]),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


def main() -> int:
    man = pd.read_parquet(LSS / "manifest.parquet")
    lss_paths = sorted(glob.glob(str(LSS / "crops/*.npy")))
    rsna_paths = sorted(glob.glob(str(RSNA / "crops/*.npy")))
    out = {
        "lss_labels": {
            "n_boxes": int(len(man)),
            "n_patients": int(man.patient.nunique()),
            "by_side": dict(Counter(man.side)),
            "by_severity_index": {
                int(k): int(v) for k, v in sorted(Counter(man.severity_index).items())
            },
            "severe_by_side": dict(Counter(man[man.severity_index == 2].side)),
            "by_level": dict(sorted(Counter(man.level).items())),
            "split_severe": {
                sp: int((man[man.split == sp].severity_index == 2).sum())
                for sp in sorted(man.split.unique())
            },
        },
        "compatibility": {
            "lss_crop_intensity": _intensity(lss_paths),
            "rsna_foraminal_crop_intensity": _intensity(rsna_paths),
            "same_shape": _intensity(lss_paths)["shape"]
            == _intensity(rsna_paths)["shape"]
            == [3, 224, 224],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
