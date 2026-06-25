#!/usr/bin/env python3
"""Build the LSS-MRI AISSLab foraminal crop cache + manifest (v1.6, Plan A).

Parses every PASCAL-VOC box under ``Foramina_Detection/``, extracts an RSNA-compatible 2.5D
foraminal crop (3,224,224) per box, and writes a patient-level train/dev split manifest. Crops are
float16 and gitignored (`data/cache/lss_foraminal/`). External LSS data is never committed.

Research-only. Not diagnostic. Reproduce: `python scripts/prepare_lss_foraminal_v1_6.py`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.lss_aisslab import (
    LSS_CONDITION,
    RSNA_SEVERITY_NAME,
    crop_lss_25d,
    iter_lss_boxes,
)

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
DETECTION = ROOT / "data/external/lss_mri_aisslab/extracted/Foramina_Detection"
OUT = ROOT / "data/cache/lss_foraminal"
CROP_SIZE = 224
DEV_FRAC = 0.15
SEED = 1337


def patient_split(patients: list[str]) -> dict[str, str]:
    """Deterministic patient-level lss_train/lss_dev split (no patient leakage)."""
    rng = np.random.default_rng(SEED)
    pats = sorted(set(patients))
    rng.shuffle(pats)
    n_dev = max(1, int(round(len(pats) * DEV_FRAC)))
    dev = set(pats[:n_dev])
    return {p: ("lss_dev" if p in dev else "lss_train") for p in pats}


def main() -> int:
    if not DETECTION.exists():
        raise SystemExit(f"missing {DETECTION}; run prepare_lss_mri_aisslab.py first")
    boxes = list(iter_lss_boxes(DETECTION))
    split_map = patient_split([b.patient for b in boxes])
    cropdir = OUT / "crops"
    cropdir.mkdir(parents=True, exist_ok=True)
    records, skipped = [], 0
    for i, b in enumerate(boxes):
        crop = crop_lss_25d(b, crop_size=CROP_SIZE)
        if crop is None:
            skipped += 1
            continue
        rel = f"crops/{b.patient}_{b.slice_name}_{b.level}_{b.side}_{i}.npy"
        np.save(OUT / rel, crop.astype(np.float16))
        records.append(
            {
                "patient": b.patient,
                "slice_name": b.slice_name,
                "side": b.side,
                "level": b.level,
                "condition": LSS_CONDITION[b.side],
                "condition_idx": CONDITION_TO_INDEX[LSS_CONDITION[b.side]],
                "level_idx": LEVEL_TO_INDEX[b.level],
                "grade_lss": b.grade,
                "severity_index": b.severity_index,
                "severity": RSNA_SEVERITY_NAME[b.severity_index],
                "bbox": json.dumps(list(b.bbox)),
                "crop_path": rel,
                "split": split_map[b.patient],
            }
        )
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(boxes)} boxes cropped", flush=True)
    df = pd.DataFrame(records)
    df.to_parquet(OUT / "manifest.parquet", index=False)

    audit = {
        "source": "LSS-MRI AISSLab (Mendeley rgb77xm3jf v4, CC BY 4.0, non-commercial research)",
        "n_boxes_parsed": len(boxes),
        "n_crops": int(len(df)),
        "n_skipped": skipped,
        "n_patients": int(df.patient.nunique()),
        "by_side": dict(Counter(df.side)),
        "by_severity_index": {
            int(k): int(v) for k, v in sorted(Counter(df.severity_index).items())
        },
        "severe_by_side": dict(Counter(df[df.severity_index == 2].side)),
        "by_level": dict(sorted(Counter(df.level).items())),
        "split_counts": dict(Counter(df.split)),
        "severe_by_split": {
            sp: int((df[df.split == sp].severity_index == 2).sum()) for sp in df.split.unique()
        },
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"\nwrote {OUT / 'manifest.parquet'} ({len(df)} crops), {OUT / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
