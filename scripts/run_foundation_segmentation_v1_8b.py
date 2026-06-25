#!/usr/bin/env python3
"""Run SAM2.1 foundation segmentation + morphometry over RSNA foraminal crops (v1.8b Phase 4/8).

For every foraminal finding (train+dev+test): SAM2.1 box-prompt segmentation of the central
foraminal-opening proxy → morphometric features. Saves a **gitignored** features parquet (no masks,
no pixels) + an area-distribution QC summary. Masks are unvalidated foundation proxies; QC is
downstream. No locked-test labels are used here (features only). Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_load_tokens_v1_8b import ensure_auth  # noqa: E402

from spinescoutx.data.locked_test import load_splits_v1  # noqa: E402
from spinescoutx.features.morphometry import FEATURE_COLS, foraminal_features  # noqa: E402
from spinescoutx.segmentation.foundation_runner import (  # noqa: E402
    load_sam2,
    segment_center_channel,
)

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
OUTDIR = ROOT / "data/cache/v1_8b_morphometry"
QC = ROOT / "outputs/real/v1_8b_morphometry_qc.json"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/sam2.1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ensure_auth()
    model, processor, device = load_sam2(ROOT / args.model)
    m = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
    m = m[m.condition.isin(FORAMINAL) & m.severity_index.isin([0, 1, 2])].copy()
    m["study_id"] = m.study_id.astype(str)
    m["key"] = m.study_id + "|" + m.level.astype(str) + "|" + m.condition
    m["split"] = m.study_id.map(load_splits_v1(SPLITS))
    if args.limit:
        m = m.head(args.limit)
    rows, t0 = [], time.time()
    keys = m.to_dict("records")
    for i in range(0, len(keys), args.batch):
        chunk = keys[i : i + args.batch]
        crops = [np.load(RSNA_CACHE / r["crop_path"]).astype(np.float32) for r in chunk]
        segs = segment_center_channel(model, processor, device, crops)
        for r, crop, seg in zip(chunk, crops, segs, strict=False):
            feats = foraminal_features(seg["mask"], crop, seg["iou"])
            rows.append({
                "key": r["key"], "study_id": r["study_id"], "level": r["level"],
                "condition": r["condition"], "side": r["side"], "split": r["split"],
                "severity_index": int(r["severity_index"]), **feats,
            })  # fmt: skip
        if (i // args.batch) % 50 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(keys)} ({rate:.1f}/s)", flush=True)
    df = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / "features.parquet", index=False)

    sev = df[df.severity_index == 2]
    nonsev = df[df.severity_index != 2]
    qc = {
        "model": args.model,
        "n_findings": int(len(df)),
        "runtime_s": round(time.time() - t0, 1),
        "rate_per_s": round(len(df) / max(time.time() - t0, 1e-6), 2),
        "seg_fail_rate": round(float(df.m_seg_fail.mean()), 4),
        "mean_iou_conf": round(float(df.m_iou_conf.mean()), 3),
        "area_frac": {
            "all_mean": round(float(df.m_area_frac.mean()), 4),
            "severe_mean": round(float(sev.m_area_frac.mean()), 4),
            "nonsevere_mean": round(float(nonsev.m_area_frac.mean()), 4),
        },
        "min_open": {
            "severe_mean": round(float(sev.m_min_open.mean()), 4),
            "nonsevere_mean": round(float(nonsev.m_min_open.mean()), 4),
        },
        "feature_cols": FEATURE_COLS,
    }
    QC.parent.mkdir(parents=True, exist_ok=True)
    QC.write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2))
    print(f"\nwrote {OUTDIR / 'features.parquet'} ({len(df)} findings); {QC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
