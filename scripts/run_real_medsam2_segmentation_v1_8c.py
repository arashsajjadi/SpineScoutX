#!/usr/bin/env python3
"""Real MedSAM2 segmentation + morphometry over RSNA foraminal crops (v1.8c Phase 5/7).

Segments every foraminal finding (train+dev+test) with REAL MedSAM2 (via VisionServeX runtime) and
extracts the shared morphometry features. Gitignored features parquet (no masks/pixels). Locked-test
labels are never used here. Research-only. Not diagnostic.
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
from spinescoutx.features.medsam2_morphometry import FEATURE_COLS, foraminal_features  # noqa: E402
from spinescoutx.segmentation.medsam2_runner import MedSAM2, available  # noqa: E402

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
OUTDIR = ROOT / "data/cache/v1_8c_medsam2_morphometry"
QC = ROOT / "outputs/real/v1_8c_real_medsam2_segmentation_qc.json"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ensure_auth()
    if not available():
        raise SystemExit("real MedSAM2 unavailable — refusing SAM2.1 fallback (v1.8c hard rule)")
    m = MedSAM2()
    info = m.info()
    assert type(m.runtime._model).__module__.startswith("sam2."), "not real MedSAM2"
    man = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
    man = man[man.condition.isin(FORAMINAL) & man.severity_index.isin([0, 1, 2])].copy()
    man["study_id"] = man.study_id.astype(str)
    man["key"] = man.study_id + "|" + man.level.astype(str) + "|" + man.condition
    man["split"] = man.study_id.map(load_splits_v1(SPLITS))
    if args.limit:
        man = man.head(args.limit)
    rows, t0 = [], time.perf_counter()
    recs = man.to_dict("records")
    for i, r in enumerate(recs):
        crop = np.load(RSNA_CACHE / r["crop_path"]).astype(np.float32)
        mask, score = m.segment(crop)
        feats = foraminal_features(mask, crop, score)
        rows.append(
            {
                "key": r["key"],
                "study_id": r["study_id"],
                "level": r["level"],
                "condition": r["condition"],
                "side": r["side"],
                "split": r["split"],
                "severity_index": int(r["severity_index"]),
                **feats,
            }
        )
        if i % 2000 == 0:
            rate = (i + 1) / max(time.perf_counter() - t0, 1e-6)
            print(f"  {i + 1}/{len(recs)} ({rate:.1f}/s)", flush=True)
    df = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / "features.parquet", index=False)
    sev, non = df[df.severity_index == 2], df[df.severity_index != 2]
    qc = {
        "model": "real_MedSAM2",
        "model_module": type(m.runtime._model).__module__,
        "checkpoint": Path(info["checkpoint_path"]).name,
        "config": info["config_path"],
        "n_findings": int(len(df)),
        "runtime_s": round(time.perf_counter() - t0, 1),
        "seg_fail_rate": round(float(df.m_seg_fail.mean()), 4),
        "mean_score": round(float(df.m_iou_conf.mean()), 3),
        "area_frac": {
            "severe": round(float(sev.m_area_frac.mean()), 4),
            "nonsevere": round(float(non.m_area_frac.mean()), 4),
        },
        "contrast": {
            "severe": round(float(sev.m_contrast.mean()), 4),
            "nonsevere": round(float(non.m_contrast.mean()), 4),
        },
        "feature_cols": FEATURE_COLS,
    }
    QC.parent.mkdir(parents=True, exist_ok=True)
    QC.write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2))
    print(f"\nwrote {OUTDIR / 'features.parquet'} ({len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
