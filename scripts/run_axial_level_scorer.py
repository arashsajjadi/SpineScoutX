#!/usr/bin/env python3
"""Train + evaluate the coordinate-supervised axial level scorer (the axial gate).

Slice-hit (assigned axial slice vs GT subarticular slice within ±1/±2) on dev and locked
test, vs the prior pure-geometry baseline (~27% within ±1). GT used for supervision + QC
only. Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.constants import LEVELS
from spinescoutx.data.axial_level import (
    load_axial_level_scorer,
    prepare_axial_level_data,
    score_and_assign_stack,
    train_axial_level_scorer,
)
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.data.rsna_index import RsnaPaths
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
CACHE = ROOT / "data/cache/axial_level"
RUN = ROOT / "runs/axial_level_scorer"
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUT = ROOT / "outputs/real/axial_level_scorer_results.json"


def _slice_hit(model, slice_size, device, split_map, split):
    man = pd.read_parquet(CACHE / "axial_level_manifest.parquet")
    man["study_id"] = man.study_id.astype(str)
    sp = {"dev": "val", "test": "test"}[split]
    rows = man[man.split == sp]
    images_dir = Path(RsnaPaths.from_root(ROOT / "data/raw/rsna").train_images_dir)
    per_level = {lv: [] for lv in LEVELS}
    d1, d2 = [], []
    for study, g in rows.groupby("study_id"):
        ax_series = str(g.series_id.iloc[0])
        res = score_and_assign_stack(model, images_dir, study, ax_series, slice_size, device)
        if res is None:
            continue
        zsorted = next(iter(res.values()))["zsorted"]
        rank = {inst: r for r, inst in enumerate(zsorted)}
        for r in g.itertuples():
            lv = r.level
            if lv not in res or int(r.instance_number) not in rank:
                continue
            gt_rank = rank[int(r.instance_number)]
            dist = abs(res[lv]["slice_index"] - gt_rank)
            per_level[lv].append(dist)
            d1.append(dist <= 1)
            d2.append(dist <= 2)
    return {
        "n": len(d1),
        "within_1": float(np.mean(d1)) if d1 else None,
        "within_2": float(np.mean(d2)) if d2 else None,
        "per_level_within_1": {
            lv: (float(np.mean(np.array(v) <= 1)) if v else None) for lv, v in per_level.items()
        },
        "median_slice_dist": float(np.median([x for v in per_level.values() for x in v]))
        if any(per_level.values())
        else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--limit-studies", type=int, default=None)
    args = ap.parse_args()
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)

    if not (CACHE / "axial_level_manifest.parquet").exists():
        print("[axial-level] preparing data...")
        s = prepare_axial_level_data(
            ROOT / "data/raw/rsna",
            CACHE,
            split_map,
            slice_size=128,
            limit_studies=args.limit_studies,
        )
        print(f"[axial-level] rows={s['n_rows']} split={s['split']}")
    if not (RUN / "best.pt").exists():
        print("[axial-level] training scorer...")
        train_axial_level_scorer(CACHE, RUN, slice_size=128, epochs=args.epochs)
    model, slice_size = load_axial_level_scorer(RUN, device)
    best = json.loads((RUN / "metrics.json").read_text())["best"]
    print(f"[axial-level] dev level-acc={best.get('dev_level_acc'):.3f}")

    out = {"prior_geometry_within_1": 0.275, "dev_level_acc": best.get("dev_level_acc")}
    for split in ("dev", "test"):
        out[split] = _slice_hit(model, slice_size, device, split_map, split)
        sh = out[split]
        print(
            f"[axial-level] {split}: slice-hit ±1={sh['within_1']:.3f} ±2={sh['within_2']:.3f} "
            f"(n={sh['n']}, median_dist={sh['median_slice_dist']})"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"[axial-level] wrote {OUT}")
    dev1 = out["dev"]["within_1"] or 0
    print(
        f"\n[GATE] dev slice-hit ±1 = {dev1:.3f} vs geometry 0.275 -> "
        f"{'PROCEED to subarticular grader' if dev1 >= 0.55 else 'still weak; stronger blocker'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
