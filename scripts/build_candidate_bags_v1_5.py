#!/usr/bin/env python3
"""Build route-specific candidate-bag caches (v1.5) for MIL grading.

Foraminal (sagittal-T1) and subarticular (axial-T2) bags: K candidate crops per
(study, level, side) + per-candidate confidence + label, for train/dev/test. Bags are gitignored
(`data/cache/v1_5_candidate_bags/`); crops are float16 (K,3,224,224). Reads NO ground-truth
coordinates (auto localizers/scorer only). Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.data import candidate_bags as cb
from spinescoutx.data.axial_match import axial_z_by_instance, pick_axial_t2
from spinescoutx.data.foraminal_localize import FORAMINAL, pick_sagittal_t1
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.data.rsna_labels import load_labels

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUTDIR = ROOT / "data/cache/v1_5_candidate_bags"
SUBARTICULAR = ("left_subarticular_stenosis", "right_subarticular_stenosis")
K = 5


def _score_axial_stack(model, images_dir, study, series, slice_size, device):
    import cv2
    import torch

    from spinescoutx.data.dicom_io import normalize_intensity, read_dicom

    azs = axial_z_by_instance(images_dir, study, series)
    if len(azs) < 5:
        return None
    zsorted = sorted(azs, key=lambda i: azs[i])
    n = len(zsorted)
    logps = np.full((n, 5), -20.0)
    for r, inst in enumerate(zsorted):
        with contextlib.suppress(Exception):
            img = normalize_intensity(read_dicom(images_dir / study / series / f"{inst}.dcm"))
            resized = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
            with torch.no_grad():
                logit = model(
                    torch.from_numpy(resized[None, None]).float().to(device),
                    torch.tensor([[r / (n - 1)]], dtype=torch.float32).to(device),
                )
                logps[r] = torch.log_softmax(logit, dim=1)[0].cpu().numpy()
    return zsorted, logps


def _save_bags(bags, route_dir, study, split, records):
    bagdir = route_dir / "bags"
    bagdir.mkdir(parents=True, exist_ok=True)
    for b in bags:
        r = b["label_row"]
        rel = f"bags/{study}_{b['condition']}_{b['level']}_{b['side']}.npy"
        np.save(route_dir / rel, b["crops"])
        records.append(
            {
                "study_id": str(study),
                "condition": b["condition"],
                "side": b["side"],
                "level": b["level"],
                "severity": str(r.severity),
                "severity_index": int(r.severity_index),
                "split": split,
                "k": int(b["crops"].shape[0]),
                "cand_conf": json.dumps(b["cand_conf"]),
                "bag_path": rel,
                "coordinate_source": "auto",
            }
        )


def build_foraminal(studies, split_map, labels, series, images_dir, device):
    from spinescoutx.data.foraminal_localize import _load_foraminal_localizer

    model, slice_size = _load_foraminal_localizer(ROOT / "runs/lf_foraminal_localizer", device)
    route_dir = OUTDIR / "foraminal"
    records: list[dict] = []
    lab = labels[labels.condition.isin(FORAMINAL)]
    done = 0
    for study in studies:
        g = lab[lab.study_id == study]
        if g.empty:
            continue
        series_id = pick_sagittal_t1(series, study, images_dir)
        if series_id is None:
            continue
        level_labels = {(str(r.side), str(r.level)): r for r in g.itertuples()}
        bags = cb.foraminal_bags(
            images_dir, study, series_id, model, slice_size, device, level_labels, k=K
        )
        _save_bags(bags, route_dir, study, split_map[study], records)
        done += 1
        if done % 100 == 0:
            print(f"  [foraminal] {done} studies, {len(records)} bags", flush=True)
    pd.DataFrame(records).to_parquet(route_dir / "bag_manifest.parquet", index=False)
    print(f"[foraminal] {len(records)} bags from {done} studies", flush=True)
    return len(records)


def build_subarticular(studies, split_map, labels, series, images_dir, device):
    from spinescoutx.data.axial_level import load_axial_level_scorer

    model, slice_size = load_axial_level_scorer(ROOT / "runs/axial_level_scorer", device)
    route_dir = OUTDIR / "subarticular"
    records: list[dict] = []
    lab = labels[labels.condition.isin(SUBARTICULAR)]
    done = 0
    for study in studies:
        g = lab[lab.study_id == study]
        if g.empty:
            continue
        ax = pick_axial_t2(series, study, images_dir)
        if ax is None:
            continue
        scored = _score_axial_stack(model, images_dir, study, ax, slice_size, device)
        if scored is None:
            continue
        zsorted, logps = scored
        level_labels = {(str(r.side), str(r.level)): r for r in g.itertuples()}
        bags = cb.subarticular_bags(images_dir, study, ax, logps, zsorted, level_labels, k=K)
        _save_bags(bags, route_dir, study, split_map[study], records)
        done += 1
        if done % 100 == 0:
            print(f"  [subarticular] {done} studies, {len(records)} bags", flush=True)
    pd.DataFrame(records).to_parquet(route_dir / "bag_manifest.parquet", index=False)
    print(f"[subarticular] {len(records)} bags from {done} studies", flush=True)
    return len(records)


def main() -> int:
    from spinescoutx.data.rsna_index import RsnaPaths, build_series_index
    from spinescoutx.training.optim import select_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="*", default=["foraminal", "subarticular"])
    ap.add_argument("--max-studies", type=int, default=0)
    args = ap.parse_args()
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)
    labels = load_labels(ROOT / "data/raw/rsna")
    labels["study_id"] = labels.study_id.astype(str)
    labels["side"] = labels["side"].fillna("").astype(str)
    series = build_series_index(ROOT / "data/raw/rsna")
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)
    images_dir = Path(RsnaPaths.from_root(ROOT / "data/raw/rsna").train_images_dir)
    studies = sorted([s for s in split_map if split_map[s] in ("train", "dev", "test")])
    if args.max_studies:
        studies = studies[: args.max_studies]
    print(f"[bags] {len(studies)} studies, routes={args.routes}, K={K}", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if "foraminal" in args.routes:
        build_foraminal(studies, split_map, labels, series, images_dir, device)
    if "subarticular" in args.routes:
        build_subarticular(studies, split_map, labels, series, images_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
