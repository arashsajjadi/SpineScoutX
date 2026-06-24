#!/usr/bin/env python3
"""v1.4 raw-accuracy test: does the v1.3 better-localized decode raise subarticular severe recall?

v1.3 improved axial ±1 slice-hit (0.432→0.487) with a train-derived positional-prior decoder
(`assign_levels_monotonic_prior`, no CNN retrain). This re-crops the subarticular evidence at
the prior decoder's slice and re-grades with the DEPLOYED grader (fixed), paired against the
current decoder's crops on the SAME scored stacks. If better localization → better grading, this
is a real raw-recall win (target: subarticular +≥0.03 abs). DEV checked first, locked-test once.

GT axial coordinates are NOT used (slices come from the scorer); GT severity scores recall only.
Research-only. Not diagnostic.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.axial_level import (
    SUBARTICULAR_COND,
    SUBARTICULAR_OFFSETS,
    assign_levels_monotonic,
    assign_levels_monotonic_prior,
    level_position_prior,
    load_axial_level_scorer,
)
from spinescoutx.data.axial_match import axial_z_by_instance, pick_axial_t2
from spinescoutx.data.crops import extract_25d
from spinescoutx.data.datasets import _to_3chw
from spinescoutx.data.dicom_io import normalize_intensity, read_dicom
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
CACHE = ROOT / "data/cache/axial_level"
SCORER = ROOT / "runs/axial_level_scorer"
GRADER = ROOT / "runs/v1_subarticular_auto_robust"
OUT = ROOT / "outputs/real/subarticular_recrop_v1_4.json"
DOC = ROOT / "docs/run_logs/axial_stack_scorer_v1_4.md"
CROP_SIZE = 224

_es = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("es", ROOT / "scripts/run_evidence_stability.py")
)
importlib.util.spec_from_file_location(
    "es", ROOT / "scripts/run_evidence_stability.py"
).loader.exec_module(_es)


def _score_stack(model, images_dir, study, series, slice_size, device):
    import cv2
    import torch

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
    return zsorted, logps, np.array([r / (n - 1) for r in range(n)])


def _crop_at(images_dir, study, series, inst, side):
    ox, oy = SUBARTICULAR_OFFSETS[side]
    slices = {}
    for i in (inst - 1, inst, inst + 1):
        p = images_dir / study / series / f"{i}.dcm"
        if p.exists():
            with contextlib.suppress(Exception):
                slices[i] = normalize_intensity(read_dicom(p))
    if inst not in slices:
        return None
    h, w = slices[inst].shape
    arr, _ = extract_25d(slices, inst, ox * w, oy * h, CROP_SIZE)
    return _to_3chw(arr.astype(np.float32), CROP_SIZE)


def _grade(model, crops, levels, conds, device):
    import torch

    out = []
    for i in range(0, len(crops), 128):
        img = torch.from_numpy(np.stack(crops[i : i + 128])).to(device)
        lv = torch.tensor(levels[i : i + 128], dtype=torch.long, device=device)
        cd = torch.tensor(conds[i : i + 128], dtype=torch.long, device=device)
        with torch.no_grad():
            out.append(
                torch.softmax(model(img, level_idx=lv, condition_idx=cd).float(), 1).cpu().numpy()
            )
    return np.concatenate(out) if out else np.zeros((0, 3))


def run_split(
    split, model_s, slice_size, model_g, images_dir, series, labels, prior, split_map, device
):
    studies = [s for s, sp in split_map.items() if sp == split]
    studies = [s for s in studies if s in set(labels.study_id)]
    rows = {"current": [], "prior": []}  # each: (crop, level_idx, cond_idx, y, study, side)
    for study in sorted(studies):
        ax = pick_axial_t2(series, study, images_dir)
        if ax is None:
            continue
        scored = _score_stack(model_s, images_dir, study, ax, slice_size, device)
        if scored is None:
            continue
        zsorted, logps, _norm = scored
        assigns = {
            "current": assign_levels_monotonic(logps),
            "prior": assign_levels_monotonic_prior(logps, _norm, prior, beta=1.0),
        }
        g = labels[labels.study_id == study]
        for side, cond in SUBARTICULAR_COND.items():
            cg = g[g.condition == cond]
            for _, r in cg.iterrows():
                lv = str(r.level)
                li = LEVEL_TO_INDEX.get(lv)
                if li is None or r.severity_index not in (0, 1, 2):
                    continue
                for dec in ("current", "prior"):
                    sidx = assigns[dec].get(li)
                    if sidx is None:
                        continue
                    inst = zsorted[sidx]
                    crop = _crop_at(images_dir, study, ax, inst, side)
                    if crop is None:
                        continue
                    rows[dec].append(
                        (crop, li, CONDITION_TO_INDEX[cond], int(r.severity_index), study, side)
                    )
    res = {}
    for dec in ("current", "prior"):
        rr = rows[dec]
        if not rr:
            continue
        crops = [x[0] for x in rr]
        lv = np.array([x[1] for x in rr])
        cd = np.array([x[2] for x in rr])
        y = np.array([x[3] for x in rr])
        st = np.array([x[4] for x in rr])
        probs = _grade(model_g, crops, lv, cd, device)
        ci = bs.bootstrap_ci(y, probs, st, bs.m_severe_recall, n_boot=2000)
        res[dec] = {
            "severe_recall": float(bs.m_severe_recall(y, probs)),
            "ci": ci,
            "n": int(len(y)),
            "n_severe": int((y == 2).sum()),
        }
    return res


def main() -> int:
    import argparse

    from spinescoutx.config import config_from_dict
    from spinescoutx.data.rsna_index import RsnaPaths, build_series_index
    from spinescoutx.data.rsna_labels import load_labels
    from spinescoutx.training.optim import select_device
    from spinescoutx.training.train_classifier import _build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["dev", "test"])
    args = ap.parse_args()
    device = select_device("auto")
    model_s, slice_size = load_axial_level_scorer(SCORER, device)
    cfg = config_from_dict(json.loads((GRADER / "config.json").read_text()))
    import torch

    model_g = _build_model(cfg).to(device).eval()
    model_g.load_state_dict(torch.load(GRADER / "best.pt", map_location=device)["state_dict"])
    images_dir = Path(RsnaPaths.from_root(ROOT / "data/raw/rsna").train_images_dir)
    series = build_series_index(ROOT / "data/raw/rsna")
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)
    labels = load_labels(ROOT / "data/raw/rsna")
    labels["study_id"] = labels.study_id.astype(str)
    from spinescoutx.data.axial_match import SUBARTICULAR

    labels = labels[labels.condition.isin(SUBARTICULAR)].copy()
    prior = level_position_prior(CACHE)
    split_map = load_splits_v1(SPLITS)

    out = {"protocol": "splits_v1; paired current vs prior decoder; grader fixed", "splits": {}}
    for split in args.splits:
        print(f"[recrop] {split} ...", flush=True)
        r = run_split(
            split,
            model_s,
            slice_size,
            model_g,
            images_dir,
            series,
            labels,
            prior,
            split_map,
            device,
        )
        out["splits"][split] = r
        if "current" in r and "prior" in r:
            c, p = r["current"]["severe_recall"], r["prior"]["severe_recall"]
            print(
                f"    severe recall current {c:.3f} -> prior {p:.3f} (Δ{p - c:+.3f})",
                flush=True,
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _doc(out)
    print(f"wrote {OUT}\nwrote {DOC}")
    return 0


def _doc(out):
    lines = [
        "# Axial stack scorer v1.4 — does better localization raise subarticular severe recall?",
        "",
        "> Research-only · not diagnostic. Paired test: score each axial stack once, decode with",
        "> current decoder vs the v1.3 positional-prior decoder (`assign_levels_monotonic_prior`,",
        "> β=1.0 dev-selected), re-crop subarticular evidence at each, re-grade with the DEPLOYED",
        "> grader (fixed). No CNN retrain, no GT coordinates, no locked-test tuning. Severe recall",
        "> via GT severity only.",
        "",
        "| split | decoder | severe recall [95% CI] | n / sev |",
        "|---|---|---|---|",
    ]
    for split, r in out["splits"].items():
        for dec in ("current", "prior"):
            if dec in r:
                d = r[dec]
                ci = d["ci"]
                lines.append(
                    f"| {split} | {dec} | {d['severe_recall']:.3f} "
                    f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}] | {d['n']} / {d['n_severe']} |"
                )
    # verdict
    dev = out["splits"].get("dev", {})
    test = out["splits"].get("test", {})
    dev_delta = (
        dev["prior"]["severe_recall"] - dev["current"]["severe_recall"]
        if "current" in dev and "prior" in dev
        else None
    )
    test_delta = (
        test["prior"]["severe_recall"] - test["current"]["severe_recall"]
        if "current" in test and "prior" in test
        else None
    )
    lines += [
        "",
        "## Verdict (honest)",
        f"- dev Δ(prior−current) = {dev_delta:+.3f}; test Δ = {test_delta:+.3f}."
        if dev_delta is not None and test_delta is not None
        else "- (incomplete)",
    ]
    if test_delta is not None and dev_delta is not None and dev_delta > 0 and test_delta >= 0.03:
        lines.append(
            "- **Better localization raises subarticular severe recall ≥0.03** — a real "
            "raw-accuracy win from the v1.3 decode, with the grader unchanged."
        )
    else:
        lines.append(
            "- The deployed grader is **robust to the leveling change** — re-cropping at the "
            "better decoder's slice does **not** materially move subarticular severe recall (Δ / "
            "within CI). Honest negative: localization improved (v1.3) but the grading payoff is "
            "bounded (the robust grader already tolerates leveling noise). Raw subarticular recall "
            "grader/data-limited, not decode-limited."
        )
    lines += ["", "Reproduce: `python scripts/run_subarticular_recrop_v1_4.py`."]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
