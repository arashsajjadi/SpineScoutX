#!/usr/bin/env python3
"""v1.5: does the BiGRU sequence localizer raise subarticular severe recall (recrop -> regrade)?

The BiGRU axial level refiner improved locked-test +-1 slice-hit 0.487 -> 0.616 (median abs err
2 -> 1). Here we re-crop the subarticular evidence at the BiGRU-decoded slice and re-grade with the
DEPLOYED grader (fixed), *paired* against the current and prior decoders on the SAME scored stacks.
If the much-larger localization gain finally moves grading, that is a real raw-recall win
(target subarticular +>=0.03 abs). DEV first; locked-test once. GT axial coords are NOT used
(slices come from the scorer/refiner); GT severity scores recall only.

Research-only. Not diagnostic. Reproduce: `python scripts/run_subarticular_recrop_bigru_v1_5.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.axial_level import (
    SUBARTICULAR_COND,
    SUBARTICULAR_OFFSETS,
    assign_levels_monotonic,
    assign_levels_monotonic_prior,
    level_position_prior,
    load_axial_level_scorer,
)
from spinescoutx.data.axial_match import SUBARTICULAR, axial_z_by_instance, pick_axial_t2
from spinescoutx.data.crops import extract_25d
from spinescoutx.data.datasets import _to_3chw
from spinescoutx.data.dicom_io import normalize_intensity, read_dicom
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.models.axial_seq_refiner import build_axial_seq_refiner

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
CACHE = ROOT / "data/cache/axial_level"
SCORER = ROOT / "runs/axial_level_scorer"
GRADER = ROOT / "runs/v1_subarticular_auto_robust"
REFINER = ROOT / "runs/axial_seq_refiner_v1_5/best.pt"
OUT = ROOT / "outputs/real/subarticular_recrop_bigru_v1_5.json"
CROP_SIZE = 224
DECODES = ("current", "prior", "bigru")


def _score_stack(model, images_dir, study, series, slice_size, device):
    import cv2

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


def _refine(ref, logps, norm_z, device):
    feats = np.concatenate([logps, norm_z[:, None]], axis=1).astype(np.float32)
    with torch.no_grad():
        logit = ref(torch.from_numpy(feats)[None].to(device), torch.tensor([feats.shape[0]]))[0]
    return torch.log_softmax(logit, dim=1).cpu().numpy()


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


def run_split(split, model_s, slice_size, model_g, ref, images_dir, series, labels, prior, sm, dev):
    studies = sorted([s for s, sp in sm.items() if sp == split and s in set(labels.study_id)])
    rows = {d: [] for d in DECODES}  # (crop, li, cd, y, key)
    for study in studies:
        ax = pick_axial_t2(series, study, images_dir)
        if ax is None:
            continue
        scored = _score_stack(model_s, images_dir, study, ax, slice_size, dev)
        if scored is None:
            continue
        zsorted, logps, norm_z = scored
        refined = _refine(ref, logps, norm_z, dev)
        assigns = {
            "current": assign_levels_monotonic(logps),
            "prior": assign_levels_monotonic_prior(logps, norm_z, prior, beta=1.0),
            "bigru": assign_levels_monotonic(refined),
        }
        g = labels[labels.study_id == study]
        for side, cond in SUBARTICULAR_COND.items():
            for _, r in g[g.condition == cond].iterrows():
                li = LEVEL_TO_INDEX.get(str(r.level))
                if li is None or r.severity_index not in (0, 1, 2):
                    continue
                key = f"{study}|{r.level}|{side}"
                for d in DECODES:
                    sidx = assigns[d].get(li)
                    if sidx is None:
                        continue
                    crop = _crop_at(images_dir, study, ax, zsorted[sidx], side)
                    if crop is None:
                        continue
                    rows[d].append((crop, li, CONDITION_TO_INDEX[cond], int(r.severity_index), key))
    # grade each decode, key -> (y, p[3])
    graded = {}
    for d in DECODES:
        rr = rows[d]
        if not rr:
            graded[d] = {}
            continue
        probs = _grade(model_g, [x[0] for x in rr], np.array([x[1] for x in rr]),
                       np.array([x[2] for x in rr]), dev)  # fmt: skip
        graded[d] = {x[4]: (x[3], probs[i]) for i, x in enumerate(rr)}
    return graded


def _metrics(y, p, st):
    r10 = bs.make_recall_at_far(0.10)
    neg = y != 2
    return {
        "severe_recall": float(bs.m_severe_recall(y, p)),
        "severe_recall_ci": bs.bootstrap_ci(y, p, st, bs.m_severe_recall, n_boot=2000),
        "recall_at_far10": float(r10(y, p)),
        "far": float((p[neg].argmax(1) == 2).mean()) if neg.any() else float("nan"),
        "n": int(len(y)),
        "n_severe": int((y == 2).sum()),
    }


def _summarize(graded):
    if not all(graded.values()):
        return {"error": "a decode produced no graded findings"}
    keys = sorted(set.intersection(*[set(graded[d]) for d in DECODES]))
    if not keys:
        return {"error": "no common keys"}
    y = np.array([graded["current"][k][0] for k in keys])
    st = np.array([k.split("|")[0] for k in keys])
    p = {d: np.stack([graded[d][k][1] for k in keys]) for d in DECODES}
    r10 = bs.make_recall_at_far(0.10)

    def paired(a, b):
        return {
            "severe_recall": bs.paired_bootstrap_delta(
                y, a, b, st, bs.m_severe_recall, n_boot=2000
            ),
            "recall_at_far10": bs.paired_bootstrap_delta(y, a, b, st, r10, n_boot=2000),
        }

    return {
        "n_common": len(keys),
        "per_decode": {d: _metrics(y, p[d], st) for d in DECODES},
        "paired_bigru_vs_current": paired(p["bigru"], p["current"]),
        "paired_bigru_vs_prior": paired(p["bigru"], p["prior"]),
    }


def main() -> int:
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
    model_g = _build_model(cfg).to(device).eval()
    model_g.load_state_dict(torch.load(GRADER / "best.pt", map_location=device)["state_dict"])
    ref = build_axial_seq_refiner().to(device).eval()
    ref.load_state_dict(torch.load(REFINER, map_location=device)["state_dict"])
    images_dir = Path(RsnaPaths.from_root(ROOT / "data/raw/rsna").train_images_dir)
    series = build_series_index(ROOT / "data/raw/rsna")
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)
    labels = load_labels(ROOT / "data/raw/rsna")
    labels["study_id"] = labels.study_id.astype(str)
    labels = labels[labels.condition.isin(SUBARTICULAR)].copy()
    prior = level_position_prior(CACHE)
    sm = load_splits_v1(SPLITS)

    out = {"protocol": "splits_v1; paired current/prior/bigru decode; grader fixed", "splits": {}}
    for split in args.splits:
        print(f"[recrop-bigru] {split} ...", flush=True)
        graded = run_split(split, model_s, slice_size, model_g, ref, images_dir, series, labels,
                           prior, sm, device)  # fmt: skip
        r = _summarize(graded)
        out["splits"][split] = r
        if "per_decode" in r:
            pc = r["per_decode"]
            bc = r["paired_bigru_vs_current"]["severe_recall"]
            print(
                f"  severe recall current {pc['current']['severe_recall']:.3f} | "
                f"prior {pc['prior']['severe_recall']:.3f} | "
                f"bigru {pc['bigru']['severe_recall']:.3f}"
                f"  (bigru-current Δ{bc['delta']:+.3f} [{bc['ci_lo']:+.3f},{bc['ci_hi']:+.3f}]"
                f"{' DECISIVE' if bc['decisive'] else ''}, n={r['n_common']})",
                flush=True,
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
