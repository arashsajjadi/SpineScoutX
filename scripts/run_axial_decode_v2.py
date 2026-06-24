#!/usr/bin/env python3
"""Axial decode v2 — positional-prior-regularized monotonic decoding (localization).

The per-slice axial level scorer reaches ±1-slice hit ~0.43; the current decoder only
enforces level order. v2 adds a TRAIN-derived positional prior (each level's typical
normalized-z) to the decode (`assign_levels_monotonic_prior`) — no CNN retrain, no test data
in the prior. We select the prior weight beta on DEV and evaluate ±0/±1/±2 slice-hit on the
locked TEST once. Geometry baseline (~0.275) and current scorer (~0.43) are the references.

GT axial level instances are used for slice-hit SCORING only (this is an evaluation of the
localizer, not auto inference). Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spinescoutx.data.axial_level import (
    assign_levels_monotonic,
    assign_levels_monotonic_prior,
    level_position_prior,
)
from spinescoutx.data.axial_match import axial_z_by_instance, pick_axial_t2
from spinescoutx.data.locked_test import load_splits_v1

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
CACHE = ROOT / "data/cache/axial_level"
RUN = ROOT / "runs/axial_level_scorer"
OUT = ROOT / "outputs/real/axial_stack_scorer_v2_results.json"
DOC = ROOT / "docs/run_logs/axial_stack_scorer_v2_results.md"
FIG = ROOT / "outputs/real/figures/axial_decode_v2.png"
ASSET = ROOT / "docs/assets/readme/axial_decode_v2_before_after.png"
LEVELS = ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")


def _score_stack(model, images_dir, study, series, slice_size, device):
    """Return (zsorted instances, logps [n,5], norm_zs [n]) for an axial stack."""
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
        try:
            img = normalize_intensity(read_dicom(images_dir / study / series / f"{inst}.dcm"))
        except Exception:  # noqa: BLE001
            continue
        resized = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            logit = model(
                torch.from_numpy(resized[None, None]).float().to(device),
                torch.tensor([[r / (n - 1)]], dtype=torch.float32).to(device),
            )
            logps[r] = torch.log_softmax(logit, dim=1)[0].cpu().numpy()
    norm_zs = np.array([r / (n - 1) for r in range(n)])
    return zsorted, logps, norm_zs


def _gt_ranks(coords, study, series, zsorted):
    """{level_idx -> gt z-rank} for the subarticular GT instances in this stack."""
    rank = {inst: r for r, inst in enumerate(zsorted)}
    g = coords[(coords.study_id == study) & (coords.series_id.astype(str) == str(series))]
    out = {}
    for li, lv in enumerate(LEVELS):
        gl = g[g.level == lv]
        if gl.empty:
            continue
        inst = int(gl.instance_number.median())
        if inst in rank:
            out[li] = rank[inst]
    return out


def _hits(assign, gt, n):
    """Per-level abs rank error; return list of (abs_err)."""
    errs = []
    for li, gr in gt.items():
        errs.append(abs(assign[li] - gr))
    return errs


def _agg(errs):
    e = np.array(errs)
    if len(e) == 0:
        return {}
    return {
        "n": int(len(e)),
        "hit_0": float((e == 0).mean()),
        "hit_1": float((e <= 1).mean()),
        "hit_2": float((e <= 2).mean()),
        "median_abs_err": float(np.median(e)),
    }


def _eval_split(model, studies, coords, images_dir, slice_size, device, betas):
    """Return {beta: errs_list} over the given studies (beta=0 == current decoder)."""
    from spinescoutx.data.rsna_index import build_series_index

    series_idx = build_series_index(ROOT / "data/raw/rsna")
    series_idx["study_id"] = series_idx.study_id.astype(str)
    series_idx["series_id"] = series_idx.series_id.astype(str)
    prior = level_position_prior(CACHE)
    errs = {b: [] for b in betas}
    n_done = 0
    for study in studies:
        series = pick_axial_t2(series_idx, study, images_dir)
        if series is None:
            continue
        scored = _score_stack(model, images_dir, study, series, slice_size, device)
        if scored is None:
            continue
        zsorted, logps, norm_zs = scored
        gt = _gt_ranks(coords, study, series, zsorted)
        if not gt:
            continue
        n_done += 1
        for b in betas:
            assign = (
                assign_levels_monotonic(logps)
                if b == 0.0
                else assign_levels_monotonic_prior(logps, norm_zs, prior, beta=b)
            )
            errs[b].extend(_hits(assign, gt, len(zsorted)))
    return errs, n_done


def main() -> int:
    import argparse

    from spinescoutx.data.axial_level import load_axial_level_scorer
    from spinescoutx.data.rsna_index import RsnaPaths
    from spinescoutx.data.rsna_labels import load_coordinates
    from spinescoutx.training.optim import select_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-studies", type=int, default=150)
    ap.add_argument("--test-studies", type=int, default=0)
    args = ap.parse_args()

    device = select_device("auto")
    model, slice_size = load_axial_level_scorer(RUN, device)
    model.eval()
    images_dir = Path(RsnaPaths.from_root(ROOT / "data/raw/rsna").train_images_dir)
    split_map = load_splits_v1(SPLITS)
    coords = load_coordinates(ROOT / "data/raw/rsna")
    coords["study_id"] = coords.study_id.astype(str)
    from spinescoutx.constants import CONDITIONS

    sub_conds = [c for c in CONDITIONS if "subarticular" in c]
    coords = coords[coords.condition.isin(sub_conds)].copy()

    dev = sorted([s for s, sp in split_map.items() if sp == "dev"])
    test = sorted([s for s, sp in split_map.items() if sp == "test"])
    if args.dev_studies:
        dev = dev[: args.dev_studies]
    if args.test_studies:
        test = test[: args.test_studies]

    # DEV: select beta
    betas = [0.0, 0.5, 1.0, 2.0, 3.0]
    print(f"[axial-v2] dev selection over {len(dev)} studies ...", flush=True)
    dev_errs, dev_n = _eval_split(model, dev, coords, images_dir, slice_size, device, betas)
    dev_agg = {b: _agg(dev_errs[b]) for b in betas}
    best_beta = max([b for b in betas if b > 0], key=lambda b: dev_agg[b].get("hit_1", 0))
    if dev_agg[best_beta].get("hit_1", 0) <= dev_agg[0.0].get("hit_1", 0):
        best_beta = 0.0  # prior does not help on dev -> keep current
    print(
        f"[axial-v2] dev hit_1: current {dev_agg[0.0]['hit_1']:.3f} | "
        f"best beta={best_beta} {dev_agg[best_beta]['hit_1']:.3f}",
        flush=True,
    )

    # TEST: current vs selected beta (once)
    print(f"[axial-v2] test eval over {len(test)} studies ...", flush=True)
    test_betas = sorted({0.0, best_beta})
    test_errs, test_n = _eval_split(model, test, coords, images_dir, slice_size, device, test_betas)
    test_agg = {b: _agg(test_errs[b]) for b in test_betas}

    out = {
        "protocol": "splits_v1; dev-selected beta, locked-test once",
        "geometry_baseline_hit_1": 0.275,
        "selected_beta": best_beta,
        "dev": {str(b): dev_agg[b] for b in betas},
        "dev_n_studies": dev_n,
        "test_current_decode": test_agg.get(0.0, {}),
        "test_prior_decode": test_agg.get(best_beta, {}),
        "test_n_studies": test_n,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    _fig(out)
    _doc(out)
    cur, new = out["test_current_decode"], out["test_prior_decode"]
    print(
        f"[axial-v2] TEST hit_1: current {cur.get('hit_1', float('nan')):.3f} -> "
        f"prior(beta={best_beta}) {new.get('hit_1', float('nan')):.3f} (n={cur.get('n')})",
        flush=True,
    )
    print(f"wrote {OUT}\nwrote {DOC}\nwrote {FIG} + {ASSET}")
    return 0


def _fig(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cur, new = out["test_current_decode"], out["test_prior_decode"]
    metrics = ["hit_0", "hit_1", "hit_2"]
    labels = ["±0 slice", "±1 slice", "±2 slice"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=100)
    ax.bar(x - 0.2, [cur.get(m, 0) for m in metrics], 0.4, label="current decoder", color="#90a4ae")
    ax.bar(
        x + 0.2,
        [new.get(m, 0) for m in metrics],
        0.4,
        label=f"v2 prior decoder (β={out['selected_beta']})",
        color="#00838f",
    )
    ax.axhline(out["geometry_baseline_hit_1"], ls="--", c="red", lw=1, label="geometry ±1 (0.275)")
    for i, m in enumerate(metrics):
        ax.text(i - 0.2, cur.get(m, 0) + 0.01, f"{cur.get(m, 0):.2f}", ha="center", fontsize=10)
        ax.text(i + 0.2, new.get(m, 0) + 0.01, f"{new.get(m, 0):.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("axial slice-hit rate (locked test)", fontsize=12)
    ax.set_title("Axial decode v2 — positional-prior monotonic decoding", fontsize=13)
    ax.legend(fontsize=10)
    fig.text(
        0.5,
        0.01,
        "Localization eval (GT used to score slice-hit only) · research-only",
        ha="center",
        fontsize=9,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET, facecolor="white")
    plt.close(fig)


def _doc(out):
    cur, new = out["test_current_decode"], out["test_prior_decode"]
    b = out["selected_beta"]
    improved = new.get("hit_1", 0) > cur.get("hit_1", 0) + 0.005
    lines = [
        "# Axial decode v2 — positional-prior monotonic decoding (localization)",
        "",
        "> Research-only · not diagnostic. A TRAIN-derived per-level normalized-z prior is added",
        "> to the monotonic decode (`assign_levels_monotonic_prior`) — no CNN retrain, no test",
        "> data in the prior. Beta selected on DEV, locked-test evaluated once. GT used to score",
        "> slice-hit only (localizer evaluation, not auto inference).",
        "",
        f"Selected beta (dev) = **{b}**. Geometry baseline ±1-hit ≈ 0.275.",
        "",
        "## Locked-test slice-hit: current decoder vs v2 prior decoder",
        "| metric | current | v2 prior |",
        "|---|---|---|",
    ]

    def _row(name, key, fmt=".3f"):
        c = format(cur.get(key, float("nan")), fmt)
        nw = format(new.get(key, float("nan")), fmt)
        return f"| {name} | {c} | {nw} |"

    lines += [
        _row("±0 slice", "hit_0"),
        _row("±1 slice", "hit_1"),
        _row("±2 slice", "hit_2"),
        _row("median abs err", "median_abs_err", ".2f"),
        f"(n = {cur.get('n')} level-instances over {out['test_n_studies']} test studies)",
        "",
        "## Verdict (honest)",
    ]
    if b == 0.0:
        lines.append(
            "- The positional prior did **not** beat the current decoder on dev → kept current "
            "decoder (documented negative; the scorer already conditions on norm_z, so the decode "
            "prior is largely redundant)."
        )
    elif improved:
        lines.append(
            f"- The v2 prior decoder **improves** ±1 slice-hit "
            f"({cur.get('hit_1', 0):.3f} → {new.get('hit_1', 0):.3f}) on the locked test — a real, "
            "no-retrain localization gain. Downstream subarticular grading is robust to leveling "
            "noise (v1.1/v1.2), so the main value is route trust/quality, not severe recall."
        )
    else:
        lines.append(
            "- The v2 prior decoder is **within noise** of the current decoder on the locked test "
            "(honest negative): the scorer already uses norm_z, so a decode prior adds little."
        )
    lines += [
        "",
        "Reproduce: `python scripts/run_axial_decode_v2.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
