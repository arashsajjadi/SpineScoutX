#!/usr/bin/env python3
"""Phase 1 runner: 2x2 oracle->auto gap decomposition (canal, E0). Research-only.

Generates the four crop cells (C1..C4), evaluates the existing E0 image-only model
on each on the identical matched node set, and reports bootstrap 95% CIs, paired
deltas (vs oracle), the slice-vs-inplane attribution, and a McNemar test on severe
hits. Writes outputs/real/gap_decomposition_2x2.json (+ figure). No retraining.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import (
    CELL_PROVENANCE,
    CELLS,
    align_cells,
    build_decomposition_crops,
    collect_probs,
)
from spinescoutx.training.optim import select_device

# sklearn warns when a bootstrap resample omits a class; expected and harmless here.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
OUT_JSON = ROOT / "outputs/real/gap_decomposition_2x2.json"
OUT_FIG = ROOT / "outputs/real/figures/gap_decomposition_2x2.png"
CACHE = ROOT / "data/cache/rsna_gap2x2"
E0_RUN = ROOT / "runs/e0_baseline_real"
LOC_RUN = ROOT / "runs/l0_disc_localizer_real"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-studies", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    device = select_device("auto")
    print(f"[gap2x2] device={device}")

    # ---- build crops (resume-safe) ---------------------------------------- #
    summ_path = CACHE / "build_summary.json"
    if args.rebuild or not summ_path.exists():
        print("[gap2x2] building 4 cells from DICOM (holding series=GT series)...")
        summary = build_decomposition_crops(
            ROOT / "data/raw/rsna",
            LOC_RUN,
            CACHE,
            oracle_cache=ROOT / "data/cache/rsna",
            split="val",
            crop_size=224,
            limit_studies=args.limit_studies,
            device="auto",
        )
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps(summary, indent=2))
    else:
        summary = json.loads(summ_path.read_text())
    print(
        f"[gap2x2] matched nodes={summary['n_matched_nodes']} "
        f"skip_localize={summary['skipped_localize_nodes']} "
        f"skip_slice={summary['skipped_slice_nodes']}"
    )

    # ---- evaluate E0 on each cell ----------------------------------------- #
    cell_preds = {}
    for cell in CELLS:
        c = summary["cells"][cell]
        preds = collect_probs(E0_RUN, c["manifest"], c["cache_root"], device)
        cell_preds[cell] = preds
        print(f"[gap2x2] {cell}: {len(preds)} nodes scored")

    studies, y, probs = align_cells(cell_preds)
    n = len(y)
    n_sev = int((y == bs.SEVERE_INDEX).sum())
    print(f"[gap2x2] aligned nodes={n} severe={n_sev}")

    # ---- per-cell metrics + CIs ------------------------------------------- #
    results = {
        "condition": "spinal_canal_stenosis",
        "split": "val",
        "model": "E0_image_only",
        "n_nodes": n,
        "n_severe": n_sev,
        "n_boot": args.n_boot,
        "design": "series held fixed to GT series; only xy-source and slice-source vary",
        "cells": {},
    }
    for cell in CELLS:
        tab = bs.ci_table(y, probs[cell], studies, n_boot=args.n_boot)
        results["cells"][cell] = {"provenance": CELL_PROVENANCE[cell], "metrics": tab}

    # ---- paired deltas vs oracle (C1) ------------------------------------- #
    base = "c1_gtxy_gtslice"
    deltas = {}
    for cell in CELLS:
        if cell == base:
            continue
        deltas[cell] = {
            "severe_recall": bs.paired_bootstrap_delta(
                y, probs[cell], probs[base], studies, bs.m_severe_recall, n_boot=args.n_boot
            ),
            "weighted_logloss": bs.paired_bootstrap_delta(
                y, probs[cell], probs[base], studies, bs.m_weighted_logloss, n_boot=args.n_boot
            ),
        }
    results["paired_delta_vs_oracle"] = deltas

    # ---- additivity / interaction (does C2+C3 explain C4?) ---------------- #
    def sr(cell: str) -> float:
        return bs.m_severe_recall(y, probs[cell])

    def wll(cell: str) -> float:
        return bs.m_weighted_logloss(y, probs[cell])

    c1, c2, c3, c4 = (sr(c) for c in CELLS)
    w1, w2, w3, w4 = (wll(c) for c in CELLS)
    results["attribution"] = {
        "severe_recall": {
            "oracle_c1": c1,
            "inplane_only_c2": c2,
            "slice_only_c3": c3,
            "combined_c4": c4,
            "drop_from_inplane": c1 - c2,
            "drop_from_slice": c1 - c3,
            "drop_combined": c1 - c4,
            "interaction_c4_minus_additive": c4 - (c2 + c3 - c1),
        },
        "weighted_logloss": {
            "oracle_c1": w1,
            "inplane_only_c2": w2,
            "slice_only_c3": w3,
            "combined_c4": w4,
            "rise_from_inplane": w2 - w1,
            "rise_from_slice": w3 - w1,
            "rise_combined": w4 - w1,
        },
    }

    # ---- McNemar on severe hits: C1 vs C4 --------------------------------- #
    results["mcnemar_severe_c1_vs_c4"] = bs.mcnemar_severe(
        y, np.argmax(probs[base], axis=1), np.argmax(probs["c4_autoxy_midslice"], axis=1)
    )

    # ---- per-level severe recall (C1..C4) --------------------------------- #
    # rebuild level array from keys
    shared_keys = sorted(set.intersection(*[set(p) for p in cell_preds.values()]))
    levels = np.array([k.split("|")[1] for k in shared_keys])
    per_level = {}
    for lv in sorted(set(levels.tolist())):
        m = levels == lv
        per_level[lv] = {
            "n": int(m.sum()),
            "n_severe": int((y[m] == bs.SEVERE_INDEX).sum()),
            **{cell: bs.m_severe_recall(y[m], probs[cell][m]) for cell in CELLS},
        }
    results["per_level_severe_recall"] = per_level

    # ---- established anchors (from oracle_auto_gap.json) ------------------- #
    gap_path = ROOT / "outputs/real/oracle_auto_gap.json"
    if gap_path.exists():
        results["established_anchors"] = json.loads(gap_path.read_text())["results"]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"[gap2x2] wrote {OUT_JSON}")

    _print_table(results)
    _make_figure(results)
    return 0


def _print_table(results: dict) -> None:
    print("\n=== 2x2 GAP DECOMPOSITION (E0, canal val, series=GT) ===")
    print(f"n={results['n_nodes']} severe={results['n_severe']}\n")
    hdr = f"{'cell':<20}{'xy':<6}{'slice':<14}{'sevR [95% CI]':<26}{'wll [95% CI]':<26}"
    print(hdr)
    for cell in CELLS:
        c = results["cells"][cell]
        prov = c["provenance"]
        sr = c["metrics"]["severe_recall"]
        wl = c["metrics"]["weighted_logloss"]
        print(
            f"{cell:<20}{prov['crop_xy_source']:<6}{prov['slice_source']:<14}"
            f"{sr['point']:.3f} [{sr['ci_lo']:.3f},{sr['ci_hi']:.3f}]   "
            f"{wl['point']:.3f} [{wl['ci_lo']:.3f},{wl['ci_hi']:.3f}]"
        )
    a = results["attribution"]["severe_recall"]
    print("\nSevere-recall drop attribution:")
    print(f"  in-plane only (C1-C2): {a['drop_from_inplane']:+.3f}")
    print(f"  slice    only (C1-C3): {a['drop_from_slice']:+.3f}")
    print(f"  combined      (C1-C4): {a['drop_combined']:+.3f}")
    print(f"  interaction          : {a['interaction_c4_minus_additive']:+.3f}")
    mc = results["mcnemar_severe_c1_vs_c4"]
    print(
        f"\nMcNemar severe C1 vs C4: b(catch->miss)={mc['b_a_catches_b_misses']} "
        f"c(miss->catch)={mc['c_a_misses_b_catches']} p={mc['p_value']:.4g}"
    )


def _make_figure(results: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[gap2x2] figure skipped: {exc}")
        return
    cells = list(CELLS)
    labels = [
        f"{c.split('_')[0].upper()}\n{results['cells'][c]['provenance']['crop_xy_source']}xy\n"
        f"{results['cells'][c]['provenance']['slice_source']}"
        for c in cells
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in (
        (axes[0], "severe_recall", "Severe recall (auto = real)"),
        (axes[1], "weighted_logloss", "Weighted log loss (lower better)"),
    ):
        pts = [results["cells"][c]["metrics"][key]["point"] for c in cells]
        lo = [results["cells"][c]["metrics"][key]["ci_lo"] for c in cells]
        hi = [results["cells"][c]["metrics"][key]["ci_hi"] for c in cells]
        err = [
            [p - lv for p, lv in zip(pts, lo, strict=False)],
            [hv - p for p, hv in zip(pts, hi, strict=False)],
        ]
        colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
        ax.bar(range(len(cells)), pts, yerr=err, capsize=5, color=colors)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Oracle->auto gap decomposition (E0, canal val, n={results['n_nodes']}, "
        f"severe={results['n_severe']}) — research-only, not diagnostic",
        fontsize=10,
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=120)
    print(f"[gap2x2] wrote {OUT_FIG}")


if __name__ == "__main__":
    raise SystemExit(main())
