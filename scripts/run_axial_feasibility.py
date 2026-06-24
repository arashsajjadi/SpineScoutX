#!/usr/bin/env python3
"""Axial subarticular route — level-matching feasibility on the locked test.

Quantifies whether sagittal-disc-z → axial-slice-z level matching (no GT) assigns lumbar
levels to the correct axial slice, by comparing matched vs GT subarticular axial instance
(GT used for evaluation only). This is the gating sub-problem for an axial subarticular
auto grader. Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spinescoutx.data.axial_match import SUBARTICULAR, qc_level_matching
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
ORACLE = ROOT / "data/cache/rsna"
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
LOC_RUN = ROOT / "runs/l0_disc_localizer_real"
OUT = ROOT / "outputs/real/axial_matching_qc.json"
DOC = ROOT / "docs/run_logs/subarticular_auto_results.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["dev", "test"])
    ap.add_argument("--limit-studies", type=int, default=200)
    args = ap.parse_args()

    split_map = load_splits_v1(SPLITS)
    man = read_manifest(ORACLE / "manifest.parquet")
    man["study_id"] = man.study_id.astype(str)
    sub_studies = man[man.condition.isin(SUBARTICULAR)].study_id.unique()
    studies = [s for s in sub_studies if split_map.get(s) in args.splits]
    if args.limit_studies:
        studies = sorted(studies)[: args.limit_studies]
    print(f"[axial] QC level-matching on {len(studies)} studies ({'+'.join(args.splits)})")

    qc = qc_level_matching(ROOT / "data/raw/rsna", LOC_RUN, studies)
    qc["splits"] = args.splits
    qc["n_studies_requested"] = len(studies)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2))
    _doc(qc)
    print(f"[axial] wrote {OUT} and {DOC}")
    return 0


def _doc(qc: dict) -> None:
    sd = qc["slice_distance"]
    feasible = sd["within_1"] is not None and sd["within_1"] >= 0.8
    lines = [
        "# Axial subarticular route — level-matching feasibility (locked test)",
        "",
        "> Research-only. Not diagnostic. Subarticular stenosis is graded on **axial-T2**.",
        "> The gating sub-problem is assigning each lumbar level to the correct axial slice",
        "> with no GT. This QC measures the z-based matcher (sagittal-disc-z → axial-slice-z)",
        "> against the GT subarticular axial instance (GT used for evaluation only).",
        "",
        f"Studies scored: {qc['n_studies']} ({qc['n_levels_scored']} level matches); "
        f"skipped {qc['skipped']}.",
        "",
        "## Matched-vs-GT axial slice distance",
        "",
        "| metric | value |",
        "|---|---|",
        f"| median |Δslice| | {sd['median']} |",
        f"| mean |Δslice| | {sd['mean']:.2f} |"
        if sd["mean"] is not None
        else "| mean |Δslice| | - |",
        f"| within 0 slices | {sd['within_0']:.3f} |"
        if sd["within_0"] is not None
        else "| within 0 | - |",
        f"| within 1 slice | {sd['within_1']:.3f} |"
        if sd["within_1"] is not None
        else "| within 1 | - |",
        f"| within 2 slices | {sd['within_2']:.3f} |"
        if sd["within_2"] is not None
        else "| within 2 | - |",
        f"| median |Δz| (mm) | {qc['z_distance_mm']['median']:.1f} |",
        "",
        "## Verdict",
    ]
    if feasible:
        lines += [
            "- **Level matching is reliable** (≥80% within ±1 axial slice). The axial",
            "  subarticular route is feasible; the remaining step is an **axial in-plane",
            "  localizer** (predict left/right lateral-recess points on the matched axial",
            "  slice), analogous to the sagittal-T1 foraminal localizer, then robust",
            "  auto-training + locked-test eval. Scoped as the immediate next build.",
        ]
    else:
        lines += [
            "- **Level matching is not yet reliable enough** for a headline auto subarticular",
            "  result (see distances above). Likely causes: cross-series geometry mismatch,",
            "  multiple axial stacks, or oblique acquisitions. This is the precise, measured",
            "  **blocker**: the axial route needs a more robust level matcher (e.g.,",
            "  coordinate-supervised slice scoring) before an axial grader is worth training.",
        ]
    lines += [
        "",
        "Either way: **no faked auto subarticular metric is reported.** Oracle locked-test",
        "subarticular baselines (upper bounds) remain in `multicondition_robust_results.md`.",
        "",
        "Artifacts: `outputs/real/axial_matching_qc.json`. Reproduce:",
        "`python scripts/run_axial_feasibility.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
