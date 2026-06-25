"""Reproduce SpineScoutX best-model locked-test metrics (v1.9).

Runs the deployed graders over the locked test split and prints the severe recall
numbers that match the published results. Does NOT re-train anything. Does NOT
use the locked test for selection — only for evaluation (already frozen).
Research-only. Not diagnostic.

Expected output (match the published numbers):
  canal           : 0.830
  left_foraminal  : 0.788
  right_foraminal : 0.660
  left_subarticular  : 0.746
  right_subarticular : 0.737
  5-route macro   : 0.752
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
RUNS = {
    "canal": ROOT / "runs/v1_canal_auto_robust",
    "foraminal": ROOT / "runs/v1_foraminal_oracle_ctrl",
    "subarticular": ROOT / "runs/v1_subarticular_auto_robust",
}
PUBLISHED = {
    "spinal_canal_stenosis": 0.830,
    "left_neural_foraminal_narrowing": 0.788,
    "right_neural_foraminal_narrowing": 0.660,
    "left_subarticular_stenosis": 0.746,
    "right_subarticular_stenosis": 0.737,
}

CONDITIONS = {
    "canal": ["spinal_canal_stenosis"],
    "foraminal": ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"],
    "subarticular": ["left_subarticular_stenosis", "right_subarticular_stenosis"],
}


def _severe_recall(probs_map: dict, cond: str) -> float:
    true_pos = 0
    total_sev = 0
    for key, (y, p) in probs_map.items():
        if not key.endswith(f"|{cond}"):
            continue
        if y == 2:
            total_sev += 1
            if int(np.argmax(p)) == 2:
                true_pos += 1
    return true_pos / total_sev if total_sev else float("nan")


def main() -> int:
    from spinescoutx.data.crops import read_manifest
    from spinescoutx.data.locked_test import load_splits_v1
    from spinescoutx.evaluation.gap_decomposition import collect_probs
    from spinescoutx.training.optim import select_device

    device = select_device("auto")
    sm = load_splits_v1(SPLITS)
    man = read_manifest(RSNA_CACHE / "manifest.parquet")
    man["study_id"] = man.study_id.astype(str)
    man["split"] = man.study_id.map(sm)
    test_man = man[man.split == "test"].reset_index(drop=True)

    results: dict[str, float] = {}
    tmp = ROOT / "outputs/real/_repro_tmp.parquet"

    for route, conds in CONDITIONS.items():
        run_dir = RUNS[route]
        if not run_dir.exists():
            print(f"  SKIP {route}: run directory missing ({run_dir})")
            continue
        combined: dict[str, tuple[int, np.ndarray]] = {}
        for cond in conds:
            sub = test_man[test_man.condition == cond].copy()
            if sub.empty:
                continue
            sub.to_parquet(tmp)
            probs = collect_probs(run_dir, tmp, RSNA_CACHE, device)
            for k, (y, p) in probs.items():
                st, lv = k.split("|")
                combined[f"{st}|{lv}|{cond}"] = (int(y), np.asarray(p, float))
        for cond in conds:
            r = _severe_recall(combined, cond)
            results[cond] = r

    if tmp.exists():
        tmp.unlink()

    print("\n=== SpineScoutX v1.9 — locked-test severe recall ===")
    print(f"{'Condition':<38}  {'Computed':>9}  {'Published':>9}  {'Match':>5}")
    print("-" * 70)
    all_ok = True
    vals = []
    for cond, pub in PUBLISHED.items():
        comp = results.get(cond, float("nan"))
        ok = abs(comp - pub) < 0.002 if not np.isnan(comp) else False
        all_ok = all_ok and ok
        match = "✅" if ok else "⚠"
        print(f"{cond:<38}  {comp:>9.3f}  {pub:>9.3f}  {match}")
        if not np.isnan(comp):
            vals.append(comp)
    macro = float(np.mean(vals)) if vals else float("nan")
    pub_macro = 0.752
    ok_macro = abs(macro - pub_macro) < 0.002
    all_ok = all_ok and ok_macro
    print("-" * 70)
    print(f"{'5-route macro':<38}  {macro:>9.3f}  {pub_macro:>9.3f}  {'✅' if ok_macro else '⚠'}")

    out = {"computed": results, "macro": macro, "published": PUBLISHED, "all_match": all_ok}
    (ROOT / "outputs/real/v1_9_reproduce_metrics.json").write_text(json.dumps(out, indent=2))

    print(f"\nReproduction {'PASSED' if all_ok else 'WARNING — check deviations'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
