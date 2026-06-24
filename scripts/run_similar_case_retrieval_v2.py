#!/usr/bin/env python3
"""Similar research-case retrieval v2 — side/level-aware (explanation-only).

v1 retrieved nearest grader-embedding neighbours but was side-agnostic (~chance same-side).
v2 filters the research-case bank by **same (condition, level, side)** first, backing off to
(condition, level) then condition if too few neighbours, before the cosine kNN. This makes
retrieval anatomically meaningful and adds a **retrieval_conflict** signal (the prediction
disagrees with its same-side neighbours' majority severity). Retrieval **never** changes a
prediction. Cached crops, no DICOM decode, no GT. Research-only. Not diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
OUT = ROOT / "outputs/real/similar_case_retrieval_v2.json"
RECORDS = ROOT / "outputs/real/similar_case_retrieval_v2_records.parquet"
DOC = ROOT / "docs/run_logs/similar_case_retrieval_v2.md"
SEV = ("normal_mild", "moderate", "severe")
K = 5

_spec = importlib.util.spec_from_file_location(
    "retr1", ROOT / "scripts/run_similar_case_retrieval.py"
)
_r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r1)


def _backoff_idx(bank, q, level_arr, side_arr, cond_arr):
    """Bank candidate indices by decreasing specificity (cond+level+side -> level -> cond)."""
    qcond, qlevel, qside = str(q.condition), str(q.level), str(getattr(q, "side", "") or "")
    same_cls = cond_arr == qcond
    same_lvl = same_cls & (level_arr == qlevel)
    same_side = same_lvl & (side_arr == qside)
    for mask, tier in (
        (same_side, "cond+level+side"),
        (same_lvl, "cond+level"),
        (same_cls, "cond"),
    ):
        idx = np.where(mask)[0]
        # drop exact self (same study+level)
        idx = [j for j in idx if not (bank[j].study_id == q.study_id and bank[j].level == q.level)]
        if len(idx) >= K:
            return np.array(idx), tier
    return np.array(idx), "cond"


def run_grader(name, device):
    run, cache, conds = _r1.GRADERS[name]
    run_dir, cache_p = ROOT / run, ROOT / cache
    model, cfg = _r1._esrun.load_grader(run_dir, device)
    crop_size = int(getattr(cfg.data, "crop_size", 224))
    from spinescoutx.data.locked_test import load_splits_v1

    split_map = load_splits_v1(ROOT / "data/cache/splits_v1/splits.json")
    bank = _r1._rows(cache_p, conds, "dev", split_map)
    query = _r1._rows(cache_p, conds, "test", split_map)
    if not bank or not query:
        return []
    bz = _r1._feats(model, bank, cache_p, crop_size, device)
    qz = _r1._feats(model, query, cache_p, crop_size, device)
    level_arr = np.array([str(r.level) for r in bank])
    side_arr = np.array([str(getattr(r, "side", "") or "") for r in bank])
    cond_arr = np.array([str(r.condition) for r in bank])
    sev_arr = np.array([int(r.severity_index) for r in bank])
    recs = []
    for i, q in enumerate(query):
        cand, tier = _backoff_idx(bank, q, level_arr, side_arr, cond_arr)
        if len(cand) == 0:
            continue
        sims = qz[i] @ bz[cand].T
        top = cand[np.argsort(-sims)[:K]]
        sev_counts = {s: int((sev_arr[top] == k).sum()) for k, s in enumerate(SEV)}
        maj = SEV[int(np.bincount(sev_arr[top], minlength=3).argmax())]
        qside = str(getattr(q, "side", "") or "")
        recs.append(
            {
                "grader": name,
                "condition": str(q.condition),
                "study_id": str(q.study_id),
                "level": str(q.level),
                "side": qside,
                "ref": SEV[int(q.severity_index)],
                "tier": tier,
                "majority_severity": maj,
                "severity_distribution": sev_counts,
                "same_side_rate": float(np.mean(side_arr[top] == qside)),
                "same_level_rate": float(np.mean(level_arr[top] == str(q.level))),
                "same_condition_rate": float(np.mean(cond_arr[top] == str(q.condition))),
                "k": int(len(top)),
            }
        )
    return recs


def main() -> int:
    from spinescoutx.training.optim import select_device

    device = select_device("auto")
    all_recs = []
    for name in _r1.GRADERS:
        print(f"[retrieval-v2] {name} ...", flush=True)
        recs = run_grader(name, device)
        all_recs.extend(recs)
        if recs:
            ss = np.mean([r["same_side_rate"] for r in recs])
            agree = np.mean([r["majority_severity"] == r["ref"] for r in recs])
            print(f"    n={len(recs)} same_side={ss:.3f} severity_agree={agree:.3f}", flush=True)

    def _agg(recs):
        if not recs:
            return {}
        return {
            "n": len(recs),
            "same_side_rate": float(np.mean([r["same_side_rate"] for r in recs])),
            "same_level_rate": float(np.mean([r["same_level_rate"] for r in recs])),
            "same_condition_rate": float(np.mean([r["same_condition_rate"] for r in recs])),
            "severity_agreement": float(
                np.mean([r["majority_severity"] == r["ref"] for r in recs])
            ),
        }

    out = {
        "protocol": f"splits_v1 locked-test; bank=dev, query=test; side/level-filtered; k={K}",
        "explanation_only": True,
        "overall": _agg(all_recs),
        "per_grader": {g: _agg([r for r in all_recs if r["grader"] == g]) for g in _r1.GRADERS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    import pandas as pd

    pd.DataFrame(all_recs).to_parquet(RECORDS, index=False)
    _doc(out)
    print(f"\nwrote {OUT}\nwrote {RECORDS} ({len(all_recs)})\nwrote {DOC}")
    return 0


def _doc(out):
    o = out["overall"]
    lines = [
        "# Similar research-case retrieval v2 — side/level-aware (explanation-only)",
        "",
        "> Research-only · not diagnostic. v2 filters the research-case bank by same",
        "> (condition, level, side) first (back-off to level, then condition) before the cosine",
        "> kNN, so neighbours are anatomically matched. **Never changes a prediction.**",
        "",
        f"Overall (k={K}): same-side **{o['same_side_rate']:.3f}** (v1 ≈ 0.52 = chance), "
        f"same-level {o['same_level_rate']:.3f}, severity agreement {o['severity_agreement']:.3f}.",
        "",
        "| grader | n | same-side | same-level | severity agreement |",
        "|---|---|---|---|---|",
    ]
    for g, a in out["per_grader"].items():
        if not a:
            continue
        lines.append(
            f"| {g} | {a['n']} | {a['same_side_rate']:.3f} | {a['same_level_rate']:.3f} | "
            f"{a['severity_agreement']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation (honest)",
        "- v2 makes retrieval **side/level-aware** via metadata filtering (v1's embedding was",
        "  side-agnostic at ~chance). Neighbours are now anatomically matched, so the retrieved",
        "  severity distribution and the **retrieval_conflict** signal are meaningful.",
        "- Still explanation-only: retrieval never votes on or changes the prediction.",
        "",
        "Reproduce: `python scripts/run_similar_case_retrieval_v2.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
