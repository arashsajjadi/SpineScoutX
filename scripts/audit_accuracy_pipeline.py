#!/usr/bin/env python3
"""On-real-data accuracy-integrity audit (v1.4).

Checks invariants that, if violated, would silently depress locked-test severe recall or bias
metrics. The scariest is **collect_probs alignment**: collect_probs returns (y_true, probs)
keyed by "study|level"; if that order desynchronizes from the manifest, every (label, prob)
pair is scrambled and recall is meaningless. We verify each returned y_true equals the
manifest severity for its key, recompute severe recall two independent ways, and check split
disjointness, auto-provenance (no hidden GT), and side consistency.

Prints PASS/FAIL per check + an overall verdict. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spinescoutx.constants import SEVERITY_TO_INDEX
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUT = ROOT / "outputs/real/accuracy_pipeline_audit.json"
ROUTES = {
    "spinal_canal_stenosis": ("runs/v1_canal_auto_robust", "data/cache/rsna_auto_canal_all"),
    "right_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "left_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
}


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    return {"check": name, "pass": bool(ok), "detail": detail}


def main() -> int:
    device = select_device("auto")
    split_map = load_splits_v1(SPLITS)
    results = []

    # 1) split disjointness + by-study
    splits = {"train": set(), "dev": set(), "test": set()}
    for s, sp in split_map.items():
        if sp in splits:
            splits[sp].add(s)
    inter = (
        (splits["train"] & splits["test"])
        | (splits["dev"] & splits["test"])
        | (splits["train"] & splits["dev"])
    )
    results.append(
        _check(
            "splits disjoint (no patient/study leakage)",
            len(inter) == 0,
            f"train {len(splits['train'])}/dev {len(splits['dev'])}/test {len(splits['test'])}; "
            f"overlap {len(inter)}",
        )
    )

    tmp = Path(
        "/tmp/claude-1000/-home-arash-PycharmProjects-SpineScoutX/"
        "ca508a4e-6a27-4c6a-a397-78976452a4e6/scratchpad/_audit.parquet"
    )
    tmp.parent.mkdir(parents=True, exist_ok=True)

    for cond, (run, cache) in ROUTES.items():
        run_dir, cpath = ROOT / run, ROOT / cache
        if not (run_dir / "best.pt").exists():
            continue
        man = read_manifest(cpath / "manifest.parquet")
        man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
        man["study_id"] = man.study_id.astype(str)
        te = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)

        # 2) auto provenance (no hidden GT in the deployed cache)
        prov = set(te.coordinate_source.unique()) if "coordinate_source" in te else {"?"}
        results.append(
            _check(
                f"[{cond}] auto provenance only (no oracle/GT in deployed crops)",
                prov == {"auto"},
                f"coordinate_source={prov}",
            )
        )

        # 3) collect_probs ALIGNMENT: returned y_true must equal the manifest severity per key
        te.to_parquet(tmp)
        preds = collect_probs(run_dir, tmp, cpath, device)
        # ground-truth severity per (study|level) from the manifest (independent of collect_probs)
        man_sev = {}
        for r in te.itertuples():
            man_sev[f"{r.study_id}|{r.level}"] = int(SEVERITY_TO_INDEX.get(str(r.severity), -1))
        mism = sum(1 for k, (y, _p) in preds.items() if k in man_sev and man_sev[k] != int(y))
        results.append(
            _check(
                f"[{cond}] collect_probs (y_true,probs) aligned to manifest labels",
                mism == 0,
                f"{mism}/{len(preds)} keys with y_true != manifest severity",
            )
        )

        # 4) severe recall recomputed two independent ways must match
        keys = sorted(preds)
        y = np.array([preds[k][0] for k in keys])
        p = np.stack([preds[k][1] for k in keys])
        manual = float((p[y == 2].argmax(1) == 2).mean()) if (y == 2).any() else float("nan")
        viabs = bs.m_severe_recall(y, p)
        results.append(
            _check(
                f"[{cond}] severe recall: manual == bootstrap metric",
                abs(manual - viabs) < 1e-9 or (np.isnan(manual) and np.isnan(viabs)),
                f"manual {manual:.4f} vs metric {viabs:.4f} (n_sev={int((y == 2).sum())})",
            )
        )

        # 5) side consistency: a side-qualified condition's crops all carry that side
        if cond.startswith(("left_", "right_")):
            side = "left" if cond.startswith("left_") else "right"
            sides = set(te.side.fillna("").astype(str).unique()) if "side" in te else set()
            results.append(
                _check(
                    f"[{cond}] crop side matches condition side",
                    sides <= {side},
                    f"sides in cache for this condition = {sides}",
                )
            )

    # 6) severe is class index 2 everywhere
    results.append(
        _check(
            "severe == class index 2", SEVERITY_TO_INDEX.get("severe") == 2, str(SEVERITY_TO_INDEX)
        )
    )

    n_fail = sum(1 for r in results if not r["pass"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"n_checks": len(results), "n_fail": n_fail, "checks": results}, indent=2)
    )
    print(f"\n=== {len(results) - n_fail}/{len(results)} PASS, {n_fail} FAIL ===")
    print(f"wrote {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
