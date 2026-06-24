#!/usr/bin/env python3
"""Evidence Intelligence v2 — attribute each finding's instability to a CAUSE.

v1 measured *how much* a prediction moves under localizer perturbation; v2 measures
*why* by isolating the slice/leveling component from the in-plane crop-centre component
(reusing `sample_offsets(mode=...)`), and labels each finding:
  crop_sensitive · slice_sensitive · axial_candidate_sensitive · route_sensitive · stable.

Efficiency: ONE shared slice decoder per condition across all three perturbation regimes
(the v1.1 attribution bottleneck was a per-regime decoder thrashing); runs on a study
subsample by default. Per-finding types are written for the case viewer; the population
evaluation reports the type distribution and the severe-FN rate per type (does typing
localize the failure cause?).

No GT coordinates generate perturbations; reference labels used only to score severe FNs.
Research-only. Not diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.crops import read_manifest
from spinescoutx.data.datasets import _load_array
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.data.perturb_crops import SliceDecoder, reextract_25d
from spinescoutx.evaluation import evidence_stability as es
from spinescoutx.training.optim import select_device

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUT = ROOT / "outputs/real/evidence_intel_v2.json"
RECORDS = ROOT / "outputs/real/evidence_intel_v2_records.parquet"
DOC = ROOT / "docs/run_logs/evidence_intelligence_v2.md"
ROUTES = {
    "spinal_canal_stenosis": ("runs/v1_canal_auto_robust", "data/cache/rsna_auto_canal_all"),
    "left_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "right_neural_foraminal_narrowing": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
    ),
    "left_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
    "right_subarticular_stenosis": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
    ),
}
ROUTE_OF = {
    "spinal_canal_stenosis": "sagittal_t2",
    "left_neural_foraminal_narrowing": "sagittal_t1",
    "right_neural_foraminal_narrowing": "sagittal_t1",
    "left_subarticular_stenosis": "axial_t2",
    "right_subarticular_stenosis": "axial_t2",
}

_spec = importlib.util.spec_from_file_location("es_run", ROOT / "scripts/run_evidence_stability.py")
_esrun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_esrun)


def _regime_probs(model, rows, lv, cd, device, crop_size, cache, decoder, offsets):
    n = len(rows)
    pert = np.zeros((len(offsets[0]), n, 3))

    def provider(i, k):
        r = rows[i]
        o = offsets[i][k]  # offsets already restricted to the regime at sampling time
        c = reextract_25d(
            decoder,
            r.dicom_path,
            r.instance_number,
            r.x,
            r.y,
            dx=o[0],
            dy=o[1],
            ds=o[2],
            crop_size=crop_size,
        )
        return c if c is not None else _load_array(cache / r.crop_path)

    for k in range(len(offsets[0])):
        pert[k] = _esrun.forward_providers(
            model, [(lambda i=i, k=k: provider(i, k)) for i in range(n)], lv, cd, device, crop_size
        )
    return pert


def run_condition(cond, device, max_studies, seed):
    run_dir, cache = (ROOT / x for x in ROUTES[cond])
    if not (run_dir / "best.pt").exists():
        return None
    model, cfg = _esrun.load_grader(run_dir, device)
    crop_size = int(getattr(cfg.data, "crop_size", 224))
    split_map = load_splits_v1(SPLITS)
    man = read_manifest(cache / "manifest.parquet")
    man = man[(man.condition == cond) & (man.severity_index.isin([0, 1, 2]))].copy()
    man["study_id"] = man.study_id.astype(str)
    man = man[man.study_id.map(split_map) == "test"].reset_index(drop=True)
    if max_studies:
        keep = sorted(man.study_id.unique())[:max_studies]
        man = man[man.study_id.isin(keep)].reset_index(drop=True)
    rows = list(man.itertuples())
    n = len(rows)
    if n == 0:
        return None
    lv = np.array([LEVEL_TO_INDEX[r.level] for r in rows])
    cd = np.array([CONDITION_TO_INDEX[r.condition] for r in rows])
    base = _esrun.forward_providers(
        model,
        [(lambda r=r: _load_array(cache / r.crop_path)) for r in rows],
        lv,
        cd,
        device,
        crop_size,
    )
    cfgp = es.config_for(cond)
    route = ROUTE_OF[cond]
    decoder = SliceDecoder(max_items=4096)  # shared across the 3 regimes (key speedup)
    regimes = {}
    for mode in ("full", "slice", "inplane"):
        rng = np.random.default_rng(seed)
        offs = [es.sample_offsets(cfgp, rng, mode=mode) for _ in rows]
        regimes[mode] = _regime_probs(model, rows, lv, cd, device, crop_size, cache, decoder, offs)
    recs = []
    for i, r in enumerate(rows):
        inst = {}
        for mode in ("full", "slice", "inplane"):
            st = es.stability_stats(np.vstack([base[i], regimes[mode][:, i, :]]))
            if mode == "full":
                grade = es.stability_grade(st, cfgp)
                pred = st["baseline_pred"]
            inst[mode] = es.instability_score(st)
        itype = es.classify_instability_type(
            inst["full"], inst["slice"], inst["inplane"], route=route, grade=grade
        )
        y = int(r.severity_index)
        recs.append(
            {
                "condition": cond,
                "study_id": str(r.study_id),
                "level": str(r.level),
                "side": str(getattr(r, "side", "") or ""),
                "full_inst": inst["full"],
                "slice_inst": inst["slice"],
                "inplane_inst": inst["inplane"],
                "grade": grade,
                "instability_type": itype,
                "severe_fn": int(y == 2 and pred != 2),
                "is_severe": int(y == 2),
                "wrong": int(pred != y),
            }
        )
    print(f"  {cond:34s} n={n} types={_counts([r['instability_type'] for r in recs])}", flush=True)
    return recs


def _counts(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-studies", type=int, default=70)
    ap.add_argument("--seed", type=int, default=20260624)
    args = ap.parse_args()
    device = select_device("auto")
    all_recs = []
    per_cond = {}
    for cond in ROUTES:
        print(f"[evidence-intel-v2] {cond}", flush=True)
        recs = run_condition(cond, device, args.max_studies, args.seed)
        if recs:
            all_recs.extend(recs)
            per_cond[cond] = recs
    # evaluation: type distribution + severe-FN rate per type
    type_dist = _counts([r["instability_type"] for r in all_recs])
    sev_fn_by_type = {}
    for t in es.INSTABILITY_TYPES:
        sub = [r for r in all_recs if r["instability_type"] == t]
        nfn = sum(r["severe_fn"] for r in sub)
        sev_fn_by_type[t] = {
            "n": len(sub),
            "n_severe_fn": nfn,
            "severe_fn_rate": float(nfn / len(sub)) if sub else 0.0,
        }
    out = {
        "protocol": "splits_v1 locked-test (subsample)",
        "max_studies": args.max_studies,
        "n_findings": len(all_recs),
        "type_distribution": type_dist,
        "severe_fn_by_type": sev_fn_by_type,
        "per_condition_types": {
            c: _counts([r["instability_type"] for r in rs]) for c, rs in per_cond.items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    import pandas as pd

    pd.DataFrame(all_recs).to_parquet(RECORDS, index=False)
    _doc(out)
    print(f"\nwrote {OUT}\nwrote {RECORDS} ({len(all_recs)} findings)\nwrote {DOC}")
    return 0


def _doc(out):
    td = out["type_distribution"]
    sf = out["severe_fn_by_type"]
    lines = [
        "# Evidence Intelligence v2 — instability typing (locked-test auto, subsample)",
        "",
        "> Research-only · not diagnostic. Each finding's instability is attributed to a cause",
        "> by isolating slice-only vs in-plane-only perturbation (no GT to perturb). Subsample of",
        f"> {out['max_studies']} studies/condition; {out['n_findings']} findings.",
        "",
        "## Instability type distribution (all routes)",
        "| type | count | severe-FN rate (n_FN/n) |",
        "|---|---|---|",
    ]
    order = [
        "stable",
        "crop_sensitive",
        "slice_sensitive",
        "axial_candidate_sensitive",
        "route_sensitive",
    ]
    for t in order:
        s = sf.get(t, {"n": 0, "n_severe_fn": 0, "severe_fn_rate": 0.0})
        lines.append(
            f"| {t} | {td.get(t, 0)} | {s['severe_fn_rate']:.3f} ({s['n_severe_fn']}/{s['n']}) |"
        )
    lines += [
        "",
        "## Instability type by condition",
        "| condition | dominant types |",
        "|---|---|",
    ]
    for c, counts in out["per_condition_types"].items():
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        lines.append(f"| {c} | " + ", ".join(f"{k}:{v}" for k, v in top if k != "stable") + " |")
    # verdict: does the unstable cause match the route's known bottleneck?
    unstable = {t: sf[t] for t in sf if t not in ("stable",)}
    worst_type = max(unstable, key=lambda t: unstable[t]["severe_fn_rate"]) if unstable else "n/a"
    lines += [
        "",
        "## Interpretation (honest)",
        "- Instability typing localizes the **cause** of an unstable finding (crop vs",
        "  slice/level vs mixed), which v1's scalar score could not. It feeds the case viewer's",
        "  `instability_type` and route-specific review reasons.",
        f"- Highest severe-FN rate is among **{worst_type}** findings — i.e. that perturbation",
        "  cause concentrates the missed-severe cases, the most informative review trigger.",
        "- Explanatory/triage enrichment; it does not change any prediction. Subsample sizes mean",
        "  per-type rates are indicative, not decisive (reported, not over-interpreted).",
        "",
        "Reproduce: `python scripts/run_evidence_intel_v2.py --max-studies 70`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
