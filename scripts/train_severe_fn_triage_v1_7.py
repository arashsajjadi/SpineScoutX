#!/usr/bin/env python3
"""Severe-FN triage / review-required fallback (v1.7, Phase 8).

Does NOT change the deployed grader. Fits a triage risk model on **dev** that predicts which
deployed-grader foraminal findings are likely severe false negatives (using cross-model
disagreement, the ensemble-minus-deployed p_severe gap, entropy, margin, p_normal_mild), then on
**locked-test (once)** routes the riskiest findings to review and measures: severe-FN capture at
5/10/15/20% review budgets, high-confidence severe-FN reduction, auto-finalised severe recall, and
review burden. A safety/triage upgrade only if these review metrics improve — never an accuracy
upgrade. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
OUTDIR = ROOT / "outputs/real"
TMP = ROOT / "outputs/real/_triage_tmp.parquet"
DEPLOYED = ROOT / "runs/v1_foraminal_oracle_ctrl"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]
V16_TAGS = ["rsna_baseline", "rsna_lss", "rsna_joint", "rsna_strong"]
BUDGETS = [0.05, 0.10, 0.15, 0.20]


def _deployed(split, sm, device):
    out = {}
    for cond in FORAMINAL:
        man = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
        man = man[(man.condition == cond) & man.severity_index.isin([0, 1, 2])].copy()
        man["study_id"] = man.study_id.astype(str)
        sub = man[man.study_id.map(sm) == split].reset_index(drop=True)
        if sub.empty:
            continue
        sub.to_parquet(TMP)
        for k, (y, p) in collect_probs(DEPLOYED, TMP, RSNA_CACHE, device).items():
            st, lv = k.split("|")[0], k.split("|")[1]
            out[f"{st}|{lv}|{cond}"] = (int(y), np.asarray(p, dtype=float))
    return out


def _v16_sev(tag, split):
    f = OUTDIR / f"foraminal_{tag}_{split}_preds.parquet"
    d = pd.read_parquet(f)
    return {str(r.key): float(r.p2) for r in d.itertuples()}


def _table(split, sm, device):
    dep = _deployed(split, sm, device)
    v16 = {t: _v16_sev(t, split) for t in V16_TAGS}
    rows = []
    for key, (y, p) in dep.items():
        sev = [v16[t].get(key, np.nan) for t in V16_TAGS]
        ens = float(np.nanmean(sev))
        ent = float(-(np.clip(p, 1e-8, 1) * np.log(np.clip(p, 1e-8, 1))).sum() / np.log(3))
        margin = float(np.sort(p)[-1] - np.sort(p)[-2])
        rows.append({
            "key": key, "y": y, "dep_pred": int(np.argmax(p)), "dep_p_severe": p[2],
            "dep_p_nm": p[0], "ens_p_severe": ens, "ens_minus_dep": ens - p[2],
            "disagreement": float(np.nanstd(sev)), "entropy": ent, "margin": margin,
            "is_severe_fn": int(y == 2 and np.argmax(p) != 2),
            "is_conf_severe_fn": int(y == 2 and np.argmax(p) == 0 and p[0] >= 0.5),
        })  # fmt: skip
    return pd.DataFrame(rows)


def main() -> int:
    from sklearn.linear_model import LogisticRegression

    device = select_device("auto")
    sm = load_splits_v1(SPLITS)
    feat = ["ens_minus_dep", "disagreement", "entropy", "dep_p_nm", "ens_p_severe"]
    dev = _table("dev", sm, device)
    test = _table("test", sm, device)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(dev[feat].fillna(0.0), dev["is_severe_fn"])
    test = test.assign(risk=clf.predict_proba(test[feat].fillna(0.0))[:, 1])

    n = len(test)
    n_sev = int((test.y == 2).sum())
    n_fn = int(test.is_severe_fn.sum())
    n_conf_fn = int(test.is_conf_severe_fn.sum())
    base_recall = float((test[(test.y == 2)].dep_pred == 2).mean())  # deployed argmax severe recall
    out = {
        "protocol": "triage fits on dev, evaluated on locked-test once; deployed grader unchanged",
        "n_test": n, "n_severe": n_sev, "n_severe_fn": n_fn, "n_high_conf_severe_fn": n_conf_fn,
        "deployed_argmax_severe_recall": base_recall, "budgets": {},
    }  # fmt: skip
    ranked = test.sort_values("risk", ascending=False).reset_index(drop=True)
    for b in BUDGETS:
        k = int(round(b * n))
        flagged = ranked.head(k)
        fn_captured = int(flagged.is_severe_fn.sum())
        conf_fn_captured = int(flagged.is_conf_severe_fn.sum())
        # cases NOT flagged are auto-finalised (review is assumed to catch the flagged severe)
        auto = ranked.iloc[k:]
        auto_sev = auto[auto.y == 2]
        # effective severe recall = (auto-correct severe + all flagged severe) / total severe
        eff = ((auto_sev.dep_pred == 2).sum() + (flagged.y == 2).sum()) / max(n_sev, 1)
        out["budgets"][f"{int(b * 100)}pct"] = {
            "review_burden": round(k / n, 4),
            "severe_fn_captured": fn_captured,
            "severe_fn_capture_rate": round(fn_captured / max(n_fn, 1), 4),
            "high_conf_severe_fn_captured": conf_fn_captured,
            "high_conf_capture_rate": round(conf_fn_captured / max(n_conf_fn, 1), 4),
            "effective_severe_recall": round(float(eff), 4),
        }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "severe_fn_triage_v1_7.json").write_text(json.dumps(out, indent=2, default=float))
    if TMP.exists():
        TMP.unlink()
    print(f"[triage] test n={n} severe={n_sev} severe-FN={n_fn} (high-conf {n_conf_fn}); "
          f"deployed argmax severe recall {base_recall:.3f}")  # fmt: skip
    for b, d in out["budgets"].items():
        print(
            f"  budget {b}: FN capture {d['severe_fn_capture_rate']:.2f} "
            f"({d['severe_fn_captured']}/{n_fn}; high-conf "
            f"{d['high_conf_severe_fn_captured']}/{n_conf_fn}) -> eff severe recall "
            f"{d['effective_severe_recall']:.3f}"
        )
    print(f"wrote {OUTDIR / 'severe_fn_triage_v1_7.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
