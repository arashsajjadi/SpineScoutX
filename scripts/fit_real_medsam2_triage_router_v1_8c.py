"""Morphometry-informed severe-FN triage router (v1.8b Phase 11).

Does the SAM2.1 morphometry add severe-FN capture beyond a deployed-only triage? Fits a severe-FN
risk model on **dev** two ways — (a) deployed-grader signals only (entropy, p_nm, margin) and
(b) + morphometry (contrast, min-opening, area, aspect, seg-confidence) — then on **locked-test
once** measures severe-FN capture + effective severe recall at 5/10/15/20% review budgets. The
deployed grader is unchanged; raw argmax is never overridden. Safety/triage only. Research-only.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
FEATS = ROOT / "data/cache/v1_8c_medsam2_morphometry/features.parquet"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
DEPLOYED = ROOT / "runs/v1_foraminal_oracle_ctrl"
OUT = ROOT / "outputs/real/v1_8c_real_medsam2_triage.json"
TMP = ROOT / "outputs/real/_mtriage_tmp.parquet"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]
MORPH = ["m_contrast", "m_min_open", "m_area_frac", "m_aspect", "m_iou_conf", "m_intensity_mean"]
BUDGETS = [0.05, 0.10, 0.15, 0.20]


def _table(split, sm, device, feats):
    out = {}
    for cond in FORAMINAL:
        man = read_manifest(RSNA_CACHE / "manifest.parquet")
        man = man[(man.condition == cond) & man.severity_index.isin([0, 1, 2])].copy()
        man["study_id"] = man.study_id.astype(str)
        sub = man[man.study_id.map(sm) == split].reset_index(drop=True)
        if sub.empty:
            continue
        sub.to_parquet(TMP)
        for k, (y, p) in collect_probs(DEPLOYED, TMP, RSNA_CACHE, device).items():
            st, lv = k.split("|")[0], k.split("|")[1]
            out[f"{st}|{lv}|{cond}"] = (int(y), np.asarray(p, float))
    rows = []
    for key, (y, p) in out.items():
        if key not in feats.index:
            continue
        ent = float(-(np.clip(p, 1e-8, 1) * np.log(np.clip(p, 1e-8, 1))).sum() / np.log(3))
        margin = float(np.sort(p)[-1] - np.sort(p)[-2])
        rec = {"key": key, "y": y, "dep_pred": int(np.argmax(p)), "dep_p_nm": p[0],
               "entropy": ent, "margin": margin,
               "is_severe_fn": int(y == 2 and np.argmax(p) != 2)}  # fmt: skip
        for m in MORPH:
            rec[m] = float(feats.loc[key, m])
        rows.append(rec)
    return pd.DataFrame(rows)


def _budget_metrics(test, risk_col, n_sev, n_fn):
    ranked = test.sort_values(risk_col, ascending=False).reset_index(drop=True)
    res = {}
    for b in BUDGETS:
        k = int(round(b * len(test)))
        flagged, auto = ranked.head(k), ranked.iloc[k:]
        eff = ((auto[auto.y == 2].dep_pred == 2).sum() + (flagged.y == 2).sum()) / max(n_sev, 1)
        res[f"{int(b * 100)}pct"] = {
            "severe_fn_capture_rate": round(int(flagged.is_severe_fn.sum()) / max(n_fn, 1), 4),
            "effective_severe_recall": round(float(eff), 4),
        }
    return res


def main() -> int:
    from sklearn.linear_model import LogisticRegression

    device = select_device("auto")
    sm = load_splits_v1(SPLITS)
    feats = pd.read_parquet(FEATS).set_index("key")
    dev = _table("dev", sm, device, feats)
    test = _table("test", sm, device, feats)
    n_sev, n_fn = int((test.y == 2).sum()), int(test.is_severe_fn.sum())
    base_dep = ["entropy", "dep_p_nm", "margin"]
    dep_recall = round(float((test[test.y == 2].dep_pred == 2).mean()), 4)
    out = {
        "n_test": len(test), "n_severe": n_sev, "n_severe_fn": n_fn,
        "deployed_argmax_severe_recall": dep_recall, "variants": {},
    }  # fmt: skip
    for name, cols in [
        ("deployed_only", base_dep),
        ("deployed_plus_morphometry", base_dep + MORPH),
    ]:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(dev[cols].fillna(0.0), dev["is_severe_fn"])
        test = test.assign(_risk=clf.predict_proba(test[cols].fillna(0.0))[:, 1])
        out["variants"][name] = _budget_metrics(test, "_risk", n_sev, n_fn)
    if TMP.exists():
        TMP.unlink()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"[m-triage] test severe={n_sev} severe-FN={n_fn} "
          f"deployed severe recall {out['deployed_argmax_severe_recall']:.3f}")  # fmt: skip
    for name, v in out["variants"].items():
        print(f"  {name}:")
        for b, d in v.items():
            print(f"    {b}: FN capture {d['severe_fn_capture_rate']:.2f} -> "
                  f"eff severe recall {d['effective_severe_recall']:.3f}")  # fmt: skip
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
