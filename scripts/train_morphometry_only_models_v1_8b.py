#!/usr/bin/env python3
"""Morphometry-only severity signal check (v1.8b Phase 9) — the pivotal go/no-go.

Before any fusion: does the SAM2.1-derived foraminal morphometry contain severity signal at all?
Fit simple models (logistic / gradient boosting) on **train** morphometry features and measure
**dev** severe-vs-nonsevere AUROC + severe recall@FAR≤10 + feature importance, per foraminal side.
Compared to the deployed image grader's dev signal. If morphometry is at chance, fusion cannot help.
Dev only; locked-test untouched. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.features.morphometry import FEATURE_COLS

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
FEATS = ROOT / "data/cache/v1_8b_morphometry/features.parquet"
OUT = ROOT / "outputs/real/v1_8b_morphometry_only.json"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]


def _severe_recall_at_far(y_sev, score, far=0.10):
    """y_sev binary (1=severe); score = predicted severe prob -> recall at FAR<=far."""
    order = np.argsort(-score)
    y = y_sev[order]
    neg_total = (y == 0).sum()
    pos_total = (y == 1).sum()
    fp = tp = 0
    best = 0.0
    for yi in y:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        if neg_total and fp / neg_total <= far:
            best = max(best, tp / max(pos_total, 1))
    return float(best)


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(FEATS)
    out = {"n": int(len(df)), "seg_fail_rate": round(float(df.m_seg_fail.mean()), 4), "routes": {}}
    for cond in FORAMINAL:
        sub = df[df.condition == cond]
        tr = sub[sub.split == "train"]
        dv = sub[sub.split == "dev"]
        if len(tr) < 50 or (dv.severity_index == 2).sum() < 3:
            continue
        Xtr = StandardScaler().fit(tr[FEATURE_COLS])
        xtr, xdv = Xtr.transform(tr[FEATURE_COLS]), Xtr.transform(dv[FEATURE_COLS])
        ytr = (tr.severity_index == 2).to_numpy().astype(int)
        ydv = (dv.severity_index == 2).to_numpy().astype(int)
        res = {"n_train": int(len(tr)), "n_dev_severe": int(ydv.sum())}
        for name, clf in [
            ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("gbm", GradientBoostingClassifier(n_estimators=150, max_depth=3)),
        ]:
            clf.fit(xtr, ytr)
            sc = clf.predict_proba(xdv)[:, 1]
            res[name] = {
                "dev_auroc": round(float(roc_auc_score(ydv, sc)), 3) if ydv.sum() else None,
                "dev_severe_recall_at_far10": round(_severe_recall_at_far(ydv, sc), 3),
            }
        # univariate: which morphometry feature separates severe best (|AUROC-0.5|)
        imp = {}
        for c in FEATURE_COLS:
            if dv[c].std() > 0 and ydv.sum():
                imp[c] = round(abs(roc_auc_score(ydv, dv[c]) - 0.5), 3)
        res["top_features"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:5])
        out["routes"][cond] = res
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2))
    # verdict
    aurocs = [
        r["logistic"]["dev_auroc"]
        for r in out["routes"].values()
        if r.get("logistic", {}).get("dev_auroc")
    ]
    signal = any(a and a >= 0.58 for a in aurocs)
    print(f"\nMORPHOMETRY SIGNAL (dev AUROC>=0.58 on any foraminal route): {signal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
