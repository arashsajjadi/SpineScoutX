"""Segmentation-morphometry fusion grader (v1.8b Phase 10).

Late-fuses the deployed image grader with a morphometry severity model:
``p_fused = (1-a)·p_deployed + a·p_morphometry`` (3-class), with the morphometry model (GBM) trained
on **train** features and the blend weight ``a`` swept on **dev** (maximize right-foraminal
recall@FAR≤10 with a FAR guardrail). The dev-selected fusion is read on **locked-test once** and
compared, paired, to the deployed baseline. Also reports a morphometry-MLP fusion variant.
Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.features.morphometry import FEATURE_COLS
from spinescoutx.training.optim import select_device

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
FEATS = ROOT / "data/cache/v1_8c_medsam2_morphometry/features.parquet"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
DEPLOYED = ROOT / "runs/v1_foraminal_oracle_ctrl"
OUT = ROOT / "outputs/real/v1_8c_real_medsam2_fusion.json"
TMP = ROOT / "outputs/real/_fusion_tmp.parquet"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]
RIGHT = "right_neural_foraminal_narrowing"
ALPHAS = [round(a, 2) for a in np.linspace(0, 0.8, 17)]


def _deployed_probs(split, sm, device):
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
    return out


def _rfor_r_at_far(keys, y, p):
    m = np.array([k.endswith(RIGHT) for k in keys])
    if m.sum() < 5 or not (y[m] == 2).any():
        return 0.0
    return float(bs.make_recall_at_far(0.10)(y[m], p[m]))


def _far(y, p):
    neg = y != 2
    return float((p[neg].argmax(1) == 2).mean()) if neg.any() else float("nan")


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    device = select_device("auto")
    sm = load_splits_v1(SPLITS)
    feats = pd.read_parquet(FEATS).set_index("key")
    dep = {s: _deployed_probs(s, sm, device) for s in ("train", "dev", "test")}

    def assemble(split):
        keys = [k for k in dep[split] if k in feats.index]
        y = np.array([dep[split][k][0] for k in keys])
        pdep = np.stack([dep[split][k][1] for k in keys])
        X = feats.loc[keys, FEATURE_COLS].to_numpy()
        return keys, y, pdep, X

    ktr, ytr, ptr, Xtr = assemble("train")
    kdv, ydv, pdv, Xdv = assemble("dev")
    kte, yte, pte, Xte = assemble("test")

    scaler = StandardScaler().fit(Xtr)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    gbm.fit(scaler.transform(Xtr), ytr)  # 3-class morphometry model

    def morph_p(X):
        pr = gbm.predict_proba(scaler.transform(X))
        full = np.zeros((len(X), 3))
        full[:, gbm.classes_] = pr
        return full

    pm_dv, pm_te = morph_p(Xdv), morph_p(Xte)

    # sweep alpha on dev (maximize R-for recall@FAR10, reject FAR>0.5)
    sweep = []
    for a in ALPHAS:
        pf = (1 - a) * pdv + a * pm_dv
        pf = pf / pf.sum(1, keepdims=True)
        far = _far(ydv, pf)
        sel = _rfor_r_at_far(kdv, ydv, pf) if (np.isnan(far) or far <= 0.5) else -1.0
        sweep.append({"alpha": a, "dev_rfor_r_at_far10": sel, "dev_far": far})
    best = max(sweep, key=lambda s: s["dev_rfor_r_at_far10"])
    a = best["alpha"]

    # locked-test once at best alpha
    pf_te = (1 - a) * pte + a * pm_te
    pf_te = pf_te / pf_te.sum(1, keepdims=True)
    sides = np.array([k.split("|")[-1] for k in kte])
    st = np.array([k.split("|")[0] for k in kte])
    out = {"selected_alpha": a, "dev_sweep_best": best, "n_test": len(kte), "test": {}}
    for cond in FORAMINAL:
        msk = sides == cond
        d_sr = bs.paired_bootstrap_delta(
            yte[msk], pf_te[msk], pte[msk], st[msk], bs.m_severe_recall, n_boot=2000
        )
        out["test"][cond] = {
            "baseline_severe_recall": float(bs.m_severe_recall(yte[msk], pte[msk])),
            "fusion_severe_recall": float(bs.m_severe_recall(yte[msk], pf_te[msk])),
            "paired_delta": d_sr,
            "baseline_recall_at_far10": float(bs.make_recall_at_far(0.10)(yte[msk], pte[msk])),
            "fusion_recall_at_far10": float(bs.make_recall_at_far(0.10)(yte[msk], pf_te[msk])),
            "n_severe": int((yte[msk] == 2).sum()),
        }
    out["test"]["foraminal_macro_baseline"] = float(
        np.mean([out["test"][c]["baseline_severe_recall"] for c in FORAMINAL])
    )
    out["test"]["foraminal_macro_fusion"] = float(
        np.mean([out["test"][c]["fusion_severe_recall"] for c in FORAMINAL])
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    if TMP.exists():
        TMP.unlink()
    print(f"[fusion] selected alpha={a} (dev R-for r@FAR10 {best['dev_rfor_r_at_far10']:.3f})")
    for cond in FORAMINAL:
        d = out["test"][cond]
        delta = d["paired_delta"]
        print(
            f"  TEST {cond.split('_')[0]:5s}: severe recall base {d['baseline_severe_recall']:.3f} "
            f"-> fusion {d['fusion_severe_recall']:.3f} "
            f"(Δ{delta['delta']:+.3f} [{delta['ci_lo']:+.3f},{delta['ci_hi']:+.3f}]"
            f"{' DECISIVE' if delta['decisive'] else ''})"
        )
    print(
        f"  macro base {out['test']['foraminal_macro_baseline']:.3f} -> "
        f"fusion {out['test']['foraminal_macro_fusion']:.3f}"
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
