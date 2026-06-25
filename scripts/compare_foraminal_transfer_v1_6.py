#!/usr/bin/env python3
"""Paired comparison: ImageNet-init vs LSS-pretrained RSNA foraminal grader (v1.6, Plan A).

Aligns the two dev-selected graders' per-finding probs on identical RSNA locked-test findings
(``study|level|condition``) and reports, per foraminal side + macro, severe recall and
recall@FAR<=10% for each, with the **paired** cluster-bootstrap delta (LSS - ImageNet). Also prints
each vs the deployed-grader reference (R-for 0.660, L-for 0.788). Dev selected each model; this is
the single locked-test read. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spinescoutx.evaluation import bootstrap as bs

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
OUTDIR = ROOT / "outputs/real"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]
DEPLOYED_REF = {
    "left_neural_foraminal_narrowing": 0.788,
    "right_neural_foraminal_narrowing": 0.660,
}


def _load(tag, split="test"):
    df = pd.read_parquet(OUTDIR / f"foraminal_{tag}_{split}_preds.parquet")
    out = {}
    for r in df.itertuples():
        out[str(r.key)] = (int(r.y), np.array([r.p0, r.p1, r.p2], dtype=float))
    return out


def _far(y, p):
    neg = y != 2
    return float((p[neg].argmax(1) == 2).mean()) if neg.any() else float("nan")


def _metrics(y, p, st):
    r10 = bs.make_recall_at_far(0.10)
    return {
        "severe_recall": float(bs.m_severe_recall(y, p)),
        "ci": bs.bootstrap_ci(y, p, st, bs.m_severe_recall, n_boot=2000),
        "recall_at_far10": float(r10(y, p)),
        "far": _far(y, p),
        "n": int(len(y)),
        "n_severe": int((y == 2).sum()),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-tag", default="rsna_baseline")
    ap.add_argument("--exp-tag", default="rsna_lss")
    ap.add_argument("--out", default="compare_foraminal_transfer_v1_6.json")
    args = ap.parse_args()
    base, lss = _load(args.baseline_tag), _load(args.exp_tag)
    keys = sorted(set(base) & set(lss))
    out = {"protocol": "paired ImageNet-init vs LSS-init RSNA foraminal grader, locked-test once",
           "n_common": len(keys), "per_condition": {}}  # fmt: skip
    macro = {"baseline": [], "lss": []}
    for cond in FORAMINAL:
        ck = [k for k in keys if k.endswith(f"|{cond}")]
        if not ck:
            continue
        y = np.array([base[k][0] for k in ck])
        st = np.array([k.split("|")[0] for k in ck])
        pb = np.stack([base[k][1] for k in ck])
        pl = np.stack([lss[k][1] for k in ck])
        r10 = bs.make_recall_at_far(0.10)
        d_sr = bs.paired_bootstrap_delta(y, pl, pb, st, bs.m_severe_recall, n_boot=2000)
        d_r10 = bs.paired_bootstrap_delta(y, pl, pb, st, r10, n_boot=2000)
        out["per_condition"][cond] = {
            "baseline": _metrics(y, pb, st),
            "lss": _metrics(y, pl, st),
            "paired_delta_severe_recall": d_sr,
            "paired_delta_recall_at_far10": d_r10,
            "deployed_ref_severe_recall": DEPLOYED_REF[cond],
        }
        macro["baseline"].append(out["per_condition"][cond]["baseline"]["severe_recall"])
        macro["lss"].append(out["per_condition"][cond]["lss"]["severe_recall"])
    out["foraminal_macro_severe_recall"] = {
        "baseline": float(np.mean(macro["baseline"])),
        "lss": float(np.mean(macro["lss"])),
        "delta": float(np.mean(macro["lss"]) - np.mean(macro["baseline"])),
        "deployed_ref": float(np.mean(list(DEPLOYED_REF.values()))),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / args.out).write_text(json.dumps(out, indent=2, default=float))
    for cond, r in out["per_condition"].items():
        b, t = r["baseline"], r["lss"]
        ds, dr = r["paired_delta_severe_recall"], r["paired_delta_recall_at_far10"]
        dec_s = " DECISIVE" if ds["decisive"] else ""
        dec_r = " DECISIVE" if dr["decisive"] else ""
        print(
            f"[{cond.split('_')[0]:5s}] severe recall base {b['severe_recall']:.3f} -> "
            f"LSS {t['severe_recall']:.3f} "
            f"(Δ{ds['delta']:+.3f} [{ds['ci_lo']:+.3f},{ds['ci_hi']:+.3f}]{dec_s}; "
            f"deployed {r['deployed_ref_severe_recall']:.3f})"
        )
        print(
            f"        recall@FAR10 base {b['recall_at_far10']:.3f} -> "
            f"LSS {t['recall_at_far10']:.3f} "
            f"(Δ{dr['delta']:+.3f} [{dr['ci_lo']:+.3f},{dr['ci_hi']:+.3f}]{dec_r}) | "
            f"FAR base {b['far']:.3f} LSS {t['far']:.3f}"
        )
    m = out["foraminal_macro_severe_recall"]
    print(
        f"[macro] severe recall base {m['baseline']:.3f} -> LSS {m['lss']:.3f} (Δ{m['delta']:+.3f})"
    )
    print(f"wrote {OUTDIR / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
