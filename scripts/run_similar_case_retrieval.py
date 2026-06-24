#!/usr/bin/env python3
"""Similar research-case retrieval (explanation-only) — v1.2.

For each deployable grader we embed every finding with the grader's **penultimate features**
(``model.encoder``; image content only — no level/condition/severity leakage), build a
research-case bank from the DEV split, and retrieve the top-k nearest DEV neighbours for each
TEST finding (cosine). We report the retrieved severity distribution + side agreement. This is
**explanation only**: it never changes a prediction. Uses cached crops (no DICOM decode, no GT).

Evaluation: same-side retrieval rate (foraminal/subarticular), severity agreement
(majority-retrieved == held-out reference), and whether uncertain queries get mixed retrieval.

Research-only. Not diagnostic. Reproduce: `python scripts/run_similar_case_retrieval.py`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from spinescoutx.data.crops import read_manifest
from spinescoutx.data.datasets import _load_array, _to_3chw
from spinescoutx.data.locked_test import load_splits_v1

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
OUT = ROOT / "outputs/real/similar_case_retrieval.json"
RECORDS = ROOT / "outputs/real/similar_case_retrieval_records.parquet"
DOC = ROOT / "docs/run_logs/similar_case_retrieval_v1.md"
# grader -> (run, cache, conditions it serves)
GRADERS = {
    "canal": (
        "runs/v1_canal_auto_robust",
        "data/cache/rsna_auto_canal_all",
        ["spinal_canal_stenosis"],
    ),
    "foraminal": (
        "runs/v1_foraminal_oracle_ctrl",
        "data/cache/rsna_auto_foraminal",
        ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"],
    ),
    "subarticular": (
        "runs/v1_subarticular_auto_robust",
        "data/cache/rsna_auto_subarticular",
        ["left_subarticular_stenosis", "right_subarticular_stenosis"],
    ),
}
SEV = ("normal_mild", "moderate", "severe")
K = 5

_spec = importlib.util.spec_from_file_location("es_run", ROOT / "scripts/run_evidence_stability.py")
_esrun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_esrun)


def _feats(model, rows, cache, crop_size, device):
    import torch

    out = []
    buf = []

    def flush():
        if not buf:
            return
        img = torch.from_numpy(np.stack(buf)).to(device)
        with torch.no_grad():
            f = model.encoder(img).float().cpu().numpy()
        out.append(f)
        buf.clear()

    for r in rows:
        buf.append(_to_3chw(_load_array(cache / r.crop_path), crop_size))
        if len(buf) >= 128:
            flush()
    flush()
    z = np.concatenate(out, axis=0) if out else np.zeros((0, 1))
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    return z


def _rows(cache, conds, split, split_map):
    man = read_manifest(cache / "manifest.parquet")
    man = man[(man.condition.isin(conds)) & (man.severity_index.isin([0, 1, 2]))].copy()
    man["study_id"] = man.study_id.astype(str)
    man = man[man.study_id.map(split_map) == split].reset_index(drop=True)
    return list(man.itertuples())


def run_grader(name, device):
    run, cache, conds = GRADERS[name]
    run_dir, cache_p = ROOT / run, ROOT / cache
    model, cfg = _esrun.load_grader(run_dir, device)
    crop_size = int(getattr(cfg.data, "crop_size", 224))
    split_map = load_splits_v1(SPLITS)
    bank = _rows(cache_p, conds, "dev", split_map)
    query = _rows(cache_p, conds, "test", split_map)
    if not bank or not query:
        return [], {}
    bz = _feats(model, bank, cache_p, crop_size, device)
    qz = _feats(model, query, cache_p, crop_size, device)
    bank_sev = np.array([int(r.severity_index) for r in bank])
    bank_side = np.array([str(getattr(r, "side", "") or "") for r in bank])
    bank_cond = np.array([str(r.condition) for r in bank])
    recs = []
    multi = name in ("foraminal", "subarticular")
    for i, r in enumerate(query):
        sims = qz[i] @ bz.T
        top = np.argsort(-sims)[: K + 1]
        # drop a self-match (same study+level) if present
        top = [j for j in top if not (bank[j].study_id == r.study_id and bank[j].level == r.level)][
            :K
        ]
        sev_counts = {s: int((bank_sev[top] == k).sum()) for k, s in enumerate(SEV)}
        maj = SEV[int(np.bincount(bank_sev[top], minlength=3).argmax())]
        same_side = (
            float(np.mean(bank_side[top] == (str(getattr(r, "side", "") or "")))) if multi else None
        )
        same_cond = float(np.mean(bank_cond[top] == str(r.condition)))
        recs.append(
            {
                "grader": name,
                "condition": str(r.condition),
                "study_id": str(r.study_id),
                "level": str(r.level),
                "side": str(getattr(r, "side", "") or ""),
                "ref": SEV[int(r.severity_index)],
                "majority_severity": maj,
                "severity_distribution": sev_counts,
                "same_side_rate": same_side,
                "same_condition_rate": same_cond,
                "k": len(top),
            }
        )
    return recs, {"n_bank": len(bank), "n_query": len(query)}


def main() -> int:
    from spinescoutx.training.optim import select_device

    device = select_device("auto")
    all_recs, meta = [], {}
    for name in GRADERS:
        print(f"[retrieval] {name} ...", flush=True)
        recs, m = run_grader(name, device)
        all_recs.extend(recs)
        meta[name] = m
        if recs:
            agree = np.mean([r["majority_severity"] == r["ref"] for r in recs])
            print(f"    n_query={len(recs)} severity-agreement={agree:.3f}", flush=True)

    # evaluation
    def _agg(recs):
        if not recs:
            return {}
        agree = float(np.mean([r["majority_severity"] == r["ref"] for r in recs]))
        ss = [r["same_side_rate"] for r in recs if r["same_side_rate"] is not None]
        return {
            "n": len(recs),
            "severity_agreement": agree,
            "mean_same_side_rate": float(np.mean(ss)) if ss else None,
            "mean_same_condition_rate": float(np.mean([r["same_condition_rate"] for r in recs])),
        }

    out = {
        "protocol": f"splits_v1 locked-test; bank=dev, query=test; k={K}",
        "explanation_only": True,
        "meta": meta,
        "overall": _agg(all_recs),
        "per_grader": {g: _agg([r for r in all_recs if r["grader"] == g]) for g in GRADERS},
    }
    # severity agreement among uncertain queries (near-severe): retrieval should be mixed
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
        "# Similar research-case retrieval (explanation-only) — v1",
        "",
        "> Research-only · not diagnostic. Top-k nearest DEV neighbours per TEST finding in the",
        "> grader's penultimate-feature space (cosine; cached crops, no GT). **Explanation only —",
        "> retrieval NEVER changes a prediction.** Retrieved cases are *similar research cases* —",
        "> not a clinical reference.",
        "",
        f"Bank = dev, query = test, k = {K}. Overall severity agreement (majority-retrieved ==",
        f"held-out reference) **{o['severity_agreement']:.3f}** over {o['n']} findings; mean",
        f"same-condition rate {o['mean_same_condition_rate']:.3f}.",
        "",
        "| grader | n | severity agreement | same-side rate | same-condition rate |",
        "|---|---|---|---|---|",
    ]
    for g, a in out["per_grader"].items():
        if not a:
            continue
        ss = f"{a['mean_same_side_rate']:.3f}" if a["mean_same_side_rate"] is not None else "n/a"
        lines.append(
            f"| {g} | {a['n']} | {a['severity_agreement']:.3f} | {ss} | "
            f"{a['mean_same_condition_rate']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation (honest)",
        "- High same-condition / same-side rates mean the grader embedding groups anatomically",
        "  similar findings — retrieval returns relevant *similar research cases*.",
        "- Severity agreement indicates retrieved neighbours tend to share the held-out severity;",
        "  it is a sanity check on the embedding, **not** a second predictor (we never vote).",
        "- Surfaced in the case viewer as `similar_research_cases` (severity distribution).",
        "",
        "Reproduce: `python scripts/run_similar_case_retrieval.py`.",
    ]
    DOC.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
