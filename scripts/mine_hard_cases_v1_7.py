#!/usr/bin/env python3
"""Mine v1.7 hard-case candidates for label repair.

Builds a per-finding foraminal table (true RSNA severity + deployed grader probs + several v1.6/
v1.5 models' p_severe) on **train+dev** (cleanable) plus a deployed reference on test, then mines
the severe-FN / confidently-normal / borderline / disagreement / uncertainty groups + controls.
Writes a gitignored candidates JSON and a pixel-free committed summary (IDs/counts only). No
locked-test labels are used for any cleaning decision. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spinescoutx.data.hard_case_mining import build_signal_table, mine_groups, select_review_set
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation.gap_decomposition import collect_probs
from spinescoutx.models.image_classifier import ImageClassifier
from spinescoutx.training.optim import select_device

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
OUTDIR = ROOT / "outputs/real"
TMP = ROOT / "outputs/real/_mine_tmp.parquet"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]
DEPLOYED = ROOT / "runs/v1_foraminal_oracle_ctrl"
V16_MODELS = {  # tag -> (run_dir, backbone)
    "v16_baseline": ("runs/foraminal_rsna_baseline_v1_6", "convnext_tiny"),
    "v16_lss": ("runs/foraminal_rsna_lss_v1_6", "convnext_tiny"),
    "v16_joint": ("runs/foraminal_rsna_joint_v1_6", "convnext_tiny"),
    "v16_strong": ("runs/foraminal_rsna_strong_v1_6", "convnext_small"),
}
CAPS = {  # per-bucket caps for the review set (deduped, priority-ranked)
    "A_severe_fn": 300,
    "B_confident_normal_severe_miss": 120,
    "C_moderate_severe_borderline": 200,
    "D_model_disagreement": 150,
    "F_high_uncertainty": 120,
    "G_control_correct_severe": 60,
    "G_control_correct_nonsevere": 60,
    "G_control_random_easy": 60,
}


def _foraminal_manifest():
    m = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
    m = m[m.condition.isin(FORAMINAL) & m.severity_index.isin([0, 1, 2])].copy()
    m["study_id"] = m.study_id.astype(str)
    m["key"] = m.study_id + "|" + m.level.astype(str) + "|" + m.condition
    m["split"] = m.study_id.map(load_splits_v1(SPLITS))
    return m


def deployed_probs(split, splits_keep):
    """Deployed grader p[3] per key on the given split (collect_probs, per condition)."""
    out = {}
    for cond in FORAMINAL:
        man = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
        man = man[(man.condition == cond) & man.severity_index.isin([0, 1, 2])].copy()
        man["study_id"] = man.study_id.astype(str)
        sub = man[man.study_id.map(splits_keep) == split].reset_index(drop=True)
        if sub.empty:
            continue
        sub.to_parquet(TMP)
        preds = collect_probs(DEPLOYED, TMP, RSNA_CACHE, select_device("auto"))
        for k, (y, p) in preds.items():
            st, lv = k.split("|")[0], k.split("|")[1]
            out[f"{st}|{lv}|{cond}"] = (int(y), np.asarray(p, dtype=float))
    return out


@torch.no_grad()
def infer_p_severe(run_dir, backbone, df, device):
    """p_severe per key for an ImageClassifier checkpoint over df's crops."""
    model = ImageClassifier(
        backbone=backbone, in_chans=3, num_classes=3, use_level_embedding=True,
        use_condition_embedding=True, embed_dim=16, dropout=0.2, pretrained=False,
    ).to(device).eval()  # fmt: skip
    sd = torch.load(ROOT / run_dir / "best.pt", map_location="cpu")["state_dict"]
    model.load_state_dict(sd)
    from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX

    out, keys, batch = {}, [], []
    rows = df.reset_index(drop=True)
    for r in rows.itertuples():
        arr = np.load(RSNA_CACHE / r.crop_path).astype(np.float32)
        batch.append(arr)
        keys.append(r.key)
        if len(batch) == 256:
            out.update(_flush(model, batch, keys, rows, device, LEVEL_TO_INDEX, CONDITION_TO_INDEX))
            batch, keys = [], []
    if batch:
        out.update(_flush(model, batch, keys, rows, device, LEVEL_TO_INDEX, CONDITION_TO_INDEX))
    return out


def _flush(model, batch, keys, rows, device, lvl_idx, cond_idx):
    idx = {r.key: r for r in rows.itertuples()}
    img = torch.from_numpy(np.stack(batch)).to(device)
    lv = torch.tensor([lvl_idx[idx[k].level] for k in keys], device=device)
    cd = torch.tensor([cond_idx[idx[k].condition] for k in keys], device=device)
    p = torch.softmax(model(img, lv, cd).float(), 1).cpu().numpy()
    return {k: float(p[i, 2]) for i, k in enumerate(keys)}


def _load_pred_parquet(tag, split):
    f = OUTDIR / f"foraminal_{tag.replace('v16_', 'rsna_')}_{split}_preds.parquet"
    if not f.exists():
        return {}
    d = pd.read_parquet(f)
    return {str(r.key): float(r.p2) for r in d.itertuples()}


def build_table(splits):
    sm = load_splits_v1(SPLITS)
    man = _foraminal_manifest()
    device = select_device("auto")
    rows = []
    for split in splits:
        sub = man[man.split == split]
        dep = deployed_probs(split, sm)
        model_sev = {}
        for tag, (run_dir, backbone) in V16_MODELS.items():
            cached = _load_pred_parquet(tag, split)
            if cached:
                model_sev[tag] = cached
            else:  # train split has no cached preds -> infer
                print(f"  inferring {tag} on {split} ({len(sub)} crops)...", flush=True)
                model_sev[tag] = infer_p_severe(run_dir, backbone, sub, device)
        for r in sub.itertuples():
            if r.key not in dep:
                continue
            y, p = dep[r.key]
            rec = {
                "key": r.key, "study_id": r.study_id, "level": r.level, "condition": r.condition,
                "side": r.side, "split": split, "severity_index": int(r.severity_index),
                "dep_p0": p[0], "dep_p1": p[1], "dep_p2": p[2],
            }  # fmt: skip
            for tag in V16_MODELS:
                rec[f"p_severe_{tag}"] = model_sev[tag].get(r.key, np.nan)
            rows.append(rec)
        print(f"  [{split}] {len(sub)} findings, {sum(r['split'] == split for r in rows)} kept",
              flush=True)  # fmt: skip
    return pd.DataFrame(rows)


def main() -> int:
    print("[mine] building multi-model prediction table (train+dev) ...", flush=True)
    df = build_table(["train", "dev"])
    model_cols = [f"p_severe_{t}" for t in V16_MODELS]
    sig = build_signal_table(df, model_cols)
    groups = mine_groups(sig)
    review = select_review_set(sig, groups, caps=CAPS)

    def _counts(d):
        sub = d[d.condition == "right_neural_foraminal_narrowing"]
        return {
            "total": int(len(d)),
            "right_foraminal": int(len(sub)),
            "left_foraminal": int((d.condition == "left_neural_foraminal_narrowing").sum()),
            "by_level": {str(k): int(v) for k, v in sorted(Counter(d.level).items())},
        }

    summary = {
        "protocol": "splits_v1 train+dev (cleanable); test never used for cleaning",
        "n_findings": int(len(sig)),
        "models": ["deployed", *V16_MODELS.keys()],
        "groups": {name: _counts(g) for name, g in groups.items()},
        "review_set": {
            "n": int(len(review)),
            "by_condition": dict(Counter(review.condition)),
            "by_group": dict(Counter(review._group)),
            "right_for_severe_fn": int(
                (
                    (review.condition == "right_neural_foraminal_narrowing")
                    & review.is_true_severe
                    & (review.dep_pred != 2)
                ).sum()
            ),  # fmt: skip
        },
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cand = {
        "summary": summary,
        "review_keys": review[
            [
                "key",
                "study_id",
                "level",
                "condition",
                "side",
                "split",
                "severity_index",
                "_group",
                "priority",
            ]
        ].to_dict("records"),  # fmt: skip
    }
    (OUTDIR / "v1_7_hard_case_candidates.json").write_text(
        json.dumps(cand, indent=2, default=float)
    )
    review.to_parquet(OUTDIR / "v1_7_review_set.parquet", index=False)
    sig.to_parquet(OUTDIR / "v1_7_signal_table.parquet", index=False)  # full train+dev (cleaning)
    if TMP.exists():
        TMP.unlink()
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTDIR / 'v1_7_hard_case_candidates.json'} + v1_7_review_set.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
