#!/usr/bin/env python3
"""Noise-aware foraminal grader trainer (v1.7) — train AFTER label repair.

Trains the deployed convnext_tiny foraminal architecture on RSNA splits_v1 with label-quality
modes (the raw RSNA labels are never modified on disk; soft labels / weights come from the
gitignored provisional parquet, train+dev only):
  A original-label baseline · C provisional soft labels · D original + sample weights ·
  E original + ambiguity downweight · F original + severe-FN upweight · G hybrid soft+ordinal+wts.
Unified loss: ``w_i · Σ_c cw_c · soft_ic · -log p_ic`` (+ ordinal MSE for G). Dev selects on
right-foraminal recall@FAR≤10 (anti-spam: reject FAR>0.5 / all-normal / all-severe). Locked-test is
read once for the dev-best mode. Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.models.image_classifier import ImageClassifier
from spinescoutx.training.optim import select_device
from spinescoutx.utils.seed import seed_everything

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
SOFT = ROOT / "data/labels/v1_7_provisional_soft_labels.parquet"
OUTDIR = ROOT / "outputs/real"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]
RIGHT = "right_neural_foraminal_narrowing"


class DS(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = np.load(RSNA_CACHE / r["crop_path"]).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1].copy()
            img = np.clip(img * (1 + np.random.uniform(-0.1, 0.1)) + np.random.uniform(-0.05, 0.05),
                          0.0, 1.0)  # fmt: skip
        return {
            "image": torch.from_numpy(img),
            "level_idx": torch.tensor(int(r["level_idx"]), dtype=torch.long),
            "condition_idx": torch.tensor(int(r["condition_idx"]), dtype=torch.long),
            "target": torch.tensor(int(r["severity_index"]), dtype=torch.long),
            "soft": torch.tensor([r["s0"], r["s1"], r["s2"]], dtype=torch.float32),
            "weight": torch.tensor(float(r["w"]), dtype=torch.float32),
            "key": str(r["key"]),
        }


def _frames(mode):
    m = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
    m = m[m.condition.isin(FORAMINAL) & m.severity_index.isin([0, 1, 2])].copy()
    m["study_id"] = m.study_id.astype(str)
    m["condition_idx"] = m.condition.map(CONDITION_TO_INDEX)
    m["level_idx"] = m.level.map(LEVEL_TO_INDEX)
    m["key"] = m.study_id + "|" + m.level.astype(str) + "|" + m.condition
    m["spl"] = m.study_id.map(load_splits_v1(SPLITS))
    soft = pd.read_parquet(SOFT).set_index("key")
    one_hot = np.eye(3)
    s0, s1, s2, w = [], [], [], []
    for r in m.itertuples():
        in_soft = r.key in soft.index
        row = soft.loc[r.key] if in_soft else None
        if in_soft and mode in ("C", "G"):  # soft-label modes
            vec = [float(row.soft0), float(row.soft1), float(row.soft2)]
        else:
            vec = list(one_hot[int(r.severity_index)])
        s0.append(vec[0])
        s1.append(vec[1])
        s2.append(vec[2])
        wt = 1.0
        if in_soft and mode in ("D", "G"):
            wt = float(row.sample_weight)
        elif in_soft and mode == "E":
            wt = 0.5 if bool(row.ambiguity_flag) else 1.0
        elif mode == "F":
            wt = 2.0 if int(r.severity_index) == 2 else 1.0
        w.append(wt)
    m["s0"], m["s1"], m["s2"], m["w"] = s0, s1, s2, w
    return m[m.spl == "train"], m[m.spl == "dev"], m[m.spl == "test"]


def _class_weights(y):
    f = np.bincount(y, minlength=3).astype(float)
    return torch.tensor(f.sum() / (3 * np.clip(f, 1, None)), dtype=torch.float32)


def _far(y, p):
    neg = y != 2
    return float((p[neg].argmax(1) == 2).mean()) if neg.any() else float("nan")


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    ys, ps, ks = [], [], []
    for b in loader:
        lg = model(b["image"].to(device), b["level_idx"].to(device), b["condition_idx"].to(device))
        ps.append(torch.softmax(lg.float(), 1).cpu().numpy())
        ys.append(b["target"].numpy())
        ks.extend(b["key"])
    return np.concatenate(ys), np.concatenate(ps), np.array(ks)


def _rfor_recall_at_far(y, p, keys):
    m = np.array([k.endswith(RIGHT) for k in keys])
    if m.sum() < 5 or not (y[m] == 2).any():
        return 0.0
    return float(bs.make_recall_at_far(0.10)(y[m], p[m]))


def train_mode(mode, tr, dv, device, epochs):
    seed_everything(1337)
    model = ImageClassifier(
        backbone="convnext_tiny", in_chans=3, num_classes=3, use_level_embedding=True,
        use_condition_embedding=True, embed_dim=16, dropout=0.2, pretrained=True,
    ).to(device)  # fmt: skip
    cw = _class_weights(tr.severity_index.to_numpy()).to(device)
    tl = DataLoader(
        DS(tr, augment=True), batch_size=32, shuffle=True, num_workers=8, drop_last=True
    )
    vl = DataLoader(DS(dv), batch_size=64, shuffle=False, num_workers=8)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    model.set_backbone_trainable(False)
    best_key, best_state, best = -1.0, None, {}
    for ep in range(epochs):
        if ep == 2:
            model.set_backbone_trainable(True)
            for g in opt.param_groups:
                g["lr"] = 3e-4 * 0.2
        model.train()
        for b in tl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                lg = model(b["image"].to(device), b["level_idx"].to(device),
                           b["condition_idx"].to(device))  # fmt: skip
                logp = torch.log_softmax(lg, 1)
                soft = b["soft"].to(device)
                ce = -(cw * soft * logp).sum(1)  # class-weighted soft CE
                loss = (b["weight"].to(device) * ce).mean()
                if mode == "G":  # ordinal consistency: E[severity] should match target
                    exp_sev = (torch.softmax(lg, 1) * torch.arange(3, device=device)).sum(1)
                    loss = loss + 0.2 * ((exp_sev - b["target"].to(device).float()) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        y, p, k = _eval(model, vl, device)
        sel = _rfor_recall_at_far(y, p, k)
        far = _far(y, p)
        sev_r = float(bs.m_severe_recall(y, p))
        ok = (not np.isnan(far)) and far <= 0.5 and 0.0 < sev_r < 1.0  # anti-spam guardrails
        print(f"    [{mode}] ep{ep} dev R-for r@FAR10 {sel:.3f} sevR {sev_r:.3f} FAR {far:.3f}",
              flush=True)  # fmt: skip
        key = sel if ok else -1.0
        if key > best_key:
            best_key = key
            best = {"sel_rfor_r@far10": float(sel), "dev_severe_recall": sev_r, "far": float(far),
                    "epoch": ep}  # fmt: skip
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
    if best_state is None:
        best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["A", "C", "F", "G"])
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    device = select_device("auto")
    results, best_overall = {}, None
    for mode in args.modes:
        tr, dv, te = _frames(mode)
        print(f"[mode {mode}] train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)
        model, dm = train_mode(mode, tr, dv, device, args.epochs)
        results[mode] = dm
        print(f"  [{mode}] dev R-for r@FAR10 {dm.get('sel_rfor_r@far10', 0):.3f} "
              f"(ep {dm.get('epoch')})", flush=True)  # fmt: skip
        if best_overall is None or dm.get("sel_rfor_r@far10", 0) > best_overall[1]:
            best_overall = (mode, dm.get("sel_rfor_r@far10", 0), model, te)
    # locked-test ONCE for the dev-best mode
    mode, _key, model, te = best_overall
    yt, pt, kt = _eval(model, DataLoader(DS(te), batch_size=64, num_workers=8), device)
    out = {"dev_best_mode": mode, "modes_dev": results, "test": {}}
    sides = np.array([k.split("|")[-1] for k in kt])
    st = np.array([k.split("|")[0] for k in kt])
    for c in FORAMINAL:
        msk = sides == c
        out["test"][c] = {
            "severe_recall": float(bs.m_severe_recall(yt[msk], pt[msk])),
            "ci": bs.bootstrap_ci(yt[msk], pt[msk], st[msk], bs.m_severe_recall, n_boot=2000),
            "recall_at_far10": float(bs.make_recall_at_far(0.10)(yt[msk], pt[msk])),
            "far": _far(yt[msk], pt[msk]), "n_severe": int((yt[msk] == 2).sum()),
        }  # fmt: skip
    out["test"]["foraminal_macro_severe_recall"] = float(
        np.mean([out["test"][c]["severe_recall"] for c in FORAMINAL])
    )
    pd.DataFrame({"key": kt, "y": yt, "p0": pt[:, 0], "p1": pt[:, 1], "p2": pt[:, 2]}).to_parquet(
        OUTDIR / "foraminal_noise_aware_test_preds.parquet", index=False
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "noise_aware_foraminal_v1_7.json").write_text(
        json.dumps(out, indent=2, default=float)
    )
    print(f"\n[noise-aware] dev-best mode {mode}")
    for c in FORAMINAL:
        d = out["test"][c]
        print(f"  TEST {c}: severe recall {d['severe_recall']:.3f} "
              f"[{d['ci']['ci_lo']:.3f},{d['ci']['ci_hi']:.3f}] r@FAR10 {d['recall_at_far10']:.3f}",
              flush=True)  # fmt: skip
    print(f"  TEST foraminal macro {out['test']['foraminal_macro_severe_recall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
