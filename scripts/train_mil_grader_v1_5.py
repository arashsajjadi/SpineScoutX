#!/usr/bin/env python3
"""Train + evaluate a candidate-bag MIL severity grader (v1.5) for a weak route.

Real training: encode K candidate crops with a shared ConvNeXt-Tiny (warm-started from the
deployed grader), aggregate (attention/gated/max), fuse level+condition embeddings → 3 severity
logits. Bounded config sweep; **dev** selects; **locked-test once** for the best config.
Severe oversampling + focal/CE. No GT coords (bags are auto); reference severity is the label.

Research-only. Not diagnostic. Usage: `--route {right_foraminal,left_foraminal,subarticular}`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.models.mil_grader import build_mil_grader
from spinescoutx.training.optim import select_device
from spinescoutx.utils.seed import seed_everything

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
BAGS = ROOT / "data/cache/v1_5_candidate_bags"
OUTDIR = ROOT / "outputs/real"
K = 5
ROUTE_CFG = {
    "right_foraminal": (
        "foraminal",
        ["right_neural_foraminal_narrowing"],
        "runs/v1_foraminal_oracle_ctrl",
    ),
    "left_foraminal": (
        "foraminal",
        ["left_neural_foraminal_narrowing"],
        "runs/v1_foraminal_oracle_ctrl",
    ),
    "foraminal": (
        "foraminal",
        ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"],
        "runs/v1_foraminal_oracle_ctrl",
    ),
    "subarticular": (
        "subarticular",
        ["left_subarticular_stenosis", "right_subarticular_stenosis"],
        "runs/v1_subarticular_auto_robust",
    ),
}


class BagDS(Dataset):
    def __init__(self, df, cache_root, augment=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(cache_root)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        arr = np.load(self.root / str(r["bag_path"])).astype(np.float32)  # (k,3,H,W)
        k = arr.shape[0]
        bags = np.zeros((K, 3, arr.shape[2], arr.shape[3]), dtype=np.float32)
        bags[:k] = arr[:K]
        mask = np.zeros(K, dtype=bool)
        mask[: min(k, K)] = True
        if self.augment:
            g = 1.0 + np.random.uniform(-0.1, 0.1)
            b = np.random.uniform(-0.05, 0.05)
            bags[mask] = np.clip(bags[mask] * g + b, 0.0, 1.0)
        return {
            "bags": torch.from_numpy(bags),
            "mask": torch.from_numpy(mask),
            "level_idx": torch.tensor(LEVEL_TO_INDEX[str(r["level"])], dtype=torch.long),
            "condition_idx": torch.tensor(
                CONDITION_TO_INDEX[str(r["condition"])], dtype=torch.long
            ),
            "target": torch.tensor(int(r["severity_index"]), dtype=torch.long),
            "study_id": str(r["study_id"]),
        }


def focal_loss(logits, target, gamma=2.0, weight=None):
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1 - pt) ** gamma) * logp.gather(1, target.unsqueeze(1)).squeeze(1)
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def _loader(df, cache_root, *, train, severe_over):
    ds = BagDS(df, cache_root, augment=train)
    if train and severe_over:
        y = df["severity_index"].to_numpy()
        # inverse-frequency sample weights, severe up-weighted
        freq = np.bincount(y, minlength=3).astype(float)
        w_class = 1.0 / np.clip(freq, 1, None)
        w_class[2] *= 2.0  # extra severe oversampling
        w = w_class[y]
        sampler = WeightedRandomSampler(w, num_samples=len(df), replacement=True)
        return DataLoader(ds, batch_size=16, sampler=sampler, num_workers=6, drop_last=True)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=6)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    ys, ps, sts = [], [], []
    for b in loader:
        logit = model(
            b["bags"].to(device),
            b["level_idx"].to(device),
            b["condition_idx"].to(device),
            b["mask"].to(device),
        )
        ps.append(torch.softmax(logit.float(), 1).cpu().numpy())
        ys.append(b["target"].numpy())
        sts.extend(b["study_id"])
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p, np.array(sts)


def train_one(cfg, tr_df, dv_df, cache_root, warm_sd, device, epochs):
    seed_everything(cfg["seed"])
    model = build_mil_grader(
        pooling=cfg["pooling"], instance_dropout=cfg.get("idrop", 0.0), pretrained=True
    ).to(device)
    if warm_sd is not None:
        model.load_encoder_from(warm_sd)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler()
    tl = _loader(tr_df, cache_root, train=True, severe_over=cfg["severe_over"])
    vl = _loader(dv_df, cache_root, train=False, severe_over=False)
    freq = np.bincount(tr_df["severity_index"].to_numpy(), minlength=3).astype(float)
    cw = torch.tensor(
        (freq.sum() / np.clip(freq, 1, None)) / (freq.sum() / np.clip(freq, 1, None)).mean(),
        dtype=torch.float32,
        device=device,
    )
    best_sr, best_state = -1.0, None
    for _ep in range(epochs):
        model.train()
        for b in tl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logit = model(
                    b["bags"].to(device),
                    b["level_idx"].to(device),
                    b["condition_idx"].to(device),
                    b["mask"].to(device),
                )
                tgt = b["target"].to(device)
                loss = (
                    focal_loss(logit, tgt, weight=cw)
                    if cfg["loss"] == "focal"
                    else torch.nn.functional.cross_entropy(logit, tgt, weight=cw)
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sch.step()
        y, p, _ = _eval(model, vl, device)
        sr = bs.m_severe_recall(y, p)
        if sr > best_sr:
            best_sr = sr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, choices=list(ROUTE_CFG))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=0, help="0=all train bags")
    args = ap.parse_args()
    device = select_device("auto")
    cache_kind, conds, warm_run = ROUTE_CFG[args.route]
    cache_root = BAGS / cache_kind
    import pandas as pd

    man = pd.read_parquet(cache_root / "bag_manifest.parquet")
    man = man[man.condition.isin(conds) & man.severity_index.isin([0, 1, 2])].copy()
    tr = man[man.split == "train"]
    if args.max_train:
        tr = tr.sample(n=min(args.max_train, len(tr)), random_state=0)
    dv = man[man.split == "dev"]
    te = man[man.split == "test"]
    warm_sd = None
    wp = ROOT / warm_run / "best.pt"
    if wp.exists():
        warm_sd = torch.load(wp, map_location="cpu")["state_dict"]

    configs = [
        {"pooling": "attention", "loss": "focal", "severe_over": True, "lr": 3e-4, "seed": 1337},
        {
            "pooling": "gated",
            "loss": "focal",
            "severe_over": True,
            "lr": 3e-4,
            "idrop": 0.1,
            "seed": 1337,
        },
        {"pooling": "max", "loss": "ce", "severe_over": True, "lr": 3e-4, "seed": 1337},
    ]
    results = []
    best = None
    for cfg in configs:
        model, dev_sr = train_one(cfg, tr, dv, cache_root, warm_sd, device, args.epochs)
        results.append({"config": cfg, "dev_severe_recall": float(dev_sr)})
        print(f"  cfg {cfg['pooling']}/{cfg['loss']}: dev severe recall {dev_sr:.3f}", flush=True)
        if best is None or dev_sr > best[1]:
            best = (model, dev_sr, cfg)
    # locked-test ONCE for the dev-best config
    model, dev_sr, cfg = best
    vl_te = _loader(te, cache_root, train=False, severe_over=False)
    yt, pt, stt = _eval(model, vl_te, device)
    out = {
        "route": args.route,
        "conditions": conds,
        "n_train_bags": int(len(tr)),
        "best_config": cfg,
        "dev_severe_recall": float(dev_sr),
        "test": {},
        "all_configs": results,
    }
    out["test"]["severe_recall"] = float(bs.m_severe_recall(yt, pt))
    out["test"]["ci"] = bs.bootstrap_ci(yt, pt, stt, bs.m_severe_recall, n_boot=2000)
    out["test"]["recall_at_far10"] = bs.bootstrap_ci(
        yt, pt, stt, bs.make_recall_at_far(0.10), n_boot=2000
    )
    out["test"]["n"] = int(len(yt))
    out["test"]["n_severe"] = int((yt == 2).sum())
    # per-condition (re-eval each condition separately for clean numbers)
    out["per_condition"] = {}
    for cond in conds:
        sub = te[te.condition == cond]
        if sub.empty:
            continue
        y, p, st = _eval(model, _loader(sub, cache_root, train=False, severe_over=False), device)
        out["per_condition"][cond] = {
            "severe_recall": float(bs.m_severe_recall(y, p)),
            "ci": bs.bootstrap_ci(y, p, st, bs.m_severe_recall, n_boot=2000),
            "n_severe": int((y == 2).sum()),
        }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"mil_{args.route}_v1_5.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[{args.route}] dev best {cfg['pooling']}/{cfg['loss']} dev_sr={dev_sr:.3f}")
    for cond, d in out["per_condition"].items():
        ci = d["ci"]
        print(
            f"  TEST {cond}: severe recall {d['severe_recall']:.3f} "
            f"[{ci['ci_lo']:.3f},{ci['ci_hi']:.3f}] (n_sev={d['n_severe']})"
        )
    print(f"wrote {OUTDIR / f'mil_{args.route}_v1_5.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
