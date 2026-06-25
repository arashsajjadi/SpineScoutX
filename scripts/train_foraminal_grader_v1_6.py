#!/usr/bin/env python3
"""Foraminal severity grader trainer (v1.6, Plan A) — LSS pretraining -> RSNA fine-tuning.

One trainer for both stages of the external-data transfer experiment:
  * ``--data lss``  trains a convnext_tiny foraminal grader on the LSS-MRI AISSLab crop cache
    (patient-level lss_train/lss_dev), and saves the encoder for transfer.
  * ``--data rsna`` trains the SAME architecture on the RSNA auto-foraminal crops (splits_v1
    train/dev/test), with ``--init imagenet`` (baseline) or ``--init <lss_ckpt>`` (transfer).

Selection is on **dev** foraminal-macro recall@FAR<=10% (spam-resistant, FAR guardrail); RSNA
**locked-test is read once** for the dev-selected model. Per-finding dev/test probs are dumped for
a paired baseline-vs-transfer comparison. No test labels as input; external/RSNA crops gitignored.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from spinescoutx.constants import CONDITION_TO_INDEX, LEVEL_TO_INDEX
from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.evaluation import bootstrap as bs
from spinescoutx.models.image_classifier import ImageClassifier
from spinescoutx.training.optim import select_device
from spinescoutx.utils.seed import seed_everything

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
LSS_CACHE = ROOT / "data/cache/lss_foraminal"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
OUTDIR = ROOT / "outputs/real"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]


class CropDS(Dataset):
    def __init__(self, df, cache_root, augment=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(cache_root)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        path = r["abs_path"] if "abs_path" in r else str(self.root / r["crop_path"])
        img = np.load(path).astype(np.float32)  # (3,224,224)
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1].copy()  # horizontal flip (sagittal L<->R ambiguity handled
                # by condition embedding; flip is a mild intensity-preserving augment)
            g = 1.0 + np.random.uniform(-0.1, 0.1)
            b = np.random.uniform(-0.05, 0.05)
            img = np.clip(img * g + b, 0.0, 1.0)
        return {
            "image": torch.from_numpy(img),
            "level_idx": torch.tensor(int(r["level_idx"]), dtype=torch.long),
            "condition_idx": torch.tensor(int(r["condition_idx"]), dtype=torch.long),
            "target": torch.tensor(int(r["severity_index"]), dtype=torch.long),
            "key": str(r["key"]),
        }


def _lss_train_rows():
    """LSS lss_train rows with absolute crop paths + a unified key (for joint training)."""
    m = pd.read_parquet(LSS_CACHE / "manifest.parquet")
    m = m[m.split == "lss_train"].copy()
    m["abs_path"] = m.crop_path.map(lambda p: str(LSS_CACHE / p))
    m["key"] = "lss|" + m.patient.astype(str) + "|" + m.index.astype(str)
    return m[["key", "abs_path", "level_idx", "condition_idx", "severity_index"]]


def _load_frames(data: str, extra_lss: bool = False):
    """Return (train_df, dev_df, test_df|None) with a unified 'key' + 'abs_path' column."""
    if data == "lss":
        m = pd.read_parquet(LSS_CACHE / "manifest.parquet")
        m["key"] = m.patient.astype(str) + "|" + m.index.astype(str)
        m["abs_path"] = m.crop_path.map(lambda p: str(LSS_CACHE / p))
        return m[m.split == "lss_train"], m[m.split == "lss_dev"], None
    m = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
    m = m[m.condition.isin(FORAMINAL) & m.severity_index.isin([0, 1, 2])].copy()
    m["study_id"] = m.study_id.astype(str)
    m["condition_idx"] = m.condition.map(CONDITION_TO_INDEX)
    m["level_idx"] = m.level.map(LEVEL_TO_INDEX)
    m["key"] = m.study_id + "|" + m.level.astype(str) + "|" + m.condition
    m["abs_path"] = m.crop_path.map(lambda p: str(RSNA_CACHE / p))
    sm = load_splits_v1(SPLITS)
    m["spl"] = m.study_id.map(sm)
    tr, dv, te = m[m.spl == "train"], m[m.spl == "dev"], m[m.spl == "test"]
    if extra_lss:  # joint training: pool LSS lss_train severe-rich crops into RSNA train
        cols = ["key", "abs_path", "level_idx", "condition_idx", "severity_index"]
        tr = pd.concat([tr[cols], _lss_train_rows()[cols]], ignore_index=True)
    return tr, dv, te


def _loader(df, cache_root, *, train, severe_over=False):
    ds = CropDS(df, cache_root, augment=train)
    if train and severe_over:
        y = df["severity_index"].to_numpy()
        freq = np.bincount(y, minlength=3).astype(float)
        w = (1.0 / np.sqrt(np.clip(freq, 1, None)))[y]
        sampler = WeightedRandomSampler(w, num_samples=len(df), replacement=True)
        return DataLoader(ds, batch_size=32, sampler=sampler, num_workers=8, drop_last=True)
    bs_ = 32 if train else 64
    return DataLoader(ds, batch_size=bs_, shuffle=train, num_workers=8, drop_last=train)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    ys, ps, ks = [], [], []
    for b in loader:
        logit = model(
            b["image"].to(device), b["level_idx"].to(device), b["condition_idx"].to(device)
        )
        ps.append(torch.softmax(logit.float(), 1).cpu().numpy())
        ys.append(b["target"].numpy())
        ks.extend(b["key"])
    return np.concatenate(ys), np.concatenate(ps), np.array(ks)


def _far(y, p):
    neg = y != 2
    return float((p[neg].argmax(1) == 2).mean()) if neg.any() else float("nan")


def _macro_recall_at_far(df_keys, y, p, conds):
    """Mean over present foraminal conditions of recall@FAR<=10% (selection metric)."""
    sides = np.array([k.split("|")[-1] for k in df_keys])
    vals = []
    for c in conds:
        m = sides == c
        if m.sum() >= 5 and (y[m] == 2).any():
            vals.append(bs.make_recall_at_far(0.10)(y[m], p[m]))
    return float(np.mean(vals)) if vals else 0.0


def _class_weights(y):
    freq = np.bincount(y, minlength=3).astype(float)
    w = freq.sum() / (3 * np.clip(freq, 1, None))
    return torch.tensor(w, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["lss", "rsna"])
    ap.add_argument("--init", default="imagenet", help="'imagenet' or path to an encoder ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--severe-over", action="store_true")
    ap.add_argument("--extra-lss", action="store_true", help="joint: pool LSS lss_train into RSNA")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = select_device("auto")
    cache_root = LSS_CACHE if args.data == "lss" else RSNA_CACHE
    tr, dv, te = _load_frames(args.data, extra_lss=args.extra_lss)
    print(f"[{args.tag}] data={args.data} train={len(tr)} dev={len(dv)} "
          f"test={0 if te is None else len(te)} (train severe={int((tr.severity_index==2).sum())})",
          flush=True)  # fmt: skip

    model = ImageClassifier(
        backbone="convnext_tiny", in_chans=3, num_classes=3,
        use_level_embedding=True, use_condition_embedding=True,
        embed_dim=16, dropout=0.2, pretrained=True,
    ).to(device)  # fmt: skip
    if args.init != "imagenet":
        sd = torch.load(args.init, map_location="cpu")["state_dict"]
        enc = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}
        miss = model.encoder.load_state_dict(enc, strict=False)
        n_loaded = len(enc) - len(getattr(miss, "missing_keys", []))
        print(f"  warm-started encoder from {args.init} ({n_loaded} keys)", flush=True)

    cw = _class_weights(tr.severity_index.to_numpy()).to(device)
    tl = _loader(tr, cache_root, train=True, severe_over=args.severe_over)
    vl = _loader(dv, cache_root, train=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    model.set_backbone_trainable(False)

    best_key, best_state, best = -1.0, None, {}
    for ep in range(args.epochs):
        if ep == args.freeze_epochs:
            model.set_backbone_trainable(True)
            for g in opt.param_groups:
                g["lr"] = args.lr * 0.2
        model.train()
        for b in tl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logit = model(b["image"].to(device), b["level_idx"].to(device),
                              b["condition_idx"].to(device))  # fmt: skip
                loss = torch.nn.functional.cross_entropy(logit, b["target"].to(device), weight=cw)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        y, p, k = _eval(model, vl, device)
        sel = _macro_recall_at_far(k, y, p, FORAMINAL) if args.data == "rsna" else \
            bs.make_recall_at_far(0.10)(y, p)  # fmt: skip
        far = _far(y, p)
        print(f"  ep{ep} dev sel(r@FAR10) {sel:.3f} sevR {bs.m_severe_recall(y,p):.3f} "
              f"FAR {far:.3f}", flush=True)  # fmt: skip
        key = sel if (np.isnan(far) or far <= 0.5) else -1.0
        if key > best_key:
            best_key = key
            best = {"sel": float(sel), "severe_recall": float(bs.m_severe_recall(y, p)),
                    "far": float(far), "epoch": ep}  # fmt: skip
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT / f"runs/foraminal_{args.tag}_v1_6"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt_dir / "best.pt")
    out = {"tag": args.tag, "data": args.data, "init": args.init, "dev_best": best,
           "selection": "dev foraminal-macro recall@FAR10", "n_train": int(len(tr))}  # fmt: skip

    def dump(df, split):
        y, p, k = _eval(model, _loader(df, cache_root, train=False), device)
        out_df = pd.DataFrame(
            {"key": k, "y": y, "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], "split": split}
        )
        out_df.to_parquet(OUTDIR / f"foraminal_{args.tag}_{split}_preds.parquet", index=False)
        return y, p, k

    dump(dv, "dev")
    if te is not None:
        yt, pt, kt = dump(te, "test")
        out["test"] = {}
        sides = np.array([k.split("|")[-1] for k in kt])
        for c in FORAMINAL:
            m = sides == c
            st = np.array([k.split("|")[0] for k in kt])[m]
            out["test"][c] = {
                "severe_recall": float(bs.m_severe_recall(yt[m], pt[m])),
                "ci": bs.bootstrap_ci(yt[m], pt[m], st, bs.m_severe_recall, n_boot=2000),
                "recall_at_far10": float(bs.make_recall_at_far(0.10)(yt[m], pt[m])),
                "far": _far(yt[m], pt[m]),
                "n_severe": int((yt[m] == 2).sum()),
            }
        macro = float(np.mean([out["test"][c]["severe_recall"] for c in FORAMINAL]))
        out["test"]["foraminal_macro_severe_recall"] = macro
    dst = OUTDIR / f"foraminal_{args.tag}_v1_6.json"
    dst.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[{args.tag}] dev sel {best.get('sel'):.3f} (ep {best.get('epoch')})", flush=True)
    if te is not None:
        for c in FORAMINAL:
            d = out["test"][c]
            ci = d["ci"]
            print(
                f"  TEST {c}: severe recall {d['severe_recall']:.3f} "
                f"[{ci['ci_lo']:.3f},{ci['ci_hi']:.3f}] r@FAR10 {d['recall_at_far10']:.3f}",
                flush=True,
            )
        macro = out["test"]["foraminal_macro_severe_recall"]
        print(f"  TEST foraminal macro severe recall {macro:.3f}")
    print(f"wrote {ckpt_dir / 'best.pt'} + preds + json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
