#!/usr/bin/env python3
"""Self-supervised spine-MRI pretraining (v1.6, Plan B) — SimCLR contrastive encoder.

Learns a convnext_tiny foraminal representation by contrastive learning (NT-Xent) over
**unlabelled** sagittal foraminal crops pooled from RSNA train+dev + the LSS external set. **RSNA
locked-test is excluded** so the encoder is valid for headline metrics. Two augmented views per crop
(flip / intensity / resized-crop / noise); the encoder is then fine-tuned by
``train_foraminal_grader_v1_6.py --init <this ckpt>``. Encoder keys match ``ImageClassifier`` for a
drop-in warm start. Crops/weights gitignored. Research-only. Not diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from spinescoutx.data.locked_test import load_splits_v1
from spinescoutx.models.image_classifier import build_backbone
from spinescoutx.training.optim import select_device
from spinescoutx.utils.seed import seed_everything

ROOT = Path("/home/arash/PycharmProjects/SpineScoutX")
SPLITS = ROOT / "data/cache/splits_v1/splits.json"
LSS_CACHE = ROOT / "data/cache/lss_foraminal"
RSNA_CACHE = ROOT / "data/cache/rsna_auto_foraminal"
FORAMINAL = ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"]


def _crop_paths() -> list[str]:
    """Unlabelled foraminal crop paths: RSNA train+dev (NOT test) + all LSS."""
    sm = load_splits_v1(SPLITS)
    r = pd.read_parquet(RSNA_CACHE / "manifest.parquet")
    r = r[r.condition.isin(FORAMINAL)].copy()
    r["study_id"] = r.study_id.astype(str)
    r = r[r.study_id.map(sm).isin(["train", "dev"])]  # exclude locked-test
    paths = [str(RSNA_CACHE / p) for p in r.crop_path]
    lss = pd.read_parquet(LSS_CACHE / "manifest.parquet")
    paths += [str(LSS_CACHE / p) for p in lss.crop_path]
    return paths


def _augment(img: np.ndarray) -> torch.Tensor:
    import cv2

    _, h, w = img.shape
    if np.random.rand() < 0.5:
        img = img[:, :, ::-1]
    # random resized crop (zoom 0.7-1.0) — one 3-channel resize via HWC (fast)
    s = np.random.uniform(0.7, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0, x0 = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
    hwc = np.ascontiguousarray(img[:, y0 : y0 + ch, x0 : x0 + cw].transpose(1, 2, 0))
    img = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    g = 1.0 + np.random.uniform(-0.2, 0.2)
    b = np.random.uniform(-0.1, 0.1)
    img = img * g + b
    if np.random.rand() < 0.5:
        img = img + np.random.normal(0, 0.03, img.shape)
    return torch.from_numpy(np.clip(img, 0.0, 1.0).astype(np.float32))


class ContrastiveDS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.load(self.paths[i]).astype(np.float32)
        return _augment(img), _augment(img)


class SimCLR(nn.Module):
    def __init__(self, backbone="convnext_tiny", proj_dim=128):
        super().__init__()
        self.encoder, feat = build_backbone(backbone, 3, pretrained=True)
        self.proj = nn.Sequential(nn.Linear(feat, 512), nn.ReLU(), nn.Linear(512, proj_dim))

    def forward(self, x):
        return F.normalize(self.proj(self.encoder(x)), dim=1)


def nt_xent(z1, z2, temp=0.2):
    n = z1.shape[0]
    z = torch.cat([z1.float(), z2.float()], dim=0)  # (2n, d), float32 (AMP-safe diagonal mask)
    sim = z @ z.t() / temp
    sim.fill_diagonal_(torch.finfo(sim.dtype).min)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="ssl_foraminal")
    args = ap.parse_args()
    seed_everything(1337)
    device = select_device("auto")
    paths = _crop_paths()
    print(f"[{args.tag}] SSL on {len(paths)} unlabelled foraminal crops "
          f"(RSNA train+dev + LSS; locked-test excluded)", flush=True)  # fmt: skip
    dl = DataLoader(
        ContrastiveDS(paths), batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True
    )
    model = SimCLR().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for v1, v2 in dl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = nt_xent(model(v1.to(device)), model(v2.to(device)))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nb += 1
        sch.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep} nt_xent {tot / max(nb, 1):.4f}", flush=True)
    ckpt_dir = ROOT / f"runs/{args.tag}_v1_6"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # save under encoder.* so ImageClassifier.encoder.load_state_dict(strict=False) warm-starts
    sd = {f"encoder.{k}": v for k, v in model.encoder.state_dict().items()}
    torch.save({"state_dict": sd}, ckpt_dir / "best.pt")
    print(f"wrote {ckpt_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
