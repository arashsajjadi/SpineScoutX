"""Train / evaluate the disc-level keypoint localizer (heatmap regression).

Model selection on mean peak-to-GT pixel distance (val). Reports PCK@{10,20,32},
per-level error, and crop-hit rate (does the predicted point land within a crop of
the GT, so an auto-crop would still contain the true localizer).
Research-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..constants import LEVELS
from ..data.localizer import LocalizerDataset, extract_peaks, pck
from ..models.disc_localizer import build_disc_localizer
from ..training.optim import EarlyStopping, build_optimizer, build_scheduler, select_device
from ..utils.logging import emit_json, get_logger
from ..utils.paths import ensure_dir
from ..utils.seed import seed_everything

log = get_logger()


def _loaders(cfg: Config) -> tuple[DataLoader, DataLoader, pd.DataFrame]:  # noqa: F821
    import pandas as pd

    cache = Path(cfg.data.rsna_cache)
    manifest = pd.read_parquet(cache / "localizer_manifest.parquet")
    size = int(cfg.data.crop_size)
    tr = LocalizerDataset(manifest[manifest.split == "train"], cache, size)
    va = LocalizerDataset(manifest[manifest.split == "val"], cache, size)
    nw = max(0, int(cfg.data.num_workers))
    return (
        DataLoader(
            tr, batch_size=cfg.train.batch_size, shuffle=True, num_workers=nw, drop_last=False
        ),
        DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False, num_workers=nw),
        manifest,
    )


@torch.no_grad()
def _evaluate(model, loader, device, size) -> dict[str, Any]:
    model.eval()
    dists: list[float] = []
    per_level = {lv: [] for lv in LEVELS}
    crop_hit = {224: [], 256: []}
    pck_all = {10: [], 20: [], 32: []}
    for batch in loader:
        hm = model.heatmaps(batch["image"].to(device)).cpu().numpy()
        gt = batch["keypoints"].numpy()  # [B, 5, 2] in slice space
        for b in range(hm.shape[0]):
            pred = extract_peaks(hm[b])  # [5,2]
            valid = np.isfinite(gt[b]).all(axis=1)
            d = np.linalg.norm(pred - gt[b], axis=1)
            dists.extend(d[valid].tolist())
            for li, lv in enumerate(LEVELS):
                if valid[li]:
                    per_level[lv].append(float(d[li]))
            p = pck(pred, gt[b], (10, 20, 32))
            for t in (10, 20, 32):
                if not np.isnan(p[f"pck@{t}"]):
                    pck_all[t].append(p[f"pck@{t}"])
            # crop-hit: predicted within crop/2 of GT (so an auto-crop contains GT)
            for cs in (224, 256):
                crop_hit[cs].extend((d[valid] <= cs / 2).tolist())
    return {
        "mean_px_dist": float(np.mean(dists)) if dists else float("inf"),
        "median_px_dist": float(np.median(dists)) if dists else float("inf"),
        "pck@10": float(np.mean(pck_all[10])) if pck_all[10] else 0.0,
        "pck@20": float(np.mean(pck_all[20])) if pck_all[20] else 0.0,
        "pck@32": float(np.mean(pck_all[32])) if pck_all[32] else 0.0,
        "crop_hit@224": float(np.mean(crop_hit[224])) if crop_hit[224] else 0.0,
        "crop_hit@256": float(np.mean(crop_hit[256])) if crop_hit[256] else 0.0,
        "per_level_mean_px": {
            lv: (float(np.mean(v)) if v else float("nan")) for lv, v in per_level.items()
        },
        "n_keypoints": len(dists),
    }


def train_localizer(cfg: Config, run_dir: str | Path, *, json_logs: bool = False) -> dict[str, Any]:
    """Train the disc-level localizer; return best metrics + checkpoint path."""
    seed_everything(cfg.seed)
    out = ensure_dir(run_dir)
    device = select_device(cfg.train.device)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    size = int(cfg.data.crop_size)

    train_loader, val_loader, _ = _loaders(cfg)
    model = build_disc_localizer(cfg.model).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    steps = min(len(train_loader), cfg.train.max_steps or len(train_loader))
    scheduler = build_scheduler(optimizer, cfg.train.epochs, steps)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience, mode="min")

    best: dict[str, Any] = {}
    best_metric: float | None = None
    history = []
    for epoch in range(int(cfg.train.epochs)):
        model.train()
        for step, batch in enumerate(train_loader):
            if cfg.train.max_steps is not None and step >= cfg.train.max_steps:
                break
            img = batch["image"].to(device)
            target = batch["heatmap"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(torch.sigmoid(model(img)), target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(torch.sigmoid(model(img)), target)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        metrics = _evaluate(model, val_loader, device, size)
        rec = {
            "epoch": epoch,
            **{f"val_{k}": v for k, v in metrics.items() if not isinstance(v, dict)},
        }
        history.append(rec)
        emit_json({"event": "epoch_end", **rec}) if json_logs else log.info(
            "epoch %d val_mean_px=%.2f pck@20=%.3f crop_hit@224=%.3f",
            epoch,
            metrics["mean_px_dist"],
            metrics["pck@20"],
            metrics["crop_hit@224"],
        )
        improved = stopper.step(metrics["mean_px_dist"])
        if improved or best_metric is None:
            best_metric, best = metrics["mean_px_dist"], dict(metrics)
            torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, out / "best.pt")
        torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, out / "last.pt")
        if stopper.should_stop:
            log.info("early stopping at epoch %d", epoch)
            break

    (out / "metrics.json").write_text(
        json.dumps({"best": best, "history": history}, indent=2, default=str)
    )
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    return {"best": best, "checkpoint": str(out / "best.pt"), "history": history}
