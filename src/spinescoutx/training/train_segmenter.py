"""Anatomy-segmentation training / evaluation loops for SpineScoutX.

Wires the config, datasets (synthetic or cached-SPIDER), the
:class:`~spinescoutx.models.anatomy_segmenter.UNet2D`, the Dice/CE loss, and the
segmentation metrics into a small, deterministic training loop that runs on CPU
for smoke tests.

Design notes
------------
- ``seed_everything(cfg.seed)`` is called first for reproducibility.
- AMP is enabled only on CUDA.
- EarlyStopping monitors ``val_mean_dice`` with ``mode="max"``.
- ``cfg.train.max_steps`` caps optimizer steps per epoch for smoke tests.
- No fabricated numbers: Dice / IoU come from ``evaluation.segmentation_metrics``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..data.datasets import SpiderSegDataset
from ..data.synthetic import make_synthetic_segmentation_data
from ..evaluation.segmentation_metrics import SegMetricAccumulator
from ..models.anatomy_segmenter import build_segmenter
from ..training.losses import build_segmentation_loss
from ..training.optim import (
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    select_device,
)
from ..utils.logging import emit_json, get_logger
from ..utils.paths import ensure_dir
from ..utils.seed import seed_everything, seed_worker

log = get_logger()

_SEG_INDEX_NAMES: tuple[str, ...] = ("seg_index.parquet", "seg_index.csv")


def _find_spider_index(cache_root: Path) -> Path | None:
    """Return the first existing SPIDER seg index under ``cache_root`` or None."""
    for name in _SEG_INDEX_NAMES:
        candidate = cache_root / name
        if candidate.is_file():
            return candidate
    return None


def _build_seg_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Build ``(train_loader, val_loader)`` for anatomy segmentation.

    Uses cached SPIDER data only when ``data.synthetic`` is False *and* a SPIDER
    seg index is present under ``data.spider_cache``; otherwise synthetic data.
    """
    train_ds: Dataset
    val_ds: Dataset

    spider_index = None
    if not cfg.data.synthetic:
        spider_index = _find_spider_index(Path(cfg.data.spider_cache))

    if spider_index is None:
        if not cfg.data.synthetic:
            log.warning(
                "No SPIDER seg index found under %s; using synthetic seg data.",
                cfg.data.spider_cache,
            )
        train_ds, val_ds = make_synthetic_segmentation_data(
            cfg.data.synthetic_n,
            cfg.data.crop_size,
            cfg.seed,
        )
    else:
        from ..data.crops import read_manifest

        cache_root = Path(cfg.data.spider_cache)
        index_df = read_manifest(spider_index)
        if "split" in index_df.columns:
            train_df = index_df[index_df["split"] == "train"].reset_index(drop=True)
            val_df = index_df[index_df["split"] == "val"].reset_index(drop=True)
            if len(train_df) == 0:
                train_df = index_df
            if len(val_df) == 0:
                val_df = index_df
        else:
            train_df = index_df
            val_df = index_df
        train_ds = SpiderSegDataset(train_df, cache_root, cfg.data.crop_size)
        val_ds = SpiderSegDataset(val_df, cache_root, cfg.data.crop_size)

    num_workers = max(0, int(cfg.data.num_workers))
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    return train_loader, val_loader


def _epoch_steps(loader: DataLoader, max_steps: int | None) -> int:
    """Number of optimizer steps to run this epoch, honoring ``max_steps``."""
    n = len(loader)
    if max_steps is not None and max_steps >= 0:
        n = min(n, int(max_steps))
    return n


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
    max_steps: int | None,
) -> float:
    """Run one segmentation training epoch; return mean loss over taken steps."""
    model.train()
    total_steps = _epoch_steps(loader, max_steps)
    running = 0.0
    taken = 0
    for step, batch in enumerate(loader):
        if step >= total_steps:
            break
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(image)
                loss = loss_fn(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(image)
            loss = loss_fn(logits, mask)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += float(loss.detach().cpu())
        taken += 1
    return running / taken if taken else 0.0


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    """Stream predictions through a :class:`SegMetricAccumulator` and compute."""
    model.eval()
    acc = SegMetricAccumulator(num_classes=num_classes)
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        logits = model(image)
        pred = torch.argmax(logits, dim=1)
        acc.update(pred, mask)
    return acc.compute()


def _checkpoint_payload(
    model: torch.nn.Module,
    cfg: Config,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a serialisable checkpoint dict (state_dict + cfg + metrics)."""
    return {
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "metrics": metrics,
        "model_kind": cfg.model.kind,
    }


def train_segmenter(
    cfg: Config,
    run_dir: str | Path,
    *,
    json_logs: bool = False,
) -> dict[str, Any]:
    """Train the anatomy segmenter and return a results dict.

    Returns ``{"best": {...val seg metrics...}, "checkpoint": str,
    "history": [...]}``. Writes ``best.pt`` / ``last.pt`` plus ``metrics.json``
    and ``config.json`` into ``run_dir``. EarlyStopping monitors ``mean_dice``
    (mode max).
    """
    seed_everything(cfg.seed)
    out_dir = ensure_dir(run_dir)
    device = select_device(cfg.train.device)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"

    num_classes = int(cfg.model.num_anatomy_classes)
    train_loader, val_loader = _build_seg_loaders(cfg)

    model = build_segmenter(cfg.model).to(device)
    loss_fn = build_segmentation_loss(cfg.train, num_classes).to(device)

    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    steps_per_epoch = _epoch_steps(train_loader, cfg.train.max_steps)
    scheduler = build_scheduler(optimizer, cfg.train.epochs, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    stopper = EarlyStopping(
        patience=cfg.train.early_stop_patience,
        mode="max",
        min_delta=0.0,
    )

    history: list[dict[str, Any]] = []
    best_report: dict[str, Any] = {}
    best_mean_dice: float | None = None

    for epoch in range(int(cfg.train.epochs)):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            cfg.train.max_steps,
        )
        report = _evaluate(model, val_loader, num_classes, device)
        mean_dice = float(report["mean_dice"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mean_dice": mean_dice,
            "val_canal_dice": float(report["canal_dice"]),
            "val_dice": report["dice"],
            "val_iou": report["iou"],
        }
        history.append(epoch_record)
        if json_logs:
            emit_json({"event": "epoch_end", **epoch_record})
        else:
            log.info(
                "epoch %d train_loss=%.4f val_mean_dice=%.4f",
                epoch,
                train_loss,
                mean_dice,
            )

        improved = stopper.step(mean_dice)
        if improved or best_mean_dice is None:
            best_mean_dice = mean_dice
            best_report = dict(report)
            torch.save(
                _checkpoint_payload(model, cfg, best_report),
                out_dir / "best.pt",
            )
        torch.save(
            _checkpoint_payload(model, cfg, dict(report)),
            out_dir / "last.pt",
        )
        if stopper.should_stop:
            log.info("early stopping at epoch %d", epoch)
            break

    metrics = {
        "best": best_report,
        "best_mean_dice": best_mean_dice,
        "monitor": "val_mean_dice",
        "monitor_mode": "max",
        "history": history,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True, default=str)
    with (out_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, sort_keys=True, default=str)

    return {
        "best": best_report,
        "checkpoint": str(out_dir / "best.pt"),
        "history": history,
    }


def evaluate_segmenter(
    cfg: Config,
    run_dir: str | Path,
    split: str = "val",
) -> dict[str, Any]:
    """Load ``best.pt`` from ``run_dir`` and evaluate segmentation on ``split``.

    Returns the seg-metrics dict (per-class Dice / IoU, mean_dice, canal_dice)
    plus the split name and checkpoint path.
    """
    seed_everything(cfg.seed)
    out_dir = Path(run_dir)
    ckpt_path = out_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No best.pt checkpoint found in {out_dir}")

    device = select_device(cfg.train.device)
    num_classes = int(cfg.model.num_anatomy_classes)
    model = build_segmenter(cfg.model).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])

    train_loader, val_loader = _build_seg_loaders(cfg)
    loader = train_loader if split == "train" else val_loader

    report = _evaluate(model, loader, num_classes, device)
    result = dict(report)
    result["split"] = split
    result["checkpoint"] = str(ckpt_path)
    return result
