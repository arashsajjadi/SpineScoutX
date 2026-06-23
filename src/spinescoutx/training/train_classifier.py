"""Severity-classification training / evaluation loops for SpineScoutX.

This module wires together the config, datasets (synthetic or cached-RSNA),
the :class:`~spinescoutx.models.image_classifier.ImageClassifier`, the loss /
optimizer / scheduler builders, and the classification metrics into a small,
deterministic training loop that runs on CPU for smoke tests.

Design notes
------------
- ``seed_everything(cfg.seed)`` is called first so the whole run is reproducible.
- Automatic mixed precision is enabled only on CUDA.
- The backbone is frozen for ``cfg.train.freeze_backbone_epochs`` warmup epochs
  and unfrozen afterwards (the optimizer is rebuilt so newly-trainable params
  are picked up).
- Either a class-weighted loss or a weighted sampler is used (per config); both
  together are allowed.
- ``cfg.train.max_steps`` caps the number of optimizer steps per epoch so smoke
  tests stay fast.
- No fabricated numbers: every reported metric is computed from real model
  outputs by ``evaluation.classification_metrics``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..constants import NUM_SEVERITY_CLASSES
from ..data.datasets import (
    RsnaCropDataset,
    build_class_weights,
    make_weighted_sampler,
)
from ..data.synthetic import make_synthetic_classification_data
from ..evaluation.classification_metrics import classification_report_dict
from ..models.anatomy_guided_classifier import build_anatomy_guided_classifier
from ..models.image_classifier import build_image_classifier
from ..training.losses import build_classification_loss
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

_MANIFEST_NAMES: tuple[str, ...] = ("manifest.parquet", "manifest.csv")


def _is_guided(cfg: Config) -> bool:
    """True when this run trains the anatomy-guided model (task or model kind)."""
    return cfg.task == "anatomy_guided" or cfg.model.kind == "anatomy_guided_classifier"


def _build_model(cfg: Config) -> torch.nn.Module:
    """Build the image-only or anatomy-guided classifier from the config."""
    if _is_guided(cfg):
        return build_anatomy_guided_classifier(cfg.model)
    return build_image_classifier(cfg.model)


def _collect_targets(dataset: Dataset) -> list[int]:
    """Read the integer severity target of every item (for weighting/sampling)."""
    targets: list[int] = []
    for i in range(len(dataset)):  # type: ignore[arg-type]
        item = dataset[i]
        targets.append(int(item["target"]))
    return targets


def _find_rsna_manifest(cache_root: Path) -> Path:
    """Return the first existing manifest file under ``cache_root``.

    Raises a clear :class:`FileNotFoundError` when no manifest is present so the
    real-data path never silently falls back to synthetic data.
    """
    for name in _MANIFEST_NAMES:
        candidate = cache_root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No RSNA crop manifest found under {cache_root} "
        f"(expected one of {_MANIFEST_NAMES}). Build the crop cache first or "
        "set data.synthetic=true."
    )


def _split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return rows of ``df`` whose ``split`` column equals ``split`` (or all)."""
    if "split" not in df.columns:
        return df
    subset = df[df["split"] == split]
    if len(subset) == 0:
        return df
    return subset.reset_index(drop=True)


def build_classification_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Build ``(train_loader, val_loader)`` for severity classification.

    Uses the synthetic dataset when ``cfg.data.synthetic`` is set, otherwise the
    cached-RSNA :class:`RsnaCropDataset` (``guided=False``). A weighted sampler is
    used for the train loader when ``cfg.train.weighted_sampler`` is set (and then
    shuffling is delegated to the sampler).
    """
    from ..data.crops import read_manifest

    guided = _is_guided(cfg)
    if cfg.data.synthetic:
        train_ds: Dataset
        val_ds: Dataset
        train_ds, val_ds = make_synthetic_classification_data(
            cfg.data.synthetic_n,
            cfg.data.crop_size,
            cfg.seed,
            guided=guided,
        )
    else:
        cache_root = Path(cfg.data.rsna_cache)
        anatomy_root = Path(cfg.data.anatomy_cache) if cfg.data.anatomy_cache else None
        manifest_path = _find_rsna_manifest(cache_root)
        manifest_df = read_manifest(manifest_path)
        train_ds = RsnaCropDataset(
            _split_frame(manifest_df, "train"),
            cache_root=cache_root,
            crop_size=cfg.data.crop_size,
            use_25d=cfg.data.use_25d,
            guided=guided,
            anatomy_cache_root=anatomy_root,
        )
        val_ds = RsnaCropDataset(
            _split_frame(manifest_df, "val"),
            cache_root=cache_root,
            crop_size=cfg.data.crop_size,
            use_25d=cfg.data.use_25d,
            guided=guided,
            anatomy_cache_root=anatomy_root,
        )

    num_workers = max(0, int(cfg.data.num_workers))
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    sampler = None
    shuffle = True
    if cfg.train.weighted_sampler:
        sampler = make_weighted_sampler(_collect_targets(train_ds))
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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


def _forward_logits(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    guided: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the classifier on a batch and return ``(logits, targets)``.

    When ``guided`` the anatomy-prior channels are passed as a second positional
    argument to the anatomy-guided model; otherwise the image-only model is used.
    """
    image = batch["image"].to(device)
    level_idx = batch["level_idx"].to(device)
    condition_idx = batch["condition_idx"].to(device)
    targets = batch["target"].to(device)
    if guided:
        anatomy = batch["anatomy"].to(device)
        logits = model(image, anatomy, level_idx=level_idx, condition_idx=condition_idx)
    else:
        logits = model(image, level_idx=level_idx, condition_idx=condition_idx)
    return logits, targets


def _epoch_steps(loader: DataLoader, max_steps: int | None) -> int:
    """Number of optimizer steps to run this epoch, honoring ``max_steps``."""
    n = len(loader)
    if max_steps is not None and max_steps >= 0:
        n = min(n, int(max_steps))
    return n


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
    max_steps: int | None,
    guided: bool,
) -> float:
    """Run one training epoch; return the mean training loss over taken steps."""
    model.train()
    total_steps = _epoch_steps(loader, max_steps)
    running = 0.0
    taken = 0
    for step, batch in enumerate(loader):
        if step >= total_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits, targets = _forward_logits(model, batch, device, guided)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, targets = _forward_logits(model, batch, device, guided)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += float(loss.detach().cpu())
        taken += 1
    return running / taken if taken else 0.0


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    guided: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_true, probs, logits)`` over the whole loader."""
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    for batch in loader:
        logits, targets = _forward_logits(model, batch, device, guided)
        probs = torch.softmax(logits.float(), dim=1)
        all_logits.append(logits.float().cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    if not all_targets:
        empty_p = np.zeros((0, NUM_SEVERITY_CLASSES), dtype=np.float64)
        return np.zeros((0,), dtype=np.int64), empty_p, empty_p
    y_true = np.concatenate(all_targets).astype(np.int64)
    probs = np.concatenate(all_probs).astype(np.float64)
    logits = np.concatenate(all_logits).astype(np.float64)
    return y_true, probs, logits


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    guided: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compute the classification report on a loader; also return probs/labels."""
    y_true, probs, _ = _collect_predictions(model, loader, device, guided)
    report = classification_report_dict(y_true, probs)
    return report, y_true, probs


@torch.no_grad()
def _collect_records(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    guided: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Collect per-sample prediction records with identifiers, plus labels/logits.

    Each record carries the identifiers needed to build a finding graph
    (study_id / level / condition / side) alongside the predicted probabilities.
    """
    from ..constants import CONDITIONS, LEVELS, split_condition

    model.eval()
    records: list[dict[str, Any]] = []
    targets: list[int] = []
    logits_rows: list[np.ndarray] = []
    for batch in loader:
        logits, t = _forward_logits(model, batch, device, guided)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        logit_np = logits.float().cpu().numpy()
        study_ids = batch["study_id"]
        level_idx = batch["level_idx"].cpu().numpy()
        cond_idx = batch["condition_idx"].cpu().numpy()
        tgt = t.cpu().numpy()
        for i in range(len(probs)):
            condition = CONDITIONS[int(cond_idx[i]) % len(CONDITIONS)]
            _, side = split_condition(condition)
            records.append(
                {
                    "study_id": str(study_ids[i]),
                    "level": LEVELS[int(level_idx[i]) % len(LEVELS)],
                    "condition": condition,
                    "side": side,
                    "probs": probs[i].astype(float).tolist(),
                    "target": int(tgt[i]),
                }
            )
            targets.append(int(tgt[i]))
            logits_rows.append(logit_np[i])
    y_true = np.asarray(targets, dtype=np.int64)
    logits = (
        np.asarray(logits_rows, dtype=np.float64)
        if logits_rows
        else np.zeros((0, NUM_SEVERITY_CLASSES), dtype=np.float64)
    )
    return records, y_true, logits


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


def _monitor_value(report: dict[str, Any], monitor: str) -> float:
    """Resolve the monitored scalar from a report dict.

    ``cfg.train.monitor`` uses a ``val_`` prefix (e.g. ``val_weighted_logloss``);
    map it onto the report keys produced by ``classification_report_dict``.
    """
    key = monitor[len("val_") :] if monitor.startswith("val_") else monitor
    aliases = {
        "weighted_logloss": "weighted_logloss",
        "logloss": "weighted_logloss",
        "macro_f1": "macro_f1",
        "balanced_accuracy": "balanced_accuracy",
        "severe_recall": "severe_recall",
        "severe_fnr": "severe_fnr",
    }
    key = aliases.get(key, key)
    if key not in report:
        raise KeyError(f"monitor {monitor!r} -> {key!r} not in report keys {sorted(report)}")
    value = float(report[key])
    if np.isnan(value):
        # NaN is unorderable; map to the worst possible value for the metric.
        return float("inf")
    return value


def train_classifier(
    cfg: Config,
    run_dir: str | Path,
    *,
    json_logs: bool = False,
) -> dict[str, Any]:
    """Train the severity classifier and return a results dict.

    Returns ``{"best": {...val metrics...}, "checkpoint": str, "history": [...]}``.
    Writes ``best.pt`` / ``last.pt`` checkpoints plus ``metrics.json`` and
    ``config.json`` into ``run_dir``.
    """
    seed_everything(cfg.seed)
    out_dir = ensure_dir(run_dir)
    device = select_device(cfg.train.device)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"

    guided = _is_guided(cfg)
    train_loader, val_loader = build_classification_loaders(cfg)

    model = _build_model(cfg).to(device)

    class_weights = None
    if cfg.train.class_weighted_loss:
        class_weights = build_class_weights(_collect_targets(train_loader.dataset))
    loss_fn = build_classification_loss(cfg.train, class_weights=class_weights)

    freeze_epochs = max(0, int(cfg.train.freeze_backbone_epochs))
    backbone_frozen = freeze_epochs > 0
    model.set_backbone_trainable(not backbone_frozen)

    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    steps_per_epoch = _epoch_steps(train_loader, cfg.train.max_steps)
    scheduler = build_scheduler(optimizer, cfg.train.epochs, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    stopper = EarlyStopping(
        patience=cfg.train.early_stop_patience,
        mode=cfg.train.monitor_mode,
        min_delta=0.0,
    )

    history: list[dict[str, Any]] = []
    best_report: dict[str, Any] = {}
    best_monitor: float | None = None

    for epoch in range(int(cfg.train.epochs)):
        if backbone_frozen and epoch >= freeze_epochs:
            # Unfreeze and rebuild the optimizer/scheduler so the newly trainable
            # backbone parameters are optimized for the remaining epochs.
            model.set_backbone_trainable(True)
            backbone_frozen = False
            optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
            remaining = max(1, int(cfg.train.epochs) - epoch)
            scheduler = build_scheduler(optimizer, remaining, steps_per_epoch)

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
            guided,
        )
        report, _, _ = _evaluate(model, val_loader, device, guided)
        monitor_value = _monitor_value(report, cfg.train.monitor)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "backbone_frozen": backbone_frozen,
            "monitor": cfg.train.monitor,
            "monitor_value": monitor_value,
            **{f"val_{k}": v for k, v in report.items()},
        }
        history.append(epoch_record)
        if json_logs:
            emit_json({"event": "epoch_end", **epoch_record})
        else:
            log.info(
                "epoch %d train_loss=%.4f %s=%.4f",
                epoch,
                train_loss,
                cfg.train.monitor,
                monitor_value,
            )

        improved = stopper.step(monitor_value)
        if improved or best_monitor is None:
            best_monitor = monitor_value
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
        "best_monitor": best_monitor,
        "monitor": cfg.train.monitor,
        "monitor_mode": cfg.train.monitor_mode,
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


def evaluate_classifier(
    cfg: Config,
    run_dir: str | Path,
    split: str = "val",
) -> dict[str, Any]:
    """Load ``best.pt`` from ``run_dir`` and evaluate on ``split``.

    Returns the classification report dict plus the checkpoint path and saves the
    raw probabilities / labels to ``run_dir/{split}_probs.npz`` for downstream
    calibration / finding-graph use.
    """
    seed_everything(cfg.seed)
    out_dir = Path(run_dir)
    ckpt_path = out_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No best.pt checkpoint found in {out_dir}")

    device = select_device(cfg.train.device)
    guided = _is_guided(cfg)
    model = _build_model(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])

    train_loader, val_loader = build_classification_loaders(cfg)
    loader = train_loader if split == "train" else val_loader

    records, y_true, logits = _collect_records(model, loader, device, guided)
    probs = (
        np.asarray([r["probs"] for r in records], dtype=np.float64)
        if records
        else np.zeros((0, NUM_SEVERITY_CLASSES), dtype=np.float64)
    )
    report = classification_report_dict(y_true, probs)

    # Calibration: fit a temperature on this split (research demo) and attach
    # calibrated probabilities to each record for the finding graph.
    from ..evaluation.calibration import TemperatureScaler, calibration_report

    calib: dict[str, Any] = {}
    if len(logits) > 0:
        scaler = TemperatureScaler().fit(logits, y_true)
        cal_probs = scaler.transform(logits)
        for rec, cp in zip(records, cal_probs, strict=False):
            rec["calibrated_probs"] = np.asarray(cp, dtype=float).tolist()
        calib = calibration_report(y_true, probs, logits=logits)
        calib["temperature"] = float(scaler.temperature)

    np.savez(out_dir / f"{split}_probs.npz", y_true=y_true, probs=probs, logits=logits)
    predictions_doc = {
        "dataset_source": "synthetic" if cfg.data.synthetic else "rsna",
        "run_id": out_dir.name,
        "split": split,
        "model_kind": cfg.model.kind,
        "predictions": records,
    }
    with (out_dir / "predictions.json").open("w", encoding="utf-8") as fh:
        json.dump(predictions_doc, fh, indent=2, sort_keys=True, default=str)
    with (out_dir / "calibration.json").open("w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2, sort_keys=True, default=str)

    result = dict(report)
    result["split"] = split
    result["checkpoint"] = str(ckpt_path)
    result["probs_path"] = str(out_dir / f"{split}_probs.npz")
    result["predictions_path"] = str(out_dir / "predictions.json")
    if calib:
        result["ece"] = calib.get("ece")
        result["ece_after"] = calib.get("ece_after")
        result["temperature"] = calib.get("temperature")
    return result
