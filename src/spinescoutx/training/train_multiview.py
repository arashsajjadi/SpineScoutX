"""Training loop for E3, the study-level multi-view anatomy-graph reasoner.

Trains :class:`MultiViewAnatomyGraphClassifier` on study-grouped canal-stenosis
nodes. The loss is a masked, severity-weighted cross-entropy over per-level logits
(absent levels carry ``IGNORE_INDEX`` and contribute nothing). Evaluation pools the
present-level predictions and reports the same metrics as the crop classifiers, so
E3 is directly comparable to E0/E1/E2.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..data.study_dataset import IGNORE_INDEX, build_study_loaders
from ..evaluation.calibration import expected_calibration_error
from ..evaluation.classification_metrics import classification_report_dict
from ..models.multiview_graph import build_multiview_graph
from ..training.losses import severity_class_weights
from ..training.optim import build_optimizer, build_scheduler, select_device
from ..utils.logging import emit_json, get_logger
from ..utils.paths import ensure_dir

log = get_logger()


def _flatten_present(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``(B, L, C)`` logits / ``(B, L)`` targets to present-node rows."""
    b, n, c = logits.shape
    flat_logits = logits.reshape(b * n, c)
    flat_targets = targets.reshape(b * n)
    keep = mask.reshape(b * n)
    return flat_logits[keep], flat_targets[keep]


@torch.no_grad()
def _evaluate(model: nn.Module, loader, device) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for batch in loader:
        logits = model(
            batch["images"].to(device),
            batch["anatomy"].to(device),
            batch["morph"].to(device),
            batch["level_idx"].to(device),
            batch["mask"].to(device),
        )
        kl, kt = _flatten_present(logits, batch["target"].to(device), batch["mask"].to(device))
        if kt.numel() == 0:
            continue
        ps.append(torch.softmax(kl.float(), dim=1).cpu().numpy())
        ys.append(kt.cpu().numpy())
    if not ys:
        return {"weighted_logloss": 0.0, "severe_recall": 0.0}, np.zeros((0,)), np.zeros((0, 3))
    y = np.concatenate(ys).astype(np.int64)
    p = np.concatenate(ps).astype(np.float64)
    report = classification_report_dict(y, p)
    report["ece"] = float(expected_calibration_error(y, p))
    return report, y, p


def _train_targets(train_ds) -> list[int]:
    out: list[int] = []
    for i in range(len(train_ds)):
        out.extend(train_ds.study_targets(i))
    return out


def train_multiview(cfg: Config, run_dir: str | Path, *, json_logs: bool = False) -> dict[str, Any]:
    """Train E3; save best/last/metrics/config; return the best report dict."""
    run_dir = ensure_dir(run_dir)
    device = select_device(cfg.train.device)

    manifest = Path(cfg.data.rsna_cache) / "manifest.parquet"
    train_loader, val_loader, train_ds = build_study_loaders(
        manifest,
        cfg.data.rsna_cache,
        cfg.data.anatomy_cache,
        crop_size=cfg.data.crop_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
    )
    log.info("E3 studies: train=%d val=%d", len(train_loader.dataset), len(val_loader.dataset))

    # severity class weights from the present train nodes
    counts = np.bincount(_train_targets(train_ds), minlength=3)[:3]
    weights = severity_class_weights([int(c) for c in counts]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    log.info("E3 train node severity counts: %s", counts.tolist())

    model = build_multiview_graph(cfg.model).to(device)
    frozen = cfg.train.freeze_backbone_epochs > 0
    model.set_backbone_trainable(not frozen)

    steps_per_epoch = max(1, len(train_loader))
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.train.epochs, steps_per_epoch)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_value = float("inf")
    best_report: dict[str, Any] = {}
    monitor = (
        cfg.train.monitor[len("val_") :]
        if cfg.train.monitor.startswith("val_")
        else "weighted_logloss"
    )
    patience, bad = cfg.train.early_stop_patience, 0

    for epoch in range(cfg.train.epochs):
        if frozen and epoch == cfg.train.freeze_backbone_epochs:
            model.set_backbone_trainable(True)
            frozen = False
            remaining = max(1, cfg.train.epochs - epoch)
            finetune_lr = cfg.train.lr * cfg.train.backbone_unfreeze_lr_scale
            optimizer = build_optimizer(model, finetune_lr, cfg.train.weight_decay)
            scheduler = build_scheduler(optimizer, remaining, steps_per_epoch)
            log.info("E3 unfroze backbone at epoch %d (lr=%.2e)", epoch, finetune_lr)

        model.train()
        running, taken = 0.0, 0
        for batch in train_loader:
            if cfg.train.max_steps is not None and taken >= cfg.train.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            images = batch["images"].to(device)
            anatomy = batch["anatomy"].to(device)
            morph = batch["morph"].to(device)
            level_idx = batch["level_idx"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images, anatomy, morph, level_idx, mask)
                    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images, anatomy, morph, level_idx, mask)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += float(loss.detach().cpu())
            taken += 1

        report, _, _ = _evaluate(model, val_loader, device)
        value = float(report.get(monitor, report.get("weighted_logloss", 0.0)))
        if cfg.train.monitor_mode == "max":
            value = -value
        train_loss = running / taken if taken else 0.0
        log.info(
            "E3 epoch %d train_loss=%.4f val_wll=%.4f severe_recall=%.4f auroc=%.4f",
            epoch,
            train_loss,
            report.get("weighted_logloss", 0.0),
            report.get("severe_recall", 0.0),
            report.get("severe_auroc", float("nan")),
        )
        if json_logs:
            scalar = {f"val_{k}": v for k, v in report.items() if isinstance(v, (int, float))}
            emit_json({"event": "epoch", "epoch": epoch, "train_loss": train_loss, **scalar})

        payload = {
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "metrics": report,
            "model_kind": "multiview_graph",
        }
        torch.save(payload, run_dir / "last.pt")
        if value < best_value:
            best_value, best_report, bad = value, dict(report), 0
            torch.save(payload, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                log.info("E3 early stop at epoch %d", epoch)
                break

    (run_dir / "metrics.json").write_text(json.dumps({"best": best_report}, indent=2, default=str))
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    return best_report
