"""Robust auto-inference training (Phase 3/4).

Trains an E0-architecture severity grader so that **training matches inference**:
the train crops carry the localizer's in-plane offset distribution (via
:class:`~spinescoutx.data.robust_crops.RobustCanalCropDataset`) and/or come from the
real auto-localized pipeline, optionally with a crop/slice **consistency loss**.
Model selection and the headline metrics are computed on the **auto** distribution
(the real inference target), with bootstrap CIs.

Reuses the validated E0 building blocks (`_build_model`, loss/optim/scheduler,
`_evaluate`, early stopping); only the data path and (optional) consistency term are
new. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import Config, config_from_dict
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from .optim import EarlyStopping, build_optimizer, build_scheduler, select_device
from .train_classifier import (
    _build_model,
    _checkpoint_payload,
    _evaluate,
    _forward_logits,
    _monitor_value,
)

log = get_logger()


def build_e0_like_config(
    e0_config_path: str | Path,
    *,
    name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    output_root: str = "runs",
    loss: str | None = None,
) -> Config:
    """Load the E0 config and override only the training-schedule / naming fields.

    ``loss`` (e.g. ``"cost_sensitive"``) overrides the classification loss when set.
    """
    d = json.loads(Path(e0_config_path).read_text())
    d["name"] = name
    d["output_root"] = output_root
    d["train"] = {
        **d["train"],
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
    }
    if loss is not None:
        d["train"]["loss"] = str(loss)
    return config_from_dict(d)


def _consistency_loss(
    logits1: torch.Tensor, logits2: torch.Tensor, targets: torch.Tensor, severe_mult: float
) -> torch.Tensor:
    """Symmetric-KL consistency between two jittered views, optionally up-weighting
    the moderate/severe classes (where stability matters most)."""
    lp1 = F.log_softmax(logits1, dim=1)
    lp2 = F.log_softmax(logits2, dim=1)
    p1, p2 = lp1.exp(), lp2.exp()
    kl = 0.5 * (
        F.kl_div(lp1, p2, reduction="none").sum(1) + F.kl_div(lp2, p1, reduction="none").sum(1)
    )
    if severe_mult != 1.0:
        w = torch.ones_like(kl)
        w[targets >= 1] = float(severe_mult)
        kl = kl * w
    return kl.mean()


def _train_epoch_robust(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    device: torch.device,
    use_amp: bool,
    *,
    consistency_weight: float,
    severe_consistency_mult: float,
) -> dict[str, float]:
    """One epoch; supports an optional two-view consistency term (when batches carry
    ``image2``). Returns mean total / ce / consistency losses."""
    model.train()
    run_total = run_ce = run_cons = 0.0
    taken = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, targets = _forward_logits(model, batch, device, guided=False)
            ce = loss_fn(logits, targets)
            cons = torch.zeros((), device=device)
            if consistency_weight > 0.0 and "image2" in batch:
                view2 = {
                    "image": batch["image2"],
                    "level_idx": batch["level_idx"],
                    "condition_idx": batch["condition_idx"],
                    "target": batch["target"],
                }
                logits2, _ = _forward_logits(model, view2, device, guided=False)
                cons = _consistency_loss(logits, logits2, targets, severe_consistency_mult)
            loss = ce + consistency_weight * cons
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        run_total += float(loss.detach().cpu())
        run_ce += float(ce.detach().cpu())
        run_cons += float(cons.detach().cpu()) if torch.is_tensor(cons) else 0.0
        taken += 1
    n = max(taken, 1)
    return {"loss": run_total / n, "ce": run_ce / n, "consistency": run_cons / n}


def train_robust_variant(
    *,
    variant: str,
    train_dataset: Dataset,
    train_targets: list[int],
    auto_val_loader: DataLoader,
    e0_config_path: str | Path,
    run_dir: str | Path,
    epochs: int = 18,
    lr: float = 3e-4,
    batch_size: int = 32,
    consistency_weight: float = 0.0,
    severe_consistency_mult: float = 1.0,
    monitor: str = "val_severe_aware",
    severe_aware_lambda: float = 0.20,
    num_workers: int = 0,
    device: str = "auto",
    seed: int = 1337,
    loss: str | None = None,
) -> dict[str, Any]:
    """Train one robust variant; select on the AUTO val set; save run artifacts.

    Returns ``{"best": <auto-val report>, "history": [...], "run_dir": ...}``.
    The val loader MUST be the auto-distribution loader (real inference target).
    """
    from ..data.datasets import build_class_weights
    from ..training.losses import build_classification_loss

    seed_everything(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device_t = select_device(device)
    use_amp = device_t.type == "cuda"

    cfg = build_e0_like_config(
        e0_config_path, name=variant, epochs=epochs, lr=lr, batch_size=batch_size, loss=loss
    )
    model = _build_model(cfg).to(device_t)

    class_weights = build_class_weights(train_targets)
    loss_fn = build_classification_loss(cfg.train, class_weights=class_weights)

    freeze_epochs = max(0, int(cfg.train.freeze_backbone_epochs))
    backbone_frozen = freeze_epochs > 0
    model.set_backbone_trainable(not backbone_frozen)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, epochs, len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience, mode="min", min_delta=0.0)

    history: list[dict[str, Any]] = []
    best_report: dict[str, Any] = {}
    best_monitor: float | None = None

    for epoch in range(epochs):
        if backbone_frozen and epoch >= freeze_epochs:
            model.set_backbone_trainable(True)
            backbone_frozen = False
            ft_lr = cfg.train.lr * cfg.train.backbone_unfreeze_lr_scale
            optimizer = build_optimizer(model, ft_lr, cfg.train.weight_decay)
            scheduler = build_scheduler(optimizer, max(1, epochs - epoch), len(train_loader))

        losses = _train_epoch_robust(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            scaler,
            device_t,
            use_amp,
            consistency_weight=consistency_weight,
            severe_consistency_mult=severe_consistency_mult,
        )
        report, _, _ = _evaluate(model, auto_val_loader, device_t, guided=False)
        monitor_value = _monitor_value(report, monitor, severe_aware_lambda)
        rec = {
            "epoch": epoch,
            **losses,
            "monitor": monitor,
            "monitor_value": monitor_value,
            **{f"auto_{k}": v for k, v in report.items() if k != "confusion"},
        }
        history.append(rec)
        log.info(
            "[%s] epoch %d loss=%.4f cons=%.4f auto_sevR=%.3f auto_wll=%.3f mon=%.4f",
            variant,
            epoch,
            losses["loss"],
            losses["consistency"],
            report["severe_recall"],
            report["weighted_logloss"],
            monitor_value,
        )
        improved = stopper.step(monitor_value)
        if improved or best_monitor is None:
            best_monitor = monitor_value
            best_report = dict(report)
            torch.save(_checkpoint_payload(model, cfg, best_report), run_dir / "best.pt")
        torch.save(_checkpoint_payload(model, cfg, dict(report)), run_dir / "last.pt")
        if stopper.should_stop:
            log.info("[%s] early stop at epoch %d", variant, epoch)
            break

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    metrics = {
        "variant": variant,
        "best_auto_val": best_report,
        "best_monitor": best_monitor,
        "monitor": monitor,
        "severe_aware_lambda": severe_aware_lambda,
        "consistency_weight": consistency_weight,
        "severe_consistency_mult": severe_consistency_mult,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return {"best": best_report, "history": history, "run_dir": str(run_dir)}
