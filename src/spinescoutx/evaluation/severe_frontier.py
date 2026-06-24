"""Severe-first operating frontier (Phase 7).

For a safety-critical screening setting the question is not "what is the accuracy"
but "how much severe recall can I buy, and at what false-alarm cost?". This module
collects per-node ``P(severe)`` from any of the crop classifiers (E0/E1/E2) or the
E3 graph reasoner on the **same** canal-stenosis validation nodes, then sweeps the
severe decision threshold to trace, per model:

  * severe recall (sensitivity) — the safety axis,
  * severe precision and false-alarm rate (= 1 − specificity) — the cost axis,
  * the operating threshold that first reaches a target recall (e.g. 0.90).

A model Pareto-dominates another when it gives more severe recall at the same or
lower false-alarm rate. Research-only. Not diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import LEVELS, SEVERE_INDEX


def sweep_severe_threshold(
    y_true: np.ndarray, p_severe: np.ndarray, thresholds: np.ndarray | None = None
) -> list[dict[str, float]]:
    """Trace severe recall / precision / false-alarm rate over a threshold grid."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_severe).astype(float)
    is_sev = y == SEVERE_INDEX
    n_pos = int(is_sev.sum())
    n_neg = int((~is_sev).sum())
    if thresholds is None:
        thresholds = np.unique(np.concatenate([[0.0], np.linspace(0.01, 0.99, 99), [1.0]]))
    rows: list[dict[str, float]] = []
    for t in thresholds:
        alarm = p >= t
        tp = int((alarm & is_sev).sum())
        fp = int((alarm & ~is_sev).sum())
        recall = tp / n_pos if n_pos else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        far = fp / n_neg if n_neg else 0.0
        rows.append(
            {
                "threshold": float(t),
                "severe_recall": recall,
                "severe_precision": precision,
                "false_alarm_rate": far,
                "alarm_fraction": float(alarm.mean()),
            }
        )
    return rows


def recall_at_far_budget(sweep: list[dict[str, float]], far_budget: float) -> dict[str, float]:
    """Best achievable severe recall whose false-alarm rate ≤ ``far_budget``."""
    feasible = [r for r in sweep if r["false_alarm_rate"] <= far_budget]
    if not feasible:
        return {"severe_recall": 0.0, "threshold": 1.0, "false_alarm_rate": 0.0}
    best = max(feasible, key=lambda r: r["severe_recall"])
    return {
        "severe_recall": best["severe_recall"],
        "threshold": best["threshold"],
        "false_alarm_rate": best["false_alarm_rate"],
        "severe_precision": best["severe_precision"],
    }


def threshold_for_recall(sweep: list[dict[str, float]], target_recall: float) -> dict[str, float]:
    """Lowest-cost operating point that first reaches ``target_recall`` severe recall."""
    feasible = [r for r in sweep if r["severe_recall"] >= target_recall]
    if not feasible:
        return {"reached": False, "target_recall": target_recall}
    best = min(feasible, key=lambda r: r["false_alarm_rate"])
    return {"reached": True, "target_recall": target_recall, **best}


def severe_auroc_pr(y_true: np.ndarray, p_severe: np.ndarray) -> dict[str, float]:
    """Threshold-free severe one-vs-rest AUROC + average precision."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = (np.asarray(y_true).astype(int) == SEVERE_INDEX).astype(int)
    p = np.asarray(p_severe).astype(float)
    if y.sum() == 0 or y.sum() == len(y):
        return {"severe_auroc": float("nan"), "severe_ap": float("nan")}
    return {
        "severe_auroc": float(roc_auc_score(y, p)),
        "severe_ap": float(average_precision_score(y, p)),
    }


# --------------------------------------------------------------------------- #
# per-node prediction collectors (aligned by study|level key)
# --------------------------------------------------------------------------- #
def _key(study: str, level: str) -> str:
    return f"{study}|{level}"


def collect_crop_classifier(
    run_dir: str | Path, rsna_cache: str | Path, anatomy_cache: str | Path | None, device
) -> dict[str, tuple[int, float]]:
    """Per-node ``key -> (y_true, P(severe))`` for an E0/E1/E2 crop classifier."""
    import torch
    from torch.utils.data import DataLoader

    from ..config import config_from_dict
    from ..data.crops import read_manifest
    from ..data.datasets import RsnaCropDataset
    from ..training.train_classifier import _build_model, _is_guided

    run_dir = Path(run_dir)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    guided = _is_guided(cfg)
    model = _build_model(cfg).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["state_dict"])

    man = read_manifest(Path(rsna_cache) / "manifest.parquet")
    val = man[(man["condition"] == "spinal_canal_stenosis") & (man["split"] == "val")]
    val = val[val["severity_index"].isin([0, 1, 2])].reset_index(drop=True)
    ds = RsnaCropDataset(
        val,
        rsna_cache,
        crop_size=cfg.data.crop_size,
        use_25d=cfg.data.use_25d,
        guided=guided,
        anatomy_cache_root=str(anatomy_cache) if guided else None,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    out: dict[str, tuple[int, float]] = {}
    idx = 0
    keys = [_key(str(r.study_id), str(r.level)) for r in val.itertuples()]
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            level_idx = batch["level_idx"].to(device)
            condition_idx = batch["condition_idx"].to(device)
            if guided:
                logits = model(
                    image,
                    batch["anatomy"].to(device),
                    level_idx=level_idx,
                    condition_idx=condition_idx,
                )
            else:
                logits = model(image, level_idx=level_idx, condition_idx=condition_idx)
            probs = torch.softmax(logits.float(), dim=1)[:, SEVERE_INDEX].cpu().numpy()
            tgt = batch["target"].numpy()
            for j in range(len(probs)):
                out[keys[idx]] = (int(tgt[j]), float(probs[j]))
                idx += 1
    return out


def collect_multiview(run_dir: str | Path, device) -> dict[str, tuple[int, float]]:
    """Per-node ``key -> (y_true, P(severe))`` for the E3 graph reasoner."""
    import torch

    from ..config import config_from_dict
    from ..data.study_dataset import build_study_loaders
    from ..models.multiview_graph import build_multiview_graph

    run_dir = Path(run_dir)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    model = build_multiview_graph(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["state_dict"])

    _, val_loader, _ = build_study_loaders(
        Path(cfg.data.rsna_cache) / "manifest.parquet",
        cfg.data.rsna_cache,
        cfg.data.anatomy_cache,
        crop_size=cfg.data.crop_size,
        batch_size=cfg.train.batch_size,
        num_workers=4,
    )
    out: dict[str, tuple[int, float]] = {}
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["images"].to(device),
                batch["anatomy"].to(device),
                batch["morph"].to(device),
                batch["level_idx"].to(device),
                batch["mask"].to(device),
            )
            probs = torch.softmax(logits.float(), dim=2)[..., SEVERE_INDEX].cpu().numpy()
            targets = batch["target"].numpy()
            mask = batch["mask"].numpy()
            for b, study in enumerate(batch["study_id"]):
                for li in range(len(LEVELS)):
                    if mask[b, li]:
                        out[_key(str(study), LEVELS[li])] = (
                            int(targets[b, li]),
                            float(probs[b, li]),
                        )
    return out


def build_frontier(
    models: dict[str, dict[str, tuple[int, float]]],
    *,
    far_budgets: tuple[float, ...] = (0.05, 0.10, 0.20),
    target_recalls: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> dict[str, Any]:
    """Align all models on shared nodes and build the severe operating frontier."""
    shared = set.intersection(*[set(m) for m in models.values()]) if models else set()
    keys = sorted(shared)
    result: dict[str, Any] = {"n_shared_nodes": len(keys), "models": {}}
    for name, preds in models.items():
        y = np.array([preds[k][0] for k in keys])
        p = np.array([preds[k][1] for k in keys])
        sweep = sweep_severe_threshold(y, p)
        result["models"][name] = {
            "n_severe": int((y == SEVERE_INDEX).sum()),
            **severe_auroc_pr(y, p),
            "recall_at_far": {f"far<={b}": recall_at_far_budget(sweep, b) for b in far_budgets},
            "threshold_for_recall": {
                f"recall>={r}": threshold_for_recall(sweep, r) for r in target_recalls
            },
            "sweep": sweep,
        }
    return result
