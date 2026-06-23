"""Anatomy-prior ablation for the anatomy-guided severity classifier.

This module evaluates how much the trained anatomy-guided classifier relies on
its anatomy-prior input by re-running validation while perturbing the anatomy
tensor in controlled, deterministic ways:

- ``correct``  : the anatomy prior is passed through unchanged.
- ``zero``     : the anatomy prior is replaced by zeros.
- ``shuffled`` : each sample receives ANOTHER sample's anatomy (batch roll).
- ``noise``    : the anatomy prior is replaced by deterministic uniform noise.

For each mode we report the full :func:`classification_report_dict` and, when
heatmaps can be produced, an optional mean Anatomical Evidence Consistency
(``aec_mean``) computed against the (correct) anatomy prior. :func:`compare_ablations`
summarises how key metrics shift relative to the ``correct`` baseline.

Research-only — not diagnostic. These numbers describe model behaviour under
input perturbation; they are not a clinical signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..constants import SEVERE_INDEX
from ..data.synthetic import SyntheticCropDataset
from ..models.anatomy_guided_classifier import (
    AnatomyGuidedClassifier,
    build_anatomy_guided_classifier,
)
from ..training.optim import select_device
from ..utils.logging import emit_json, get_logger
from ..utils.paths import ensure_dir, run_dir
from ..utils.seed import seed_everything
from .classification_metrics import classification_report_dict
from .evidence_metrics import anatomical_evidence_consistency

log = get_logger()

#: Ablation modes understood by :func:`perturb_anatomy`.
ABLATION_MODES: tuple[str, ...] = ("correct", "zero", "shuffled", "noise")

#: Fixed, nonzero batch-roll offset for the ``shuffled`` mode.
_SHUFFLE_OFFSET: int = 1


def perturb_anatomy(
    anatomy: torch.Tensor,
    mode: str,
    *,
    seed: int = 1337,
) -> torch.Tensor:
    """Return a perturbed copy of an anatomy-prior batch.

    Parameters
    ----------
    anatomy:
        Anatomy-prior tensor of shape ``(B, C, H, W)``.
    mode:
        One of ``"correct"``, ``"zero"``, ``"shuffled"``, ``"noise"``.
    seed:
        Seed for the deterministic generator used by ``"noise"``.

    Modes
    -----
    - ``correct``  : identity (returns ``anatomy`` unchanged).
    - ``zero``     : ``torch.zeros_like(anatomy)``.
    - ``shuffled`` : roll along the batch dim by a fixed nonzero offset so each
      sample receives another sample's anatomy. With a single-sample batch the
      roll is a no-op, which is unavoidable and harmless.
    - ``noise``    : ``torch.rand_like(anatomy)`` drawn from a seeded generator.
    """
    if anatomy.dim() != 4:
        raise ValueError(f"anatomy must be 4D (B, C, H, W), got shape {tuple(anatomy.shape)}")
    if mode == "correct":
        return anatomy
    if mode == "zero":
        return torch.zeros_like(anatomy)
    if mode == "shuffled":
        return torch.roll(anatomy, shifts=_SHUFFLE_OFFSET, dims=0)
    if mode == "noise":
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.rand(
            anatomy.shape,
            generator=generator,
            dtype=anatomy.dtype,
        )
        return noise.to(anatomy.device)
    raise ValueError(f"Unknown ablation mode: {mode!r} (expected one of {ABLATION_MODES}).")


def _resolve_modes(cfg: Config) -> list[str]:
    """Return the list of ablation modes to evaluate, defaulting to all modes."""
    requested = cfg.ablation.get("modes") if isinstance(cfg.ablation, dict) else None
    if not requested:
        return list(ABLATION_MODES)
    modes: list[str] = []
    for mode in requested:
        if mode not in ABLATION_MODES:
            raise ValueError(
                f"Unknown ablation mode in config: {mode!r} "
                f"(expected a subset of {ABLATION_MODES})."
            )
        modes.append(str(mode))
    return modes


def _build_synthetic_val_loader(cfg: Config) -> DataLoader:
    """Build a deterministic guided synthetic validation loader."""
    n = max(8, int(cfg.data.synthetic_n) // 4)
    dataset: Dataset = SyntheticCropDataset(
        n=n,
        crop_size=int(cfg.data.crop_size),
        seed=int(cfg.data.split_seed) + 100_003,
        guided=True,
        study_prefix="synthstudy_ablate_val",
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=0,
    )


def _load_checkpoint_model(
    cfg: Config,
    checkpoint_path: Path,
    device: torch.device,
) -> AnatomyGuidedClassifier:
    """Load a trained anatomy-guided model from ``checkpoint_path``."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Ablation checkpoint not found: {checkpoint_path}. "
            "Set cfg.ablation['run'] to a directory containing best.pt, or use "
            "synthetic data (cfg.data.synthetic = true)."
        )
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing a 'state_dict' entry.")
    model = build_anatomy_guided_classifier(cfg.model)
    model.load_state_dict(payload["state_dict"])
    return model


def _prepare_model(
    cfg: Config,
    device: torch.device,
) -> tuple[AnatomyGuidedClassifier, DataLoader]:
    """Build (or load) the model and the validation loader for the ablation."""
    if cfg.data.synthetic:
        model = build_anatomy_guided_classifier(cfg.model)
        loader = _build_synthetic_val_loader(cfg)
        return model.to(device), loader

    ablation_cfg = cfg.ablation if isinstance(cfg.ablation, dict) else {}
    run_name = ablation_cfg.get("run")
    if not run_name:
        raise ValueError(
            "cfg.ablation['run'] must name a run directory when cfg.data.synthetic is False."
        )
    # Accept either a full run path ("runs/e1_...") or a bare run id ("e1_...").
    resolved = Path(str(run_name))
    if not resolved.exists():
        resolved = run_dir(cfg.output_root, str(run_name))
    checkpoint_path = resolved / "best.pt"
    model = _load_checkpoint_model(cfg, checkpoint_path, device)

    # Real-data guided validation loaders live in data.datasets; importing them
    # lazily keeps this module usable in the synthetic-only smoke path.
    loader = _build_real_val_loader(cfg)
    return model.to(device), loader


def _build_real_val_loader(cfg: Config) -> DataLoader:
    """Build a guided validation loader from cached RSNA crops (real-data path)."""
    from ..data.crops import read_manifest
    from ..data.datasets import RsnaCropDataset

    manifest_path = Path(cfg.data.rsna_cache) / "manifest.parquet"
    if not manifest_path.exists():
        manifest_path = Path(cfg.data.rsna_cache) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No cached RSNA manifest found under {cfg.data.rsna_cache}. "
            "Build the crop cache first or use synthetic data."
        )
    manifest = read_manifest(manifest_path)
    val_manifest = manifest[manifest["split"] == "val"]
    if "severity_index" in val_manifest.columns:  # drop unlabeled crops
        val_manifest = val_manifest[val_manifest["severity_index"] >= 0]
    val_manifest = val_manifest.reset_index(drop=True)
    anatomy_root = cfg.data.anatomy_cache or cfg.data.spider_cache
    dataset = RsnaCropDataset(
        manifest_df=val_manifest,
        cache_root=cfg.data.rsna_cache,
        crop_size=int(cfg.data.crop_size),
        use_25d=bool(cfg.data.use_25d),
        guided=True,
        anatomy_cache_root=anatomy_root,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=max(0, int(cfg.data.num_workers)),
    )


def _aec_for_batch(
    model: AnatomyGuidedClassifier,
    image: torch.Tensor,
    anatomy: torch.Tensor,
    level_idx: torch.Tensor,
    condition_idx: torch.Tensor,
    preds: torch.Tensor,
) -> list[float]:
    """Compute per-sample AEC of Grad-CAM heatmaps against the anatomy prior.

    Returns the defined (non-``None``) AEC values for the batch. Heatmap
    generation is best-effort; any failure is logged and yields an empty list so
    the ablation never crashes on the optional evidence metric.
    """
    from ..viz.heatmaps import make_heatmap

    values: list[float] = []
    region = anatomy.detach().cpu().numpy()
    region_mask = region.sum(axis=1) > 0.0  # (B, H, W)
    for i in range(image.shape[0]):
        sample = {
            "image": image[i],
            "anatomy": anatomy[i],
            "level_idx": level_idx[i],
            "condition_idx": condition_idx[i],
        }
        try:
            heatmap = make_heatmap(model, sample, class_idx=int(preds[i].item()))
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            log.warning("Skipping AEC for one sample (heatmap failed): %s", exc)
            continue
        aec = anatomical_evidence_consistency(heatmap, region_mask[i])
        if aec is not None:
            values.append(float(aec))
    return values


def _evaluate_mode(
    model: AnatomyGuidedClassifier,
    loader: DataLoader,
    mode: str,
    device: torch.device,
    *,
    seed: int,
    compute_aec: bool,
    aec_max_samples: int | None = 800,
) -> dict[str, Any]:
    """Evaluate the model over ``loader`` with the anatomy prior perturbed.

    Classification metrics use the full loader; AEC (which needs a Grad-CAM pass
    per sample) is estimated on at most ``aec_max_samples`` samples for speed.
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_conditions: list[np.ndarray] = []
    all_levels: list[np.ndarray] = []
    aec_values: list[float] = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device).float()
            anatomy = batch["anatomy"].to(device).float()
            level_idx = batch["level_idx"].to(device).long()
            condition_idx = batch["condition_idx"].to(device).long()
            target = batch["target"].long()

            perturbed = perturb_anatomy(anatomy, mode, seed=seed)
            logits = model(image, perturbed, level_idx, condition_idx)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            all_conditions.append(condition_idx.detach().cpu().numpy())
            all_levels.append(level_idx.detach().cpu().numpy())

    if compute_aec:
        # AEC is always scored against the CORRECT anatomy prior (the region of
        # interest) regardless of which prior the model saw. It needs gradients,
        # so it runs outside the no_grad block.
        for batch in loader:
            if aec_max_samples is not None and len(aec_values) >= aec_max_samples:
                break
            image = batch["image"].to(device).float()
            anatomy = batch["anatomy"].to(device).float()
            level_idx = batch["level_idx"].to(device).long()
            condition_idx = batch["condition_idx"].to(device).long()
            perturbed = perturb_anatomy(anatomy, mode, seed=seed)
            with torch.no_grad():
                preds = model(image, perturbed, level_idx, condition_idx).argmax(dim=1)
            aec_values.extend(
                _aec_for_batch(model, image, anatomy, level_idx, condition_idx, preds)
            )

    y_true = np.concatenate(all_targets) if all_targets else np.zeros((0,), dtype=np.int64)
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 3), dtype=np.float64)
    conditions = np.concatenate(all_conditions) if all_conditions else None
    levels = np.concatenate(all_levels) if all_levels else None

    report = classification_report_dict(
        y_true,
        probs,
        conditions=conditions,
        levels=levels,
    )
    if compute_aec and aec_values:
        report["aec_mean"] = float(np.mean(aec_values))
    return report


def run_ablation(
    cfg: Config,
    run_dir: str | Path,  # noqa: A002 - mirrors sibling module signatures
    *,
    json_logs: bool = False,
) -> dict[str, Any]:
    """Run the anatomy-prior ablation and write ``run_dir/ablation.json``.

    Builds a synthetic anatomy-guided model + guided synthetic validation loader
    when ``cfg.data.synthetic``; otherwise loads ``cfg.ablation['run']/best.pt``
    and a cached guided validation loader. The model is evaluated under each mode
    in ``cfg.ablation['modes']`` (all modes if unspecified). Returns a mapping
    ``{mode: classification_report_dict (+ optional aec_mean)}``.
    """
    seed_everything(cfg.seed)
    device = select_device(cfg.train.device)

    out_dir = ensure_dir(run_dir)
    modes = _resolve_modes(cfg)
    model, loader = _prepare_model(cfg, device)

    ablation_cfg = cfg.ablation if isinstance(cfg.ablation, dict) else {}
    compute_aec = bool(ablation_cfg.get("compute_aec", True))
    aec_max_samples = ablation_cfg.get("aec_max_samples", 800)

    results: dict[str, Any] = {}
    for mode in modes:
        report = _evaluate_mode(
            model,
            loader,
            mode,
            device,
            seed=cfg.seed,
            compute_aec=compute_aec,
            aec_max_samples=aec_max_samples,
        )
        results[mode] = report
        if json_logs:
            emit_json(
                {
                    "event": "ablation_mode",
                    "mode": mode,
                    "severe_recall": report.get("severe_recall"),
                    "weighted_logloss": report.get("weighted_logloss"),
                    "aec_mean": report.get("aec_mean"),
                }
            )
        else:
            log.info(
                "ablation mode=%s severe_recall=%.4f weighted_logloss=%.4f",
                mode,
                float(report.get("severe_recall", float("nan"))),
                float(report.get("weighted_logloss", float("nan"))),
            )

    out_path = out_dir / "ablation.json"
    payload = {
        "research_only": True,
        "not_diagnostic": True,
        "severe_index": SEVERE_INDEX,
        "modes": modes,
        "results": results,
        "comparison": compare_ablations(results),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return results


def compare_ablations(results: dict[str, Any]) -> dict[str, Any]:
    """Summarise metric deltas of each mode relative to the ``correct`` baseline.

    For every mode other than ``correct`` (and when a ``correct`` baseline is
    present), report the signed change in ``severe_recall`` and
    ``weighted_logloss`` versus ``correct``. A drop in ``severe_recall`` or a
    rise in ``weighted_logloss`` under perturbation indicates reliance on the
    anatomy prior.
    """
    baseline = results.get("correct")
    if not isinstance(baseline, dict):
        return {}

    base_recall = float(baseline.get("severe_recall", float("nan")))
    base_logloss = float(baseline.get("weighted_logloss", float("nan")))

    deltas: dict[str, Any] = {}
    for mode, report in results.items():
        if mode == "correct" or not isinstance(report, dict):
            continue
        deltas[mode] = {
            "severe_recall_delta": float(report.get("severe_recall", float("nan"))) - base_recall,
            "weighted_logloss_delta": float(report.get("weighted_logloss", float("nan")))
            - base_logloss,
        }
    return deltas


__all__ = [
    "ABLATION_MODES",
    "compare_ablations",
    "perturb_anatomy",
    "run_ablation",
]
