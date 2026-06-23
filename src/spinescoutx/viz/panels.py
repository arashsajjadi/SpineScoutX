"""Composite matplotlib figures for research communication.

All panels run on synthetic numpy inputs (no real data, no identifiers) and
stamp every figure with a "Research-only - Not diagnostic" banner. Functions
write a PNG to the given path and return the resolved :class:`Path`.

Research-only - not diagnostic. Figures must never display patient, study or
series identifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..constants import SEVERITIES
from ..utils.paths import ensure_dir
from .heatmaps import normalize_heatmap
from .overlays import overlay_heatmap, overlay_mask, to_rgb

_STAMP_TEXT = "Research-only • Not diagnostic"
SYNTHETIC_PROVENANCE = "⚠ SYNTHETIC SMOKE — not a real RSNA/SPIDER result"


def provenance_label(dataset_source: str | None) -> str | None:
    """Map a finding-graph/report ``dataset_source`` to a provenance watermark.

    Returns the synthetic-smoke warning when the source is synthetic/absent so a
    figure can never be mistaken for a real-data research result; returns ``None``
    for a genuine dataset source (e.g. ``"rsna"``/``"spider"``).
    """
    if not dataset_source or str(dataset_source).lower() in {"synthetic", "smoke", "toy", "fake"}:
        return SYNTHETIC_PROVENANCE
    return None


def _stamp(fig: plt.Figure, provenance: str | None = None) -> None:
    """Add the mandatory research-only banner (and optional provenance warning)."""
    fig.text(
        0.5,
        0.01,
        _STAMP_TEXT,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#b00020",
        fontweight="bold",
    )
    if provenance:
        fig.text(
            0.5,
            0.965,
            provenance,
            ha="center",
            va="top",
            fontsize=10,
            color="#b00020",
            fontweight="bold",
            bbox={"facecolor": "#fff3cd", "edgecolor": "#b00020", "boxstyle": "round,pad=0.3"},
        )


def _save(fig: plt.Figure, path: str | Path) -> Path:
    """Save and close a figure, returning the resolved output path."""
    out = Path(path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _show_image(ax: plt.Axes, image: Any, title: str | None = None) -> None:
    """Render a 2D/3D image array into an axis with no ticks."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray")
    else:
        ax.imshow(to_rgb(arr))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def make_examples_grid(
    items: list[dict], path: str | Path, *, provenance: str | None = None
) -> Path:
    """Grid of example crops with optional anatomy / heatmap overlays.

    Each ``item`` may contain: ``image`` or ``crop`` (2D/3D array),
    ``anatomy`` (label map or prior channels), ``heatmap`` (2D), ``title``.
    """
    n = max(1, len(items))
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= len(items):
            ax.axis("off")
            continue
        item = items[idx]
        base = item.get("crop", item.get("image"))
        rgb = to_rgb(_first_channel(base))
        if item.get("anatomy") is not None:
            rgb = overlay_mask(rgb, _label_map(item["anatomy"]), alpha=0.4)
        if item.get("heatmap") is not None:
            rgb = overlay_heatmap(rgb, normalize_heatmap(np.asarray(item["heatmap"])))
        _show_image(ax, rgb, str(item.get("title", f"example {idx}")))
    fig.suptitle("Example crops (synthetic / research)", fontsize=11)
    _stamp(fig, provenance)
    return _save(fig, path)


def make_failure_cases(
    items: list[dict], path: str | Path, *, provenance: str | None = None
) -> Path:
    """Grid of clearly-labeled FAILURE examples (wrong grade / off-target heatmap)."""
    n = max(1, len(items))
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= len(items):
            ax.axis("off")
            continue
        item = items[idx]
        base = item.get("crop", item.get("image"))
        rgb = to_rgb(_first_channel(base))
        if item.get("heatmap") is not None:
            rgb = overlay_heatmap(rgb, normalize_heatmap(np.asarray(item["heatmap"])))
        pred = item.get("pred_grade", "?")
        true = item.get("true_grade", "?")
        _show_image(ax, rgb, f"FAILURE\npred={pred} / true={true}")
        for spine in ax.spines.values():
            spine.set_edgecolor("#b00020")
            spine.set_linewidth(2.0)
    fig.suptitle("Failure cases (synthetic / research)", fontsize=11)
    _stamp(fig, provenance)
    return _save(fig, path)


def make_segmentation_examples(
    items: list[dict],
    path: str | Path,
    *,
    provenance: str | None = None,
    title: str = "SPIDER anatomy segmentation",
) -> Path:
    """Grid of ``MRI | ground-truth | prediction`` overlays for anatomy segmentation.

    Each ``item`` is ``{image (H,W), gt (H,W label map), pred (H,W label map),
    title?}``. Masks are tinted with the 4-class anatomy palette
    (vertebra/disc/spinal_canal); background is untinted.
    """
    n = max(1, len(items))
    fig, axes = plt.subplots(n, 3, figsize=(9.0, 3.0 * n), squeeze=False)
    headers = ["MRI slice", "ground truth", "prediction"]
    for r in range(n):
        item = items[r] if r < len(items) else {}
        image = _first_channel(item.get("image"))
        gt = np.asarray(item.get("gt"))
        pred = np.asarray(item.get("pred"))
        _show_image(axes[r][0], to_rgb(image), headers[0] if r == 0 else None)
        _show_image(axes[r][1], overlay_mask(image, gt), headers[1] if r == 0 else None)
        _show_image(axes[r][2], overlay_mask(image, pred), headers[2] if r == 0 else None)
        axes[r][0].set_ylabel(str(item.get("title", f"case {r}")), fontsize=8)
    fig.suptitle(title, fontsize=11)
    _stamp(fig, provenance)
    return _save(fig, path)


def make_reliability_diagram(
    curve: dict,
    path: str | Path,
    *,
    ece: float | None = None,
    provenance: str | None = None,
    low_n_threshold: int = 50,
) -> Path:
    """Reliability diagram from a ``reliability_curve``-style dict."""
    conf = np.asarray(curve.get("bin_confidence", []), dtype=np.float32)
    acc = np.asarray(curve.get("bin_accuracy", []), dtype=np.float32)
    counts = np.asarray(curve.get("bin_count", []), dtype=np.float32)
    total_n = int(np.nansum(counts)) if counts.size else 0
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    if conf.size and acc.size:
        mask = counts > 0 if counts.size == conf.size else np.ones_like(conf, dtype=bool)
        ax.plot(conf[mask], acc[mask], marker="o", color="#1f77b4", label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical accuracy")
    title = "Reliability diagram"
    if ece is not None:
        title += f"  (ECE = {float(ece):.3f})"
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    if total_n and total_n < low_n_threshold:
        ax.text(
            0.5,
            0.30,
            f"⚠ low sample count (n={total_n})\ncalibration estimate is unreliable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#b00020",
            fontweight="bold",
        )
    _stamp(fig, provenance)
    return _save(fig, path)


def make_ablation_summary(
    results: dict, path: str | Path, *, provenance: str | None = None
) -> Path:
    """Bar chart of severe_recall and weighted_logloss across ablation modes."""
    modes = list(results.keys())
    severe_recall = [float(_metric(results[m], "severe_recall")) for m in modes]
    logloss = [float(_metric(results[m], "weighted_logloss")) for m in modes]
    x = np.arange(len(modes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    ax1.bar(x, severe_recall, color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, rotation=20, ha="right", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_title("Severe recall by anatomy mode", fontsize=10)
    ax2.bar(x, logloss, color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels(modes, rotation=20, ha="right", fontsize=8)
    ax2.set_title("Weighted log-loss by anatomy mode", fontsize=10)
    fig.suptitle("Anatomy ablation summary (research)", fontsize=11)
    _stamp(fig, provenance)
    return _save(fig, path)


def make_linkedin_hero_panel(
    panel: dict, path: str | Path, *, provenance: str | None = None
) -> Path:
    """Composite hero panel: overview, crop, anatomy prior, evidence, card, failure.

    ``panel`` keys (all optional, synthetic numpy arrays / dicts):
    ``sagittal``, ``crop``, ``anatomy``, ``heatmap``, ``card`` (dict of strings),
    ``failure`` (dict with ``image``/``heatmap``/``pred_grade``/``true_grade``).
    """
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0))

    _show_image(axes[0][0], _first_channel(panel.get("sagittal")), "Sagittal overview")
    _show_image(axes[0][1], _first_channel(panel.get("crop")), "Disc crop")

    crop_rgb = to_rgb(_first_channel(panel.get("crop")))
    anatomy = panel.get("anatomy")
    if anatomy is not None:
        prior_rgb = overlay_mask(crop_rgb, _label_map(anatomy), alpha=0.5)
    else:
        prior_rgb = crop_rgb
    _show_image(axes[0][2], prior_rgb, "Anatomy prior")

    heatmap = panel.get("heatmap")
    if heatmap is not None:
        evid_rgb = overlay_heatmap(crop_rgb, normalize_heatmap(np.asarray(heatmap)))
    else:
        evid_rgb = crop_rgb
    _show_image(axes[1][0], evid_rgb, "Evidence heatmap")

    _draw_card(axes[1][1], panel.get("card", {}))

    failure = panel.get("failure", {})
    fail_rgb = to_rgb(_first_channel(failure.get("image", panel.get("crop"))))
    if failure.get("heatmap") is not None:
        fail_rgb = overlay_heatmap(fail_rgb, normalize_heatmap(np.asarray(failure["heatmap"])))
    pred = failure.get("pred_grade", "?")
    true = failure.get("true_grade", "?")
    _show_image(axes[1][2], fail_rgb, f"FAILURE  pred={pred}/true={true}")
    for spine in axes[1][2].spines.values():
        spine.set_edgecolor("#b00020")
        spine.set_linewidth(2.0)

    fig.suptitle("SpineScoutX - anatomy-guided evidence (research demo)", fontsize=13)
    _stamp(fig, provenance)
    return _save(fig, path)


def figures_from_report(
    report: dict,
    out_dir: str | Path,
    *,
    sample: dict | None = None,
) -> list[Path]:
    """Render figures from a finding-graph / metrics report dict.

    Always produces schematic cards from the report so it works with no real
    arrays. If ``sample`` arrays are supplied they are rendered as an examples
    grid as well. Returns the list of written paths.
    """
    out = Path(out_dir)
    ensure_dir(out)
    written: list[Path] = []
    # Watermark every figure as synthetic unless the report carries a real source.
    prov = provenance_label(report.get("dataset_source"))

    findings = report.get("findings", [])
    written.append(_findings_card_figure(findings, out / "findings_card.png", provenance=prov))

    if "reliability_curve" in report:
        written.append(
            make_reliability_diagram(
                report["reliability_curve"],
                out / "reliability.png",
                ece=report.get("ece"),
                provenance=prov,
            )
        )
    if "ablation" in report and isinstance(report["ablation"], dict):
        written.append(
            make_ablation_summary(report["ablation"], out / "ablation.png", provenance=prov)
        )

    if sample is not None and sample.get("image") is not None:
        item = {
            "crop": sample["image"],
            "anatomy": sample.get("anatomy"),
            "heatmap": sample.get("heatmap"),
            "title": "sample",
        }
        written.append(make_examples_grid([item], out / "examples.png", provenance=prov))

    return written


# --- internal helpers -----------------------------------------------------------


def _first_channel(image: Any) -> np.ndarray:
    """Return a 2D view: first channel of a ``(C,H,W)`` array, else the array."""
    if image is None:
        return np.zeros((32, 32), dtype=np.float32)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] <= arr.shape[-1]:
        return arr[0]
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        return arr[:, :, 0]
    return arr


def _label_map(anatomy: Any) -> np.ndarray:
    """Coerce an anatomy input into an integer label map for ``overlay_mask``.

    Accepts a 2D label map directly, or ``(C,H,W)`` prior channels (argmax+1
    where any channel is active, background 0 elsewhere).
    """
    arr = np.asarray(anatomy)
    if arr.ndim == 2:
        return arr.astype(np.int32)
    if arr.ndim == 3 and arr.shape[0] <= arr.shape[-1]:
        active = arr.max(axis=0) > 0.5
        labels = (arr.argmax(axis=0) + 1).astype(np.int32)
        return np.where(active, labels, 0)
    raise ValueError(f"Unsupported anatomy shape for label map: {arr.shape}")


def _metric(entry: Any, key: str) -> float:
    """Extract a numeric metric from a possibly-nested ablation entry."""
    if isinstance(entry, dict):
        if key in entry:
            value = entry[key]
            return float(value) if value is not None else float("nan")
        for sub in entry.values():
            if isinstance(sub, dict) and key in sub:
                value = sub[key]
                return float(value) if value is not None else float("nan")
    return float("nan")


def _draw_card(ax: plt.Axes, card: dict) -> None:
    """Render a finding-graph summary card (text only, no identifiers)."""
    ax.axis("off")
    ax.set_title("Finding-graph card", fontsize=10)
    lines = ["Research-only finding graph", ""]
    for grade in SEVERITIES:
        count = int(card.get(grade, 0))
        lines.append(f"{grade}: {count}")
    if "limitations" in card:
        lines.append("")
        lines.append("Limitations: anatomy priors, not pathology")
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        transform=ax.transAxes,
    )


def _findings_card_figure(
    findings: list, path: str | Path, *, provenance: str | None = None
) -> Path:
    """Schematic card summarizing finding grades with no identifiers."""
    counts = dict.fromkeys(SEVERITIES, 0)
    for finding in findings:
        grade = finding.get("grade") if isinstance(finding, dict) else None
        if grade in counts:
            counts[grade] += 1
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    grades = list(counts.keys())
    values = [counts[g] for g in grades]
    ax.bar(np.arange(len(grades)), values, color=["#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_xticks(np.arange(len(grades)))
    ax.set_xticklabels(grades, fontsize=9)
    ax.set_ylabel("finding count")
    ax.set_title("Finding-graph summary (research)", fontsize=11)
    if len(findings) <= 1:
        ax.text(
            0.5,
            0.5,
            "⚠ toy smoke output\n(≤1 finding — not a real study)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#b00020",
            fontweight="bold",
        )
    _stamp(fig, provenance)
    return _save(fig, path)


__all__ = [
    "figures_from_report",
    "provenance_label",
    "make_ablation_summary",
    "make_examples_grid",
    "make_failure_cases",
    "make_segmentation_examples",
    "make_linkedin_hero_panel",
    "make_reliability_diagram",
]
