"""Evidence-consistency metrics for SpineScoutX heatmaps.

These metrics quantify how well a model's evidence heatmap (e.g. Grad-CAM) is
concentrated inside an anatomically meaningful region mask. They are pure numpy,
deterministic, and unit-testable on synthetic arrays.

Research-only: these metrics describe attribution behaviour and are not a
diagnostic or clinical signal.
"""

from __future__ import annotations

import numpy as np


def normalize_map(arr: np.ndarray) -> np.ndarray:
    """Shift to be non-negative and scale so the maximum is 1.

    A map that is constant or all-zero (after shifting) is returned as zeros so
    downstream mass computations stay well defined. Pure numpy.
    """
    a = np.asarray(arr, dtype=np.float64)
    a = a - a.min()
    peak = a.max()
    if peak <= 0.0:
        return np.zeros_like(a, dtype=np.float64)
    return a / peak


def anatomical_evidence_consistency(
    heatmap: np.ndarray,
    region_mask: np.ndarray,
    eps: float = 1e-8,
) -> float | None:
    """Anatomical Evidence Consistency (AEC).

    AEC = sum(heatmap * region_mask) / sum(heatmap), i.e. the fraction of total
    heatmap mass that falls inside ``region_mask``. The heatmap is normalized to
    be non-negative before the computation.

    Returns ``None`` when the region mask is empty (no positive entries) or when
    the heatmap carries effectively no mass (total <= ``eps``).
    """
    hm = normalize_map(heatmap)
    mask = np.asarray(region_mask, dtype=np.float64)
    mask = (mask > 0).astype(np.float64)

    if mask.sum() <= 0.0:
        return None

    total = float(hm.sum())
    if total <= eps:
        return None

    inside = float((hm * mask).sum())
    return inside / total


def evidence_leakage(
    heatmap: np.ndarray,
    region_mask: np.ndarray,
    eps: float = 1e-8,
) -> float | None:
    """Fraction of heatmap mass that falls OUTSIDE ``region_mask``.

    Equals ``1 - AEC`` whenever AEC is defined; returns ``None`` under the same
    conditions as :func:`anatomical_evidence_consistency`.
    """
    aec = anatomical_evidence_consistency(heatmap, region_mask, eps=eps)
    if aec is None:
        return None
    return 1.0 - aec


def peak_to_localizer_distance(
    heatmap: np.ndarray,
    x: float,
    y: float,
    normalize: bool = False,
) -> float:
    """Euclidean distance from the heatmap peak to the localizer ``(x, y)``.

    The heatmap argmax is interpreted with the image convention ``row = y`` and
    ``col = x`` (so a flat index is unravelled to ``(row, col)`` and compared
    against ``(y, x)``). When ``normalize`` is True the distance is divided by
    the image diagonal length ``sqrt(H**2 + W**2)``.
    """
    hm = np.asarray(heatmap, dtype=np.float64)
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be 2D (H, W), got shape {hm.shape}")

    h, w = hm.shape
    flat_idx = int(np.argmax(hm))
    peak_row, peak_col = np.unravel_index(flat_idx, hm.shape)

    dy = float(peak_row) - float(y)
    dx = float(peak_col) - float(x)
    dist = float(np.hypot(dx, dy))

    if normalize:
        diag = float(np.hypot(float(h), float(w)))
        if diag <= 0.0:
            return 0.0
        return dist / diag
    return dist


def evidence_consistency_batch(
    heatmaps: list[np.ndarray],
    region_masks: list[np.ndarray],
) -> list[float | None]:
    """Compute :func:`anatomical_evidence_consistency` over a batch.

    ``heatmaps`` and ``region_masks`` must be the same length; each pair is
    scored independently and the per-item AEC values (``float`` or ``None``) are
    returned in order.
    """
    if len(heatmaps) != len(region_masks):
        raise ValueError(
            f"heatmaps and region_masks length mismatch: {len(heatmaps)} != {len(region_masks)}"
        )
    return [
        anatomical_evidence_consistency(hm, mask)
        for hm, mask in zip(heatmaps, region_masks, strict=True)
    ]
