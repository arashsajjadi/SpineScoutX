"""Segmentation-derived morphometric features (v1.8b).

Turns a foundation-model mask + its crop into geometry/intensity features for severity grading —
foraminal opening area, vertical/horizontal extent, compactness, intensity contrast, segmentation
confidence, and shape sanity. These are the *new* evidence the v1.8b fusion grader receives.
Research-only. Not diagnostic.
"""

from __future__ import annotations

import numpy as np


def _bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def foraminal_features(mask: np.ndarray, crop: np.ndarray, iou: float) -> dict:
    """Morphometry of the central foraminal-opening proxy mask (mask HxW bool; crop (3,H,W) [0,1]).

    The neural foramen is the (mostly) fat/CSF opening at the crop centre; severe stenosis shrinks
    it. We measure the segmented central object's size/shape + how it contrasts with the surround.
    """
    h, w = mask.shape
    area = float(mask.sum())
    img = crop[1]  # center channel
    feats = {
        "m_iou_conf": float(iou),
        "m_area_frac": area / (h * w),
        "m_n_components": _n_components(mask),
    }
    bb = _bbox(mask)
    if bb is None or area < 4:
        feats.update({
            "m_h_frac": 0.0, "m_w_frac": 0.0, "m_aspect": 0.0, "m_compactness": 0.0,
            "m_cy": 0.5, "m_cx": 0.5, "m_intensity_mean": float(img.mean()),
            "m_contrast": 0.0, "m_seg_fail": 1.0, "m_min_open": 0.0,
        })  # fmt: skip
        return feats
    y0, y1, x0, x1 = bb
    bh, bw = (y1 - y0 + 1), (x1 - x0 + 1)
    ys, xs = np.where(mask)
    inside = float(img[mask].mean())
    outside = img[~mask]
    feats.update(
        {
            "m_h_frac": bh / h,
            "m_w_frac": bw / w,
            "m_aspect": bh / max(bw, 1),
            "m_compactness": area / (bh * bw),  # 1 = filled bbox
            "m_cy": float(ys.mean()) / h,
            "m_cx": float(xs.mean()) / w,
            "m_intensity_mean": inside,
            "m_contrast": inside - float(outside.mean()) if outside.size else 0.0,
            "m_seg_fail": 0.0,
            # min opening proxy: the smaller normalized extent (a narrow foramen => small)
            "m_min_open": min(bh / h, bw / w),
        }
    )
    return feats


def _n_components(mask: np.ndarray) -> int:
    try:
        from scipy import ndimage

        return int(ndimage.label(mask)[1])
    except Exception:  # noqa: BLE001
        return 1 if mask.any() else 0


def stability_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """IoU between two masks (e.g., adjacent-slice masks) — a stability proxy."""
    u = np.logical_or(m1, m2).sum()
    return float(np.logical_and(m1, m2).sum() / u) if u else 0.0


FEATURE_COLS = [
    "m_iou_conf",
    "m_area_frac",
    "m_n_components",
    "m_h_frac",
    "m_w_frac",
    "m_aspect",
    "m_compactness",
    "m_cy",
    "m_cx",
    "m_intensity_mean",
    "m_contrast",
    "m_seg_fail",
    "m_min_open",
]
