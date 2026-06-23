"""Image overlay helpers for research visualizations.

Pure numpy/OpenCV/matplotlib utilities for compositing masks, heatmaps and
localizer markers onto grayscale crops. All functions return ``HxWx3`` uint8
RGB arrays so they can be embedded directly in figures or saved with cv2.

Research-only — not diagnostic.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
from matplotlib import colormaps

# Default per-class colors (RGB) for the 4-class anatomy scheme. Background is
# left untinted (alpha-blended with weight 0 below).
_DEFAULT_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (66, 135, 245),
    2: (245, 173, 66),
    3: (66, 245, 132),
}


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a 2D grayscale (``[0,1]`` float or uint8) image to ``HxWx3`` uint8.

    Already-RGB ``HxWx3`` inputs are returned as uint8 unchanged (scaled if
    they are in ``[0,1]`` float range).
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        return rgb.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"to_rgb expects 2D or HxWx3 image, got shape {arr.shape}")
    gray = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.floating):
        gray = np.clip(gray, 0.0, 1.0) * 255.0
    gray_u8 = gray.astype(np.uint8)
    return np.repeat(gray_u8[:, :, None], 3, axis=2)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Alpha-blend an integer label ``mask`` over ``image``; return ``HxWx3`` uint8.

    ``mask`` holds integer class ids; background (id 0) is not tinted. Colors
    default to the 4-class anatomy palette and may be overridden per class.
    """
    rgb = to_rgb(image).astype(np.float32)
    mask_arr = np.asarray(mask)
    if mask_arr.shape[:2] != rgb.shape[:2]:
        mask_arr = cv2.resize(
            mask_arr.astype(np.int32),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    colors = class_colors if class_colors is not None else _DEFAULT_CLASS_COLORS
    a = float(np.clip(alpha, 0.0, 1.0))
    out = rgb.copy()
    for class_id in np.unique(mask_arr):
        cid = int(class_id)
        if cid == 0:
            continue
        color = np.asarray(colors.get(cid, (255, 0, 0)), dtype=np.float32)
        sel = mask_arr == cid
        out[sel] = (1.0 - a) * out[sel] + a * color
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    """Color-map a ``[0,1]`` heatmap and alpha-blend it over ``image``.

    Returns ``HxWx3`` uint8. The heatmap is resized to the image and normalized
    to ``[0,1]`` defensively before mapping.
    """
    rgb = to_rgb(image).astype(np.float32)
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.shape[:2] != rgb.shape[:2]:
        hm = cv2.resize(hm, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    hm_min = float(hm.min())
    hm_max = float(hm.max())
    if hm_max - hm_min > 1e-12:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm)
    colormap = colormaps[cmap]
    colored = colormap(hm)[:, :, :3] * 255.0
    a = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - a) * rgb + a * colored
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_localizer(
    rgb: np.ndarray,
    x: float,
    y: float,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 4,
) -> np.ndarray:
    """Draw a small crosshair circle at localizer ``(x, y)`` on an RGB image.

    ``x`` is the column and ``y`` the row. Returns a new ``HxWx3`` uint8 array.
    """
    out = to_rgb(rgb).copy()
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    cv2.circle(out, (cx, cy), int(radius), color, thickness=1, lineType=cv2.LINE_AA)
    cv2.drawMarker(
        out,
        (cx, cy),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=max(2, int(radius)),
        thickness=1,
        line_type=cv2.LINE_AA,
    )
    return out


__all__ = [
    "draw_localizer",
    "overlay_heatmap",
    "overlay_mask",
    "to_rgb",
]
