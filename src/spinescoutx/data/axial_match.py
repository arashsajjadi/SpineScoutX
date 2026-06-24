"""Axial-T2 level matching for subarticular stenosis (feasibility core).

Subarticular findings live on **axial-T2**. The hard sub-problem unique to the axial
route is **assigning each lumbar level to the right axial slice** at inference, with no
GT. This module does it from DICOM geometry alone:

1. run the existing sagittal-T2 disc localizer → 5 disc-level points (orig px);
2. convert each disc point to a physical **z** (superior–inferior, patient frame) using
   the sagittal slice's ``ImagePositionPatient`` / ``ImageOrientationPatient`` /
   ``PixelSpacing``;
3. read each axial slice's physical z (``ImagePositionPatient[2]``);
4. match each level to the nearest-z axial slice (top-k for pooling).

A QC routine compares the matched axial slice to the **GT subarticular axial instance**
(used for evaluation only, never for the match) to quantify whether z-matching is
reliable enough to build the full subarticular grader on. Research-only. Not diagnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import LEVELS
from ..utils.logging import get_logger

log = get_logger()

SUBARTICULAR = ("left_subarticular_stenosis", "right_subarticular_stenosis")


def _geom(path: Path) -> dict[str, Any] | None:
    """Return {ipp, iop, ps, rows, cols, inst} for a DICOM slice (geometry only)."""
    import pydicom

    if not path.exists():
        return None
    try:
        d = pydicom.dcmread(str(path), stop_before_pixels=True)
        ipp = getattr(d, "ImagePositionPatient", None)
        iop = getattr(d, "ImageOrientationPatient", None)
        ps = getattr(d, "PixelSpacing", None)
        if ipp is None or iop is None or ps is None:
            return None
        return {
            "ipp": np.array([float(v) for v in ipp]),
            "iop": np.array([float(v) for v in iop]),
            "ps": np.array([float(v) for v in ps]),  # [row_spacing, col_spacing]
            "rows": int(getattr(d, "Rows", 0)),
            "cols": int(getattr(d, "Columns", 0)),
        }
    except Exception:  # noqa: BLE001
        return None


def pixel_to_patient(geom: dict[str, Any], x_col: float, y_row: float) -> np.ndarray:
    """Map an image pixel (x=col, y=row) to a 3D patient-frame position (mm).

    P = IPP + Xdir*(col*colSpacing) + Ydir*(row*rowSpacing), where Xdir=IOP[0:3]
    (increasing column) and Ydir=IOP[3:6] (increasing row); PixelSpacing=[row,col].
    """
    ipp, iop, ps = geom["ipp"], geom["iop"], geom["ps"]
    xdir, ydir = iop[0:3], iop[3:6]
    return ipp + xdir * (x_col * ps[1]) + ydir * (y_row * ps[0])


def axial_z_by_instance(images_dir: Path, study: str, series: str) -> dict[int, float]:
    """{instance -> physical z (IPP[2])} for an axial series."""
    out: dict[int, float] = {}
    for p in (images_dir / study / series).glob("*.dcm"):
        if not p.stem.isdigit():
            continue
        g = _geom(p)
        if g is not None:
            out[int(p.stem)] = float(g["ipp"][2])
    return out


def pick_axial_t2(series_index: pd.DataFrame, study: str, images_dir: Path) -> str | None:
    """Most-populated axial-T2 series for a study."""
    cand = series_index[
        (series_index.study_id.astype(str) == str(study))
        & (series_index.sequence_type == "axial_t2")
    ]
    best, best_n = None, -1
    for sid in cand.series_id.astype(str):
        n = len(list((images_dir / study / sid).glob("*.dcm")))
        if n > best_n:
            best, best_n = sid, n
    return best


def disc_level_z(geom: dict[str, Any], points: np.ndarray) -> dict[str, float]:
    """Per-level physical z from sagittal disc points (orig px, [5,2] x,y)."""
    out: dict[str, float] = {}
    for li, lv in enumerate(LEVELS):
        x, y = float(points[li, 0]), float(points[li, 1])
        out[lv] = float(pixel_to_patient(geom, x, y)[2])
    return out


def match_levels_to_axial(
    level_z: dict[str, float], axial_z: dict[int, float], top_k: int = 1
) -> dict[str, list[int]]:
    """Match each level to its nearest-z axial instance(s). Returns {level -> [inst,...]}."""
    insts = sorted(axial_z)
    zs = np.array([axial_z[i] for i in insts])
    out: dict[str, list[int]] = {}
    for lv, z in level_z.items():
        order = np.argsort(np.abs(zs - z))[: max(1, top_k)]
        out[lv] = [insts[i] for i in order]
    return out


def qc_level_matching(
    rsna_root: str | Path,
    localizer_run: str | Path,
    studies: list[str],
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """QC z-based level matching against GT subarticular axial instances (eval only).

    For each study: localize discs on sagittal-T2 → disc z → match to axial slices; then
    compare the matched axial instance to the GT subarticular instance for each level
    (GT used ONLY to score the match). Reports slice-distance + z-distance distributions.
    """
    from ..training.optim import select_device
    from .auto_localize import load_localizer, localize_study
    from .rsna_index import RsnaPaths, build_series_index
    from .rsna_labels import load_coordinates

    rsna_root = Path(rsna_root)
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)
    device_t = select_device(device)
    model, slice_size = load_localizer(localizer_run, device_t)

    series = build_series_index(rsna_root)
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)
    coords = load_coordinates(rsna_root)
    coords["study_id"] = coords.study_id.astype(str)
    coords["series_id"] = coords.series_id.astype(str)
    sub = coords[coords.condition.isin(SUBARTICULAR)].copy()

    slice_dists: list[int] = []
    z_dists: list[float] = []
    n_studies, skipped = 0, 0
    for study in [str(s) for s in studies]:
        loc = localize_study(study, images_dir, series, model, slice_size, device_t)
        if loc is None:
            skipped += 1
            continue
        sag_geom = _geom(
            images_dir / study / str(loc["series_id"]) / f"{loc['instance_number']}.dcm"
        )
        ax_series = pick_axial_t2(series, study, images_dir)
        if sag_geom is None or ax_series is None:
            skipped += 1
            continue
        axial_z = axial_z_by_instance(images_dir, study, ax_series)
        if len(axial_z) < 3:
            skipped += 1
            continue
        lvl_z = disc_level_z(sag_geom, loc["points"])
        matched = match_levels_to_axial(lvl_z, axial_z, top_k=1)
        # GT subarticular instances for this study, per level (any side; axial slice ~same)
        gt = sub[(sub.study_id == study)]
        gt = gt[gt.series_id.astype(str) == str(ax_series)]
        if gt.empty:
            continue
        n_studies += 1
        for lv in LEVELS:
            mlist = matched.get(lv, [])
            if not mlist:
                continue
            m_inst = mlist[0]
            gl = gt[gt.level == lv]
            if gl.empty:
                continue
            gt_inst = int(gl.instance_number.median())
            slice_dists.append(abs(m_inst - gt_inst))
            if gt_inst in axial_z and m_inst in axial_z:
                z_dists.append(abs(axial_z[m_inst] - axial_z[gt_inst]))

    sd = np.array(slice_dists)
    zd = np.array(z_dists)
    return {
        "n_studies": n_studies,
        "skipped": skipped,
        "n_levels_scored": int(sd.size),
        "slice_distance": {
            "median": float(np.median(sd)) if sd.size else None,
            "mean": float(np.mean(sd)) if sd.size else None,
            "within_0": float((sd == 0).mean()) if sd.size else None,
            "within_1": float((sd <= 1).mean()) if sd.size else None,
            "within_2": float((sd <= 2).mean()) if sd.size else None,
        },
        "z_distance_mm": {
            "median": float(np.median(zd)) if zd.size else None,
            "p90": float(np.percentile(zd, 90)) if zd.size else None,
        },
    }
