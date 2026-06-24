"""Sagittal-T1 side-aware foraminal localization + auto-crop generation.

Neural-foraminal-narrowing findings live on **sagittal-T1** parasagittal slices and are
**side-specific**. Two facts (measured on RSNA, see `five_finding_auto_plan.md`) make a
clean auto route possible:

1. **Laterality from DICOM** — within a sagittal-T1 series, ``ImagePositionPatient[0]``
   (physical L–R) is monotonic; +x is the patient's LEFT (verified vs GT: GT-left mean
   x +18 vs GT-right −15). Instance order alone is unreliable. So we sort slices by
   physical x and assign side by which half.
2. **Per-side co-planarity** — the 5 foraminal levels of a side sit on ~one parasagittal
   slice (instance std 0.55), so the canal-style single-slice 5-keypoint heatmap localizer
   (:class:`DiscLevelLocalizer`) transfers directly.

Training supervision uses GT coordinates; **auto inference reads NO GT coordinates** —
the T1 series and the parasagittal slice are chosen from DICOM geometry + the localizer's
own confidence (best-slice scoring). Research-only. Not diagnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import LEVELS, SEVERITY_TO_INDEX
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .crops import CropRecord, extract_25d, write_manifest
from .localizer import extract_peaks, peak_confidence

log = get_logger()

FORAMINAL = ("left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing")


def _ipp_x(images_dir: Path, study: str, series: str, inst: int) -> float | None:
    """Physical L–R coordinate (ImagePositionPatient[0]) of one instance; +x = patient LEFT."""
    import pydicom

    p = images_dir / study / series / f"{inst}.dcm"
    if not p.exists():
        return None
    try:
        d = pydicom.dcmread(str(p), stop_before_pixels=True)
        ipp = getattr(d, "ImagePositionPatient", None)
        return float(ipp[0]) if ipp is not None else None
    except Exception:  # noqa: BLE001
        return None


def slices_by_lr(images_dir: Path, study: str, series: str) -> list[tuple[int, float]]:
    """Return ``[(instance, x)]`` for a series sorted by physical x ascending (right→left)."""
    import pydicom

    out: list[tuple[int, float]] = []
    for p in (images_dir / study / series).glob("*.dcm"):
        if not p.stem.isdigit():
            continue
        try:
            d = pydicom.dcmread(str(p), stop_before_pixels=True)
            ipp = getattr(d, "ImagePositionPatient", None)
            if ipp is not None:
                out.append((int(p.stem), float(ipp[0])))
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda t: t[1])
    return out


def pick_sagittal_t1(series_index: pd.DataFrame, study: str, images_dir: Path) -> str | None:
    """Most-populated sagittal-T1 series for a study (no GT involved)."""
    cand = series_index[
        (series_index.study_id.astype(str) == str(study))
        & (series_index.sequence_type == "sagittal_t1")
    ]
    best, best_n = None, -1
    for sid in cand.series_id.astype(str):
        n = len(list((images_dir / study / sid).glob("*.dcm")))
        if n > best_n:
            best, best_n = sid, n
    return best


def side_candidate_instances(
    lr_sorted: list[tuple[int, float]], side: str, lo: float = 0.5, hi: float = 0.95
) -> list[int]:
    """Candidate parasagittal instances for a side, by rank-fraction of the L–R-sorted stack.

    +x = patient LEFT, so ``left`` takes the high-x end and ``right`` the low-x end. The
    localizer's confidence then selects the best slice among these candidates.
    """
    n = len(lr_sorted)
    if n == 0:
        return []
    if side == "left":
        idx = range(int(lo * n), min(n, int(hi * n) + 1))
    else:  # right -> low-x end (mirror of the same fractions)
        idx = range(max(0, n - int(hi * n) - 1), n - int(lo * n))
    return [lr_sorted[i][0] for i in idx]


# --------------------------------------------------------------------------- #
# localizer training data (sagittal-T1 parasagittal slice + 5 foraminal keypoints/side)
# --------------------------------------------------------------------------- #
def prepare_foraminal_localizer_data(
    rsna_root: str | Path,
    out_cache: str | Path,
    split_map: dict[str, str],
    *,
    slice_size: int = 256,
    limit_studies: int | None = None,
) -> dict[str, Any]:
    """Cache one parasagittal-T1 slice per (study, side) + its 5 foraminal keypoints.

    Output schema matches the canal localizer manifest (``slice_path``, ``kx_<lv>`` /
    ``ky_<lv>``, ``orig_h/w``, ``split``) so ``train_localizer`` works unchanged. ``split``
    is mapped from ``split_map`` (splits_v1): train→train, dev→val (selection), test→test
    (excluded from localizer training/selection). Resume-safe.
    """
    import cv2

    from .dicom_io import normalize_intensity, read_dicom
    from .rsna_index import RsnaPaths, build_series_index
    from .rsna_labels import load_coordinates

    rsna_root = Path(rsna_root)
    out = Path(out_cache)
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)
    ensure_dir(out / "slices")

    coords = load_coordinates(rsna_root)
    coords["study_id"] = coords.study_id.astype(str)
    coords["series_id"] = coords.series_id.astype(str)
    series = build_series_index(rsna_root)
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)
    fc = coords[coords.condition.isin(FORAMINAL)].merge(
        series[["study_id", "series_id", "sequence_type"]], on=["study_id", "series_id"], how="left"
    )
    fc = fc[fc.sequence_type == "sagittal_t1"].copy()

    studies = sorted(fc.study_id.unique())
    if limit_studies is not None:
        studies = studies[: int(limit_studies)]

    rows: list[dict[str, Any]] = []
    skipped = 0
    for study in studies:
        g = fc[fc.study_id == study]
        for side in ("left", "right"):
            gs = g[g.side == side]
            if gs.empty:
                continue
            series_id = str(gs.series_id.mode().iloc[0])
            inst = int(gs.instance_number.median())
            dpath = images_dir / study / series_id / f"{inst}.dcm"
            if not dpath.exists():
                skipped += 1
                continue
            try:
                img = normalize_intensity(read_dicom(dpath))
            except Exception as exc:  # noqa: BLE001
                log.warning("foraminal localizer decode failed %s: %s", dpath, exc)
                skipped += 1
                continue
            h, w = img.shape
            sx, sy = slice_size / w, slice_size / h
            resized = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
            rel = f"slices/{study}_{side}.npy"
            np.save(out / rel, resized.astype(np.float32))
            kpts = np.full((len(LEVELS), 2), np.nan, dtype=np.float32)
            for _, r in gs.iterrows():
                if r.level in LEVELS:
                    kpts[LEVELS.index(r.level)] = (float(r.x) * sx, float(r.y) * sy)
            row: dict[str, Any] = {
                "study_id": study,
                "series_id": series_id,
                "side": side,
                "instance_number": inst,
                "slice_path": rel,
                "orig_h": int(h),
                "orig_w": int(w),
                "split": {"train": "train", "dev": "val", "test": "test"}.get(
                    split_map.get(study, "train"), "train"
                ),
                "n_levels": int(np.isfinite(kpts).all(axis=1).sum()),
            }
            for li, lv in enumerate(LEVELS):
                row[f"kx_{lv}"], row[f"ky_{lv}"] = float(kpts[li, 0]), float(kpts[li, 1])
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "localizer_manifest.parquet", index=False)
    summary = {
        "rsna_root": str(rsna_root),
        "out_cache": str(out),
        "slice_size": int(slice_size),
        "n_rows": len(frame),
        "skipped": skipped,
        "split": {s: int((frame.split == s).sum()) for s in ("train", "val", "test")}
        if len(frame)
        else {},
    }
    return summary


def _load_foraminal_localizer(run_dir: str | Path, device):
    import json

    import torch

    from ..config import config_from_dict
    from ..models.disc_localizer import build_disc_localizer

    run_dir = Path(run_dir)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    model = build_disc_localizer(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["state_dict"])
    return model, int(cfg.data.crop_size)


def _localize_slice(images_dir, study, series, inst, model, slice_size, device):
    """Run the foraminal localizer on one slice; return (points[5,2] orig px, conf[5])."""
    import cv2
    import torch

    from .dicom_io import normalize_intensity, read_dicom

    dpath = images_dir / study / series / f"{inst}.dcm"
    try:
        img = normalize_intensity(read_dicom(dpath))
    except Exception:  # noqa: BLE001
        return None, None
    h, w = img.shape
    resized = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        hm = model.heatmaps(torch.from_numpy(resized[None, None]).float().to(device))[0]
        hm = hm.cpu().numpy()
    pts = extract_peaks(hm).astype(np.float64)
    conf = peak_confidence(hm)
    pts[:, 0] *= w / slice_size
    pts[:, 1] *= h / slice_size
    return pts, conf


def prepare_rsna_foraminal_auto_crops(
    rsna_root: str | Path,
    localizer_run: str | Path,
    out_cache: str | Path,
    *,
    studies: list[str] | None = None,
    crop_size: int = 224,
    n_candidates: int = 7,
    device: str = "auto",
) -> dict[str, Any]:
    """Auto foraminal crops (L/R) at localizer-predicted points on the best parasagittal-T1
    slice per side (best-slice scoring by localizer confidence). Reads NO GT coordinates."""
    import contextlib

    from ..training.optim import select_device
    from .dicom_io import normalize_intensity, read_dicom
    from .rsna_index import RsnaPaths, build_series_index
    from .rsna_labels import load_labels

    rsna_root = Path(rsna_root)
    out = Path(out_cache)
    ensure_dir(out / "crops")
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)
    device_t = select_device(device)
    model, slice_size = _load_foraminal_localizer(localizer_run, device_t)

    labels = load_labels(rsna_root)
    labels["study_id"] = labels.study_id.astype(str)
    fl = labels[labels.condition.isin(FORAMINAL)].copy()
    series = build_series_index(rsna_root)
    series["study_id"] = series.study_id.astype(str)
    series["series_id"] = series.series_id.astype(str)

    if studies is None:
        studies = sorted(fl.study_id.unique())
    studies = [str(s) for s in studies]

    records: list[CropRecord] = []
    skipped, low_conf = 0, 0
    for study in studies:
        series_id = pick_sagittal_t1(series, study, images_dir)
        if series_id is None:
            skipped += 1
            continue
        lr = slices_by_lr(images_dir, study, series_id)
        if len(lr) < 3:
            skipped += 1
            continue
        g = fl[fl.study_id == study]
        for cond in FORAMINAL:
            side = "left" if cond.startswith("left") else "right"
            cands = side_candidate_instances(lr, side)
            if not cands:
                skipped += 1
                continue
            # best-slice scoring by mean localizer confidence (track scalar mean for
            # selection, keep the per-level confidence array for the low-conf flag)
            best_inst, best_pts, best_conf, best_mc = None, None, None, -1.0
            for inst in cands:
                pts, conf = _localize_slice(
                    images_dir, study, series_id, inst, model, slice_size, device_t
                )
                if pts is None:
                    continue
                mc = float(np.mean(conf))
                if mc > best_mc:
                    best_inst, best_pts, best_conf, best_mc = inst, pts, conf, mc
            if best_inst is None:
                skipped += 1
                continue
            # 2.5D slices around the chosen parasagittal slice
            slices: dict[int, np.ndarray] = {}
            for i in (best_inst - 1, best_inst, best_inst + 1):
                p = images_dir / study / series_id / f"{i}.dcm"
                if p.exists():
                    with contextlib.suppress(Exception):
                        slices[i] = normalize_intensity(read_dicom(p))
            if best_inst not in slices:
                skipped += 1
                continue
            cg = g[g.condition == cond]
            for _, r in cg.iterrows():
                if r.level not in LEVELS:
                    continue
                li = LEVELS.index(r.level)
                x, y = float(best_pts[li, 0]), float(best_pts[li, 1])
                if float(best_conf[li]) < 0.05:
                    low_conf += 1
                rel = f"crops/{study}_{series_id}_{best_inst}_{r.level}_{cond}.npy"
                if not (out / rel).exists():
                    arr, pad = extract_25d(slices, best_inst, x, y, crop_size)
                    np.save(out / rel, arr.astype(np.float32))
                else:
                    pad = ""
                sev = str(r.severity)
                records.append(
                    CropRecord(
                        study_id=study,
                        series_id=series_id,
                        instance_number=int(best_inst),
                        condition=cond,
                        level=str(r.level),
                        side=side,
                        severity=sev,
                        severity_index=SEVERITY_TO_INDEX.get(sev, -1),
                        x=x,
                        y=y,
                        crop_path=rel,
                        dicom_path=str(images_dir / study / series_id / f"{best_inst}.dcm"),
                        split="auto",
                        sequence="sagittal_t1",
                        patient_id=study,
                        pad_note=pad,
                        coordinate_source="auto",
                    )
                )
    manifest = write_manifest(records, out / "manifest.parquet")
    return {
        "rsna_root": str(rsna_root),
        "out_cache": str(out),
        "localizer_run": str(localizer_run),
        "n_studies": len(studies),
        "n_auto_crops": len(records),
        "skipped": skipped,
        "low_confidence_crops": low_conf,
        "manifest": str(manifest),
    }
