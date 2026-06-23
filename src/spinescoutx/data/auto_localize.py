"""Auto disc-level localization + auto-crop generation (real inference path).

At inference this path uses ONLY: the sagittal T2 series (from the series index),
the geometric mid slice, the trained localizer, and the GT *severity label* as the
evaluation target. It MUST NOT read ``train_label_coordinates.csv`` — crop centres
come from the localizer's predictions, never from GT coordinates.

Research-only. Not diagnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import LEVELS, SEVERITY_TO_INDEX
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .crops import CropRecord, extract_25d, extract_crop, write_manifest
from .localizer import _mid_instance, extract_peaks, peak_confidence

log = get_logger()


def load_localizer(run_dir: str | Path, device):
    import torch

    from ..config import config_from_dict
    from ..models.disc_localizer import build_disc_localizer

    run_dir = Path(run_dir)
    import json

    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    model = build_disc_localizer(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["state_dict"])
    return model, int(cfg.data.crop_size)


def _pick_sagittal_t2(series_index: pd.DataFrame, study: str, images_dir: Path) -> str | None:
    """Pick the sagittal-T2 series with the most instances (no GT involved)."""
    cand = series_index[
        (series_index.study_id == study) & (series_index.sequence_type == "sagittal_t2")
    ]
    best, best_n = None, -1
    for sid in cand.series_id.astype(str):
        n = len(list((images_dir / study / sid).glob("*.dcm")))
        if n > best_n:
            best, best_n = sid, n
    return best


def localize_study(
    study: str, images_dir: Path, series_index: pd.DataFrame, model, slice_size: int, device
) -> dict[str, Any] | None:
    """Predict 5 disc-level points (original-pixel space) for a study. No GT coords."""
    import cv2
    import torch

    from .dicom_io import normalize_intensity, read_dicom

    series_id = _pick_sagittal_t2(series_index, study, images_dir)
    if series_id is None:
        return None
    inst = _mid_instance(images_dir / study / series_id)
    if inst is None:
        return None
    dpath = images_dir / study / series_id / f"{inst}.dcm"
    try:
        img = normalize_intensity(read_dicom(dpath))
    except Exception as exc:  # noqa: BLE001 - logged, never faked
        log.warning("auto-localize decode failed %s: %s", dpath, exc)
        return None
    h, w = img.shape
    resized = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        hm = (
            model.heatmaps(torch.from_numpy(resized[None, None]).float().to(device))[0]
            .cpu()
            .numpy()
        )
    peaks = extract_peaks(hm)  # slice space (x,y)
    conf = peak_confidence(hm)
    # scale predicted points back to original pixel space
    points = peaks.copy()
    points[:, 0] *= w / slice_size
    points[:, 1] *= h / slice_size
    return {
        "series_id": series_id,
        "instance_number": inst,
        "points": points,  # [5,2] original pixel space, per LEVELS order
        "confidence": conf,  # [5]
    }


def prepare_rsna_auto_crops(
    rsna_root: str | Path,
    localizer_run: str | Path,
    out_cache: str | Path,
    *,
    split: str = "val",
    crop_size: int = 224,
    use_25d: bool = True,
    limit_studies: int | None = None,
    device: str = "auto",
) -> dict[str, object]:
    """Auto-crop spinal-canal-stenosis findings at PREDICTED disc-level points.

    Uses labels (severity target) but NOT ``train_label_coordinates.csv``. Crops are
    centred on the localizer's predictions (``coordinate_source="auto"``).
    """
    from ..training.optim import select_device
    from .dicom_io import normalize_intensity, read_dicom
    from .rsna_index import RsnaPaths, build_series_index
    from .rsna_labels import load_labels
    from .splits import patient_level_split

    rsna_root = Path(rsna_root)
    out = Path(out_cache)
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)
    device_t = select_device(device)
    model, slice_size = load_localizer(localizer_run, device_t)

    # Targets come from labels only (canal stenosis severities). NO coordinates.
    labels = load_labels(rsna_root)
    canal = labels[labels.condition == "spinal_canal_stenosis"].copy()
    series_index = build_series_index(rsna_root)

    studies = sorted(canal.study_id.unique())
    split_map = patient_level_split(studies, 0.2, 1337)
    studies = [s for s in studies if split_map.get(s) == split]
    if limit_studies is not None:
        studies = studies[: int(limit_studies)]

    import contextlib

    ensure_dir(out / "crops")
    records: list[CropRecord] = []
    skipped, low_conf = 0, 0
    for study in studies:
        loc = localize_study(study, images_dir, series_index, model, slice_size, device_t)
        if loc is None:
            skipped += 1
            continue
        series_id, inst = loc["series_id"], loc["instance_number"]
        slices: dict[int, np.ndarray] = {}
        for i in (inst - 1, inst, inst + 1):
            p = images_dir / study / series_id / f"{i}.dcm"
            if p.exists():
                with contextlib.suppress(Exception):
                    slices[i] = normalize_intensity(read_dicom(p))
        if inst not in slices:
            skipped += 1
            continue
        sg = canal[canal.study_id == study]
        for _, r in sg.iterrows():
            li = LEVELS.index(r.level)
            x, y = float(loc["points"][li, 0]), float(loc["points"][li, 1])
            conf = float(loc["confidence"][li])
            if conf < 0.05:
                low_conf += 1
            rel = f"crops/{study}_{series_id}_{inst}_{r.level}_spinal_canal_stenosis.npy"
            if not (out / rel).exists():
                if use_25d:
                    arr, pad = extract_25d(slices, inst, x, y, crop_size)
                else:
                    arr, pad = (
                        np.repeat(extract_crop(slices[inst], x, y, crop_size)[None], 3, 0),
                        "",
                    )
                np.save(out / rel, arr.astype(np.float32))
            else:
                pad = ""
            sev = str(r.severity)
            records.append(
                CropRecord(
                    study_id=study,
                    series_id=series_id,
                    instance_number=inst,
                    condition="spinal_canal_stenosis",
                    level=str(r.level),
                    side=None,
                    severity=sev,
                    severity_index=SEVERITY_TO_INDEX.get(sev, -1),
                    x=x,
                    y=y,
                    crop_path=rel,
                    dicom_path=str(images_dir / study / series_id / f"{inst}.dcm"),
                    split=split,
                    sequence="sagittal_t2",
                    patient_id=study,
                    pad_note=pad,
                    coordinate_source="auto",
                )
            )
    manifest_path = write_manifest(records, out / "manifest.parquet")
    return {
        "rsna_root": str(rsna_root),
        "out_cache": str(out),
        "localizer_run": str(localizer_run),
        "split": split,
        "condition": "spinal_canal_stenosis",
        "coordinate_source": "auto",
        "n_studies": len(studies),
        "n_auto_crops": len(records),
        "skipped_studies": skipped,
        "low_confidence_crops": low_conf,
        "manifest": str(manifest_path),
    }
