"""Cross-dataset anatomy priors: apply the SPIDER segmenter (E4) to RSNA slices.

For every RSNA crop, the SPIDER-trained 2D segmenter predicts a 4-class anatomy
mask on the crop's source sagittal slice; the disc/canal/vertebra channels are
cropped to the localizer window and cached aligned 1:1 with the image crop. This
is the SPIDER->RSNA transfer; the masks are ANATOMY, not pathology, and
foraminal/subarticular target regions are flagged "approximate".
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..constants import (
    ANATOMY_CLASS_TO_INDEX,
    ANATOMY_PRIOR_CHANNELS,
    evidence_region_for,
)
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .crops import extract_crop, frame_to_records, read_manifest
from .dicom_io import normalize_intensity, read_dicom

log = get_logger()

# Anatomy-prior channel -> 4-class index (ANATOMY_PRIOR_CHANNELS = disc, canal, vertebra).
_CHANNEL_CLASS_IDX = tuple(ANATOMY_CLASS_TO_INDEX[c] for c in ANATOMY_PRIOR_CHANNELS)


def _load_segmenter(segmenter_run: Path, device):
    import torch

    from ..config import config_from_dict
    from ..models.anatomy_segmenter import build_segmenter

    cfg = config_from_dict(json.loads((segmenter_run / "config.json").read_text()))
    model = build_segmenter(cfg.model).to(device).eval()
    payload = torch.load(segmenter_run / "best.pt", map_location=device)
    model.load_state_dict(payload["state_dict"])
    return model, int(cfg.data.crop_size)


def _predict_slice_mask(model, slice2d: np.ndarray, seg_size: int, device) -> np.ndarray:
    """Predict a full-resolution 4-class anatomy mask for one sagittal slice."""
    import torch

    h, w = slice2d.shape
    inp = cv2.resize(slice2d, (seg_size, seg_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(inp[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def generate_anatomy_priors(
    rsna_cache: str | Path,
    segmenter_run: str | Path,
    out_cache: str | Path,
    *,
    limit_crops: int | None = None,
    dry_run: bool = False,
    device: str = "auto",
) -> dict[str, object]:
    """Build cached disc/canal/vertebra prior channels for every RSNA crop.

    Returns a JSON-able summary including per-condition target-region validity.
    With ``dry_run`` nothing is decoded or written. Resume-safe.
    """

    from ..training.optim import select_device

    rsna_cache = Path(rsna_cache)
    out = Path(out_cache)
    segmenter_run = Path(segmenter_run)

    manifest_path = rsna_cache / "manifest.parquet"
    if not manifest_path.exists():
        manifest_path = rsna_cache / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No RSNA manifest under {rsna_cache}; run prepare-rsna first.")
    records = frame_to_records(read_manifest(manifest_path))
    if limit_crops is not None:
        records = records[: int(limit_crops)]

    # Per-condition target region + validity (canal = real anatomy; others approximate).
    region_validity: dict[str, dict[str, str]] = {}
    for rec in records:
        region, _side, source = evidence_region_for(rec.condition)
        region_validity[rec.condition] = {"region": region, "region_source": source}

    summary: dict[str, object] = {
        "rsna_cache": str(rsna_cache),
        "segmenter_run": str(segmenter_run),
        "out_cache": str(out),
        "n_crops": len(records),
        "region_validity": region_validity,
        "note": "Anatomy priors are SPIDER-derived anatomy masks, not pathology. "
        "Foraminal/subarticular target regions are approximate.",
    }
    if dry_run:
        summary["dry_run"] = True
        return summary

    if not (segmenter_run / "best.pt").exists():
        raise FileNotFoundError(
            f"No trained segmenter at {segmenter_run}/best.pt; train E4 first "
            "(spinescoutx train-segmenter). Anatomy priors will not be fabricated."
        )

    dev = select_device(device)
    model, seg_size = _load_segmenter(segmenter_run, dev)

    # Crop size = cached image crop side length.
    crop_size = int(np.load(rsna_cache / records[0].crop_path).shape[-1])

    slice_cache: dict[str, np.ndarray] = {}
    written = 0
    skipped = 0
    prior_rows: list[dict[str, object]] = []
    for rec in records:
        prior_rel = rec.crop_path
        prior_abs = out / prior_rel
        region, _side, source = evidence_region_for(rec.condition)
        prior_rows.append(
            {
                "crop_path": prior_rel,
                "condition": rec.condition,
                "level": rec.level,
                "evidence_region": region,
                "evidence_region_source": source,
            }
        )
        if prior_abs.exists():
            continue
        if rec.dicom_path not in slice_cache:
            dpath = Path(rec.dicom_path)
            if not dpath.exists():
                skipped += 1
                continue
            try:
                slice2d = normalize_intensity(read_dicom(dpath))
            except Exception as exc:  # noqa: BLE001 - decode failures are logged, never faked
                log.warning("prior decode failed %s: %s", dpath, exc)
                skipped += 1
                continue
            slice_cache[rec.dicom_path] = _predict_slice_mask(model, slice2d, seg_size, dev)
        mask_full = slice_cache[rec.dicom_path]
        crop_labels = np.rint(extract_crop(mask_full.astype(np.float32), rec.x, rec.y, crop_size))
        channels = np.stack(
            [(crop_labels == idx).astype(np.float32) for idx in _CHANNEL_CLASS_IDX], axis=0
        )
        ensure_dir(prior_abs.parent)
        np.save(prior_abs, channels)
        written += 1

    # Persist the prior manifest with region-validity flags.
    import pandas as pd

    ensure_dir(out)
    pd.DataFrame(prior_rows).drop_duplicates("crop_path").to_csv(
        out / "anatomy_prior_manifest.csv", index=False
    )
    summary["priors_written"] = written
    summary["skipped"] = skipped
    summary["crop_size"] = crop_size
    summary["device"] = str(dev)
    return summary
