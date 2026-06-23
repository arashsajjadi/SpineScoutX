"""RSNA crop-extraction pipeline: localizers -> cached 2.5D crops + manifest.

Turns the RSNA label/coordinate/series CSVs plus the DICOM tree into a cached
crop dataset that :class:`~spinescoutx.data.datasets.RsnaCropDataset` can train
on. Decode-once-per-series, cache-first (resume-safe), patient/study-level split.

This module only runs when real RSNA data is present; it never fabricates pixels.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .crops import CropRecord, extract_25d, extract_crop, write_manifest
from .dicom_io import DicomDecodeError, normalize_intensity, read_dicom
from .rsna_index import RsnaPaths, build_series_index, check_rsna_available
from .rsna_labels import load_coordinates, load_labels
from .splits import patient_level_split, split_stats

log = get_logger()


def _dicom_path(images_dir: Path, study: str, series: str, instance: int) -> Path:
    return images_dir / str(study) / str(series) / f"{instance}.dcm"


def _crop_rel(study: str, series: str, inst: int, level: str, cond: str) -> str:
    return f"crops/{study}_{series}_{inst}_{level}_{cond}.npy"


def _decode_series_slices(
    images_dir: Path, study: str, series: str, instances: set[int]
) -> dict[int, object]:
    """Decode the needed instances (and ±1 neighbours) of a series once.

    Returns ``{instance_number: normalized 2D float32 array}``; instances that
    fail to decode are skipped (logged), never fabricated.
    """
    wanted: set[int] = set()
    for i in instances:
        wanted.update((i - 1, i, i + 1))
    slices: dict[int, object] = {}
    for i in sorted(wanted):
        path = _dicom_path(images_dir, study, series, i)
        if not path.exists():
            continue
        try:
            slices[i] = normalize_intensity(read_dicom(path))
        except DicomDecodeError as exc:
            log.warning(
                "decode failed study=%s series=%s inst=%s: %s", study, series, i, exc.category
            )
    return slices


def _merge_findings(
    coords: pd.DataFrame, labels: pd.DataFrame, series: pd.DataFrame
) -> pd.DataFrame:
    """Join localizer coordinates with severities and series sequence types."""
    sev = labels[["study_id", "condition", "level", "side", "severity", "severity_index"]]
    merged = coords.merge(sev, on=["study_id", "condition", "level", "side"], how="left")
    merged["severity"] = merged["severity"].fillna("")
    merged["severity_index"] = merged["severity_index"].fillna(-1).astype(int)
    seq = series[["study_id", "series_id", "sequence_type"]]
    merged = merged.merge(seq, on=["study_id", "series_id"], how="left")
    merged["sequence_type"] = merged["sequence_type"].fillna("unknown")
    return merged


def prepare_rsna(
    rsna_root: str | Path,
    out_cache: str | Path,
    *,
    crop_size: int = 224,
    use_25d: bool = True,
    val_fraction: float = 0.2,
    seed: int = 1337,
    limit_studies: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Extract localizer-centred crops from RSNA DICOMs into a cached manifest.

    Returns a JSON-able summary (counts + distributions). With ``dry_run`` no
    DICOMs are decoded and no files are written. Resume-safe: crops already on
    disk are reused.
    """
    report = check_rsna_available(rsna_root)
    if not report.exists:
        raise FileNotFoundError(
            f"RSNA data not available under {rsna_root}; missing {report.missing}"
        )
    paths = RsnaPaths.from_root(rsna_root)
    images_dir = Path(paths.train_images_dir)
    out = Path(out_cache)

    coords = load_coordinates(rsna_root)
    labels = load_labels(rsna_root)
    series = build_series_index(rsna_root)
    findings = _merge_findings(coords, labels, series)

    studies = sorted(findings["study_id"].unique())
    if limit_studies is not None:
        studies = studies[: int(limit_studies)]
        findings = findings[findings["study_id"].isin(studies)].reset_index(drop=True)
    split_map = patient_level_split(studies, val_fraction, seed)

    summary: dict[str, object] = {
        "rsna_root": str(rsna_root),
        "out_cache": str(out),
        "crop_size": int(crop_size),
        "use_25d": bool(use_25d),
        "n_studies": len(studies),
        "n_series": int(findings["series_id"].nunique()),
        "n_findings": int(len(findings)),
        "split_counts": {s: int(sum(v == s for v in split_map.values())) for s in ("train", "val")},
        "severity_distribution": findings["severity"].value_counts().to_dict(),
        "condition_distribution": findings["condition"].value_counts().to_dict(),
        "level_distribution": findings["level"].value_counts().to_dict(),
        "sequence_distribution": findings["sequence_type"].value_counts().to_dict(),
    }
    if dry_run:
        summary["dry_run"] = True
        return summary

    ensure_dir(out / "crops")
    records: list[CropRecord] = []
    skipped = 0

    # Group by series so each series is decoded once.
    by_series: dict[tuple[str, str], list[pd.Series]] = defaultdict(list)
    for _, row in findings.iterrows():
        by_series[(str(row["study_id"]), str(row["series_id"]))].append(row)

    for (study, series_id), rows in by_series.items():
        instances = {int(r["instance_number"]) for r in rows}
        # Cache-first resume: skip the (expensive) series decode when every crop
        # for this series is already on disk.
        all_cached = all(
            (
                out
                / _crop_rel(study, series_id, int(r["instance_number"]), r["level"], r["condition"])
            ).exists()
            for r in rows
        )
        slices = (
            {} if all_cached else _decode_series_slices(images_dir, study, series_id, instances)
        )
        for r in rows:
            inst = int(r["instance_number"])
            level, cond = str(r["level"]), str(r["condition"])
            crop_rel = _crop_rel(study, series_id, inst, level, cond)
            crop_abs = out / crop_rel
            pad_note = ""
            if not crop_abs.exists():
                if inst not in slices:
                    skipped += 1
                    continue
                if use_25d:
                    arr, pad_note = extract_25d(
                        slices, inst, float(r["x"]), float(r["y"]), crop_size
                    )
                else:
                    arr = extract_crop(slices[inst], float(r["x"]), float(r["y"]), crop_size)
                    arr = np.repeat(arr[None], 3, axis=0)
                ensure_dir(crop_abs.parent)
                np.save(crop_abs, arr.astype(np.float32))
            records.append(
                CropRecord(
                    study_id=study,
                    series_id=series_id,
                    instance_number=inst,
                    condition=cond,
                    level=level,
                    side=r["side"] if pd.notna(r["side"]) else None,
                    severity=str(r["severity"]),
                    severity_index=int(r["severity_index"]),
                    x=float(r["x"]),
                    y=float(r["y"]),
                    crop_path=crop_rel,
                    dicom_path=str(_dicom_path(images_dir, study, series_id, inst)),
                    split=split_map.get(study, "train"),
                    sequence=str(r["sequence_type"]),
                    patient_id=study,
                    pad_note=pad_note,
                )
            )

    manifest_path = write_manifest(records, out / "manifest.parquet")
    frame = pd.DataFrame({"split": [rec.split for rec in records]})
    summary["n_crops_cached"] = len(records)
    summary["skipped_findings"] = skipped
    summary["manifest"] = str(manifest_path)
    summary["crop_split"] = split_stats(frame) if len(frame) else {}
    return summary
