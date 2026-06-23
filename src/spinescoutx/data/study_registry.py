"""Study-level series/view registry (the move from crop-level to study-level).

For every RSNA study, records which views are available (sagittal T1 / sagittal
T2-STIR / axial T2), the best series per view, counts, a view-availability mask,
and usability flags. Never crashes on a missing view. Research-only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .rsna_index import RsnaPaths, build_series_index

log = get_logger()

VIEWS: tuple[str, ...] = ("sagittal_t1", "sagittal_t2", "axial_t2")


def _series_len(images_dir: Path, study: str, series: str) -> int:
    d = images_dir / str(study) / str(series)
    return len(list(d.glob("*.dcm"))) if d.is_dir() else 0


def build_study_index(rsna_root: str | Path, out_cache: str | Path | None = None) -> pd.DataFrame:
    """Build the per-study series/view registry DataFrame (optionally cached)."""
    rsna_root = Path(rsna_root)
    images_dir = Path(RsnaPaths.from_root(rsna_root).train_images_dir)
    series = build_series_index(rsna_root)
    series = series.copy()
    series["n_slices"] = [
        _series_len(images_dir, r.study_id, r.series_id) for r in series.itertuples()
    ]

    rows: list[dict[str, object]] = []
    for study, g in series.groupby("study_id"):
        row: dict[str, object] = {"study_id": str(study), "n_series": int(len(g))}
        for view in VIEWS:
            v = g[g.sequence_type == view].sort_values("n_slices", ascending=False)
            row[f"has_{view}"] = bool(len(v) > 0)
            row[f"n_{view}"] = int(len(v))
            row[f"best_{view}"] = str(v.series_id.iloc[0]) if len(v) else ""
            row[f"slices_{view}"] = int(v.n_slices.iloc[0]) if len(v) else 0
        row["view_mask"] = "".join("1" if row[f"has_{v}"] else "0" for v in VIEWS)
        # usable = at least one sagittal view (needed to localize disc levels)
        row["usable"] = bool(row["has_sagittal_t2"] or row["has_sagittal_t1"])
        row["n_unknown_series"] = int((g.sequence_type == "unknown").sum())
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("study_id").reset_index(drop=True)
    if out_cache is not None:
        out = ensure_dir(out_cache)
        frame.to_parquet(out / "studies.parquet", index=False)
    return frame


def view_distribution(index: pd.DataFrame) -> dict[str, object]:
    """Summary of view availability across the registry."""
    return {
        "n_studies": int(len(index)),
        "has_sagittal_t1": int(index.has_sagittal_t1.sum()),
        "has_sagittal_t2": int(index.has_sagittal_t2.sum()),
        "has_axial_t2": int(index.has_axial_t2.sum()),
        "usable": int(index.usable.sum()),
        "view_mask_counts": index.view_mask.value_counts().to_dict(),
        "studies_missing_axial": int((~index.has_axial_t2).sum()),
        "studies_missing_sagittal_t2": int((~index.has_sagittal_t2).sum()),
    }


def inspect_study(rsna_root: str | Path, study_id: str) -> dict[str, object]:
    """Return the registry row + per-series detail for one study."""
    index = build_study_index(rsna_root)
    row = index[index.study_id == str(study_id)]
    if len(row) == 0:
        return {"study_id": str(study_id), "found": False}
    series = build_series_index(rsna_root)
    detail = series[series.study_id == str(study_id)][
        ["series_id", "series_description", "sequence_type"]
    ].to_dict("records")
    return {"study_id": str(study_id), "found": True, **row.iloc[0].to_dict(), "series": detail}
