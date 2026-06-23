"""Tests for the study-level series/view registry."""

from __future__ import annotations

import pytest

pytest.importorskip("pydicom")

from conftest import make_synthetic_rsna_root  # noqa: E402
from spinescoutx.data.study_registry import (  # noqa: E402
    VIEWS,
    build_study_index,
    inspect_study,
    view_distribution,
)


def test_build_study_index_columns_and_mask(tmp_path) -> None:
    root = tmp_path / "rsna"
    make_synthetic_rsna_root(root)
    index = build_study_index(root, out_cache=tmp_path / "study_index")

    assert (tmp_path / "study_index" / "studies.parquet").exists()
    assert len(index) == 2  # two synthetic studies
    for view in VIEWS:
        for col in (f"has_{view}", f"n_{view}", f"best_{view}", f"slices_{view}"):
            assert col in index.columns
    # synthetic fixture has only a sagittal-T2 series per study
    assert index.has_sagittal_t2.all()
    assert not index.has_axial_t2.any()
    assert (index.view_mask == "010").all()  # t1=0, t2=1, axial=0
    assert index.usable.all()  # a sagittal view is present -> usable
    # the best sagittal-T2 series must be the most-populated real series id
    assert (index.best_sagittal_t2.str.len() > 0).all()
    assert (index.slices_sagittal_t2 > 0).all()


def test_view_distribution_counts(tmp_path) -> None:
    root = tmp_path / "rsna"
    make_synthetic_rsna_root(root)
    index = build_study_index(root)
    dist = view_distribution(index)
    assert dist["n_studies"] == 2
    assert dist["has_sagittal_t2"] == 2
    assert dist["has_axial_t2"] == 0
    assert dist["usable"] == 2
    assert dist["studies_missing_axial"] == 2
    assert dist["view_mask_counts"].get("010") == 2


def test_inspect_study_found_and_missing(tmp_path) -> None:
    root = tmp_path / "rsna"
    make_synthetic_rsna_root(root)
    index = build_study_index(root)
    sid = str(index.study_id.iloc[0])

    detail = inspect_study(root, sid)
    assert detail["found"] is True
    assert detail["study_id"] == sid
    assert detail["has_sagittal_t2"] is True
    assert isinstance(detail["series"], list) and len(detail["series"]) >= 1

    missing = inspect_study(root, "nonexistent-study")
    assert missing["found"] is False
