"""Regression test for the RSNA crop pipeline on synthetic DICOM fixtures.

Writes tiny real DICOMs (via pydicom) + the three RSNA CSVs, runs prepare_rsna,
and feeds the resulting manifest through RsnaCropDataset. Validates the real code
path end-to-end without the credential-gated RSNA dataset. Skipped if no pydicom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pydicom")

from conftest import make_synthetic_rsna_root, write_synthetic_dicom  # noqa: E402
from spinescoutx.data.datasets import RsnaCropDataset  # noqa: E402
from spinescoutx.data.rsna_prepare import prepare_rsna  # noqa: E402


def test_dicom_roundtrip(tmp_path) -> None:
    from spinescoutx.data.dicom_io import read_dicom

    arr = (np.arange(64 * 64).reshape(64, 64) % 1000).astype(np.uint16)
    p = tmp_path / "x.dcm"
    write_synthetic_dicom(p, arr)
    back = read_dicom(p)
    assert back.shape == (64, 64)
    assert np.allclose(back, arr.astype(np.float32))


def test_prepare_rsna_end_to_end(tmp_path) -> None:
    root = tmp_path / "rsna"
    root.mkdir()
    make_synthetic_rsna_root(root)
    cache = tmp_path / "cache"

    summary = prepare_rsna(root, cache, crop_size=32, use_25d=True, val_fraction=0.5, seed=1)
    assert summary["n_studies"] == 2
    assert summary["n_findings"] == 4
    assert summary["n_crops_cached"] == 4
    assert summary["skipped_findings"] == 0
    assert set(summary["severity_distribution"]) <= {"normal_mild", "moderate", "severe"}

    mpath = cache / "manifest.parquet"
    manifest = pd.read_parquet(mpath) if mpath.exists() else pd.read_csv(cache / "manifest.csv")
    assert len(manifest) == 4
    arr = np.load(cache / manifest.iloc[0]["crop_path"])
    assert arr.shape == (3, 32, 32)

    ds = RsnaCropDataset(manifest, cache, crop_size=32, use_25d=True, guided=False)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert {"image", "level_idx", "condition_idx", "target", "study_id"}.issubset(item.keys())


def test_prepare_rsna_dry_run(tmp_path) -> None:
    root = tmp_path / "rsna"
    root.mkdir()
    make_synthetic_rsna_root(root)
    cache = tmp_path / "cache"
    summary = prepare_rsna(root, cache, crop_size=32, dry_run=True)
    assert summary["dry_run"] is True
    assert not (cache / "crops").exists()
