"""Regression test for the SPIDER slice-caching pipeline (cache_spider_slices).

Uses real SimpleITK I/O on tiny synthetic .mha volumes so the whole path
(build_spider_index -> load_volume -> remap_spider_labels -> resize/cache ->
seg_index) is exercised without the real dataset. Skipped if no volume reader.
"""

from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from spinescoutx.data.spider_index import (  # noqa: E402
    _patient_id,
    _slice_axis,
    cache_spider_slices,
)


def _write_volume(path, arr: np.ndarray) -> None:
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, str(path))


def _make_spider_root(root) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for subj in ("1", "2"):
        # (slices=4, H=16, W=16); slice axis is the smallest dim (0).
        image = rng.random((4, 16, 16)).astype(np.float32)
        mask = np.zeros((4, 16, 16), dtype=np.int16)
        mask[1, 4:8, 4:8] = 1  # vertebra (raw id 1)
        mask[2, 4:8, 4:8] = 201  # disc (raw id >= 200)
        mask[1, 10:12, 10:12] = 100  # spinal canal (raw id 100)
        _write_volume(images / f"{subj}_t1.mha", image)
        _write_volume(masks / f"{subj}_t1.mha", mask)


def test_helpers() -> None:
    assert _patient_id("12_t1") == "12"
    assert _slice_axis((300, 320, 15)) == 2
    assert _slice_axis((10, 256, 256)) == 0


def test_cache_spider_slices_end_to_end(tmp_path) -> None:
    root = tmp_path / "raw"
    _make_spider_root(root)
    cache = tmp_path / "cache"

    summary = cache_spider_slices(
        root,
        cache,
        crop_size=16,
        modalities=("t1",),
        min_foreground=0.001,
        val_fraction=0.5,
        seed=1,
    )
    assert summary["n_patients"] == 2
    assert summary["n_volumes"] == 2
    # Only the 2 foreground slices (idx 1 and 2) per volume should be cached.
    assert summary["n_slices_cached"] == 4
    seg_index = cache / "seg_index.parquet"
    if not seg_index.exists():
        seg_index = cache / "seg_index.csv"
    assert seg_index.exists()

    # Cached masks are remapped into 0..3 and sized to crop_size.
    import pandas as pd

    df = pd.read_parquet(seg_index) if seg_index.suffix == ".parquet" else pd.read_csv(seg_index)
    assert {"subject_id", "image_path", "mask_path", "split"}.issubset(df.columns)
    m = np.load(cache / df.iloc[0]["mask_path"])
    assert m.shape == (16, 16)
    assert set(np.unique(m)).issubset({0, 1, 2, 3})
    assert m.max() > 0  # foreground present
    img = np.load(cache / df.iloc[0]["image_path"])
    assert img.shape == (16, 16)


def test_cache_spider_slices_dry_run_writes_nothing(tmp_path) -> None:
    root = tmp_path / "raw"
    _make_spider_root(root)
    cache = tmp_path / "cache"
    summary = cache_spider_slices(root, cache, crop_size=16, modalities=("t1",), dry_run=True)
    assert summary["dry_run"] is True
    assert not (cache / "images").exists()


def test_official_split_from_overview(tmp_path) -> None:
    import pandas as pd

    root = tmp_path / "raw"
    _make_spider_root(root)
    overview = root / "overview.csv"
    pd.DataFrame({"new_file_name": ["1_t1", "2_t1"], "subset": ["training", "validation"]}).to_csv(
        overview, index=False
    )
    cache = tmp_path / "cache"

    summary = cache_spider_slices(
        root,
        cache,
        crop_size=16,
        modalities=("t1",),
        min_foreground=0.001,
        official_split_csv=overview,
    )
    assert summary["split_source"] == "spider_official_overview_csv"
    seg = cache / "seg_index.parquet"
    df = pd.read_parquet(seg) if seg.exists() else pd.read_csv(cache / "seg_index.csv")
    by_subject = dict(zip(df["subject_id"].astype(str), df["split"], strict=False))
    assert by_subject["1_t1"] == "train"
    assert by_subject["2_t1"] == "val"
