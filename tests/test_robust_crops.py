"""Tests for localizer-aware jitter + the robust on-the-fly crop dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spinescoutx.constants import NUM_ANATOMY_PRIOR_CHANNELS
from spinescoutx.data.robust_crops import (
    CropJitterConfig,
    JitterSampler,
    LocalizerErrorProfile,
    RobustCanalCropDataset,
    _slice_npy_rel,
)


def test_jitter_none_is_zero():
    s = JitterSampler(CropJitterConfig(mode="none"))
    rng = np.random.default_rng(0)
    assert s.sample("l4_l5", rng) == (0.0, 0.0, 0)


def test_jitter_fixed_is_clamped_and_seeded():
    cfg = CropJitterConfig(mode="fixed", xy_sigma=3.0, max_offset=5.0, slice_jitter=2)
    a = [JitterSampler(cfg).sample("l4_l5", np.random.default_rng(7)) for _ in range(3)]
    b = [JitterSampler(cfg).sample("l4_l5", np.random.default_rng(7)) for _ in range(3)]
    assert a == b  # deterministic under the same seed
    for dx, dy, ds in a:
        assert abs(dx) <= 5.0 and abs(dy) <= 5.0
        assert -2 <= ds <= 2


def test_jitter_level_aware_requires_profile():
    try:
        JitterSampler(CropJitterConfig(mode="level_aware"))
    except ValueError:
        return
    raise AssertionError("level_aware without a profile should raise")


def _toy_profile():
    per = {
        lv: np.array([[1.0, -1.0], [2.0, 0.0]])
        for lv in ("l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1")
    }
    sig = dict.fromkeys(per, (1.5, 1.0))
    return LocalizerErrorProfile(per, np.array([0, 1, -1, 0]), sig)


def test_jitter_empirical_samples_measured_residuals():
    s = JitterSampler(CropJitterConfig(mode="empirical"), _toy_profile())
    dx, dy, _ = s.sample("l4_l5", np.random.default_rng(1))
    assert (dx, dy) in ((1.0, -1.0), (2.0, 0.0))


def _make_slice_cache(tmp_path, study="100", series="200", insts=range(8, 13), size=64):
    sl = tmp_path / "slices"
    sl.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in insts:
        np.save(sl / _slice_npy_rel(study, series, i), rng.random((size, size)).astype(np.float32))
    nodes = pd.DataFrame(
        [
            {
                "study_id": study,
                "series_id": series,
                "instance_number": 10,
                "level": "l4_l5",
                "condition": "spinal_canal_stenosis",
                "x": 32.0,
                "y": 32.0,
                "severity_index": 2,
            }
        ]
    )
    return nodes


def test_robust_dataset_schema_and_no_jitter_is_deterministic(tmp_path):
    nodes = _make_slice_cache(tmp_path)
    j = JitterSampler(CropJitterConfig(mode="none"))
    ds = RobustCanalCropDataset(nodes, tmp_path, crop_size=32, jitter=j)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["anatomy"].shape == (NUM_ANATOMY_PRIOR_CHANNELS, 32, 32)
    assert int(item["target"]) == 2
    assert item["study_id"] == "100"
    # no-jitter crop is reproducible
    a = RobustCanalCropDataset(nodes, tmp_path, crop_size=32, jitter=j)[0]["image"].numpy()
    b = RobustCanalCropDataset(nodes, tmp_path, crop_size=32, jitter=j)[0]["image"].numpy()
    assert np.array_equal(a, b)


def test_robust_dataset_two_views_present(tmp_path):
    nodes = _make_slice_cache(tmp_path)
    j = JitterSampler(CropJitterConfig(mode="fixed", xy_sigma=4.0), None)
    ds = RobustCanalCropDataset(nodes, tmp_path, crop_size=32, jitter=j, two_views=True, seed=5)
    item = ds[0]
    assert "image2" in item and item["image2"].shape == (3, 32, 32)
