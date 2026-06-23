"""Tests for crop-geometry helpers and the 2.5D stacker."""

from __future__ import annotations

import numpy as np

from spinescoutx.data.crops import crop_bounds, extract_25d, extract_crop


def test_crop_bounds_center_no_pad() -> None:
    # A size-4 box centred at (5, 5) inside a 20x20 image needs no padding.
    y0, y1, x0, x1, needs_pad = crop_bounds(x=5.0, y=5.0, size=4, h=20, w=20)
    assert (y1 - y0) == 4
    assert (x1 - x0) == 4
    assert needs_pad is False


def test_crop_bounds_corner_needs_pad() -> None:
    # A box centred at the top-left corner overruns the image bounds.
    y0, y1, x0, x1, needs_pad = crop_bounds(x=0.0, y=0.0, size=8, h=20, w=20)
    assert needs_pad is True
    assert x0 == 0 and y0 == 0


def test_extract_crop_shape_and_center_value() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    image[10, 10] = 9.0
    crop = extract_crop(image, x=10.0, y=10.0, size=6)
    assert crop.shape == (6, 6)
    assert crop.dtype == np.float32
    # The bright pixel must appear somewhere in the crop.
    assert np.isclose(crop.max(), 9.0)


def test_extract_crop_zero_pads_out_of_bounds() -> None:
    image = np.ones((10, 10), dtype=np.float32)
    crop = extract_crop(image, x=0.0, y=0.0, size=8)
    assert crop.shape == (8, 8)
    # Out-of-bounds region (top-left quadrant) must be zero-padded.
    assert crop[0, 0] == 0.0
    # In-bounds region carries the image value.
    assert crop[-1, -1] == 1.0


def test_extract_25d_all_neighbors_present() -> None:
    slices = {
        4: np.ones((10, 10), np.float32),
        5: np.ones((10, 10), np.float32) * 2,
        6: np.ones((10, 10), np.float32) * 3,
    }
    stacked, pad_note = extract_25d(slices, center=5, x=5.0, y=5.0, size=4)
    assert stacked.shape == (3, 4, 4)
    assert pad_note == ""


def test_extract_25d_duplicates_missing_neighbor() -> None:
    # Only center + one neighbor; the other neighbor is filled by duplication.
    slices = {5: np.ones((10, 10), np.float32) * 2, 6: np.ones((10, 10), np.float32) * 3}
    stacked, pad_note = extract_25d(slices, center=5, x=5.0, y=5.0, size=4)
    assert stacked.shape == (3, 4, 4)
    assert "dup_nearest_slice" in pad_note


def test_extract_25d_zero_pads_when_no_neighbor() -> None:
    # Single center slice: both neighbors duplicate the center (nearest available).
    slices = {5: np.ones((10, 10), np.float32)}
    stacked, pad_note = extract_25d(slices, center=5, x=5.0, y=5.0, size=4)
    assert stacked.shape == (3, 4, 4)
    # With only the center available, neighbors duplicate it rather than zero-pad.
    assert np.allclose(stacked[0], stacked[1])
    assert np.allclose(stacked[2], stacked[1])
