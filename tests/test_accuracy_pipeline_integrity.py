"""Pure-logic accuracy-pipeline integrity tests (no data required).

These guard the invariants whose violation would silently corrupt severe recall: class-index
order, the severe-recall metric, recall@FAR, crop geometry, and the foraminal side convention.
On-real-data invariants (collect_probs alignment, split disjointness, auto provenance) are in
`scripts/audit_accuracy_pipeline.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinescoutx.constants import SEVERITIES, SEVERITY_TO_INDEX
from spinescoutx.data import crops
from spinescoutx.evaluation import bootstrap as bs


def test_severe_is_class_index_2():
    assert SEVERITIES == ("normal_mild", "moderate", "severe")
    assert SEVERITY_TO_INDEX["severe"] == 2
    assert SEVERITY_TO_INDEX["normal_mild"] == 0 and SEVERITY_TO_INDEX["moderate"] == 1


def test_m_severe_recall_definition():
    # 4 severe (y==2): 3 predicted severe (argmax index 2), 1 missed -> recall 0.75
    y = np.array([2, 2, 2, 2, 0, 1])
    p = np.array(
        [
            [0.1, 0.1, 0.8],  # severe -> caught
            [0.2, 0.2, 0.6],  # severe -> caught
            [0.3, 0.2, 0.5],  # severe -> caught
            [0.6, 0.3, 0.1],  # severe -> MISSED
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
        ]
    )
    assert bs.m_severe_recall(y, p) == pytest.approx(0.75)


def test_m_severe_recall_all_missed_and_all_caught():
    y = np.array([2, 2])
    caught = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    missed = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert bs.m_severe_recall(y, caught) == pytest.approx(1.0)
    assert bs.m_severe_recall(y, missed) == pytest.approx(0.0)


def test_recall_at_far_monotone_and_bounded():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.2).astype(int) * 2  # ~20% severe
    psev = np.clip(rng.normal(np.where(y == 2, 0.6, 0.3), 0.2), 0, 1)
    p = np.stack([1 - psev, np.zeros(n), psev], axis=1)
    p = p / p.sum(1, keepdims=True)
    r5 = bs.make_recall_at_far(0.05)(y, p)
    r20 = bs.make_recall_at_far(0.20)(y, p)
    assert 0.0 <= r5 <= 1.0 and 0.0 <= r20 <= 1.0
    assert r20 >= r5 - 1e-9  # more FAR budget -> at least as much recall


def test_crop_bounds_centered_no_off_by_one():
    # crop_bounds returns (y0, y1, x0, x1, needs_pad); x=col, y=row
    y0, y1, x0, x1, needs_pad = crops.crop_bounds(10.0, 10.0, 4, 20, 20)
    assert (x1 - x0) == 4 and (y1 - y0) == 4
    assert 0 <= x0 < x1 <= 20 and 0 <= y0 < y1 <= 20
    assert needs_pad is False
    # a box near the edge must flag needs_pad (so callers zero-pad, not silently shift)
    assert crops.crop_bounds(1.0, 1.0, 8, 20, 20)[4] is True


def test_extract_crop_shape_and_xy_convention():
    img = np.zeros((20, 30), dtype=np.float32)  # H=20 rows (y), W=30 cols (x)
    img[5:9, 12:16] = 1.0  # a bright box at rows 5-8 (y), cols 12-15 (x)
    # crop centered on the box centre (x=13.5 col, y=6.5 row)
    c = crops.extract_crop(img, 13.5, 6.5, 6)
    assert c.shape == (6, 6)
    # the crop should contain the bright pixels (x indexes columns, y indexes rows)
    assert c.max() == pytest.approx(1.0)


def test_extract_25d_shape():
    sl = {i: np.full((20, 20), float(i), dtype=np.float32) for i in (9, 10, 11)}
    arr, _note = crops.extract_25d(sl, 10, 10.0, 10.0, 8)
    assert arr.shape == (3, 8, 8)
    # channel order is (prev, center, next) -> increasing values 9,10,11
    assert arr[0].mean() < arr[1].mean() < arr[2].mean()


def test_foraminal_side_convention():
    # +x = patient-LEFT (ImagePositionPatient[0]); left = high-x tail, right = low-x tail
    from spinescoutx.data.foraminal_localize import side_candidate_instances

    # list of (instance_number, physical_x) sorted by x ascending (right -> left)
    lr_sorted = [(i, float(i)) for i in range(10)]
    left = side_candidate_instances(lr_sorted, "left")
    right = side_candidate_instances(lr_sorted, "right")
    assert left and right
    # +x = patient-left, so left instances come from the high-x (high-instance) end
    assert min(left) > max(right)
