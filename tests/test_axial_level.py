"""Tests for the coordinate-supervised axial level scorer + monotonic decoding."""

from __future__ import annotations

import numpy as np
import torch

from spinescoutx.data import axial_level as al
from spinescoutx.models.axial_level_scorer import build_axial_level_scorer


def test_offsets_and_conditions():
    assert set(al.SUBARTICULAR_OFFSETS) == {"left", "right"}
    # left lateral recess is right-of-centre in image x, right is left-of-centre
    assert al.SUBARTICULAR_OFFSETS["left"][0] > 0.5 > al.SUBARTICULAR_OFFSETS["right"][0]
    assert al.SUBARTICULAR_COND["left"] == "left_subarticular_stenosis"


def test_assign_levels_monotonic_recovers_clear_peaks():
    # 10 z-ascending slices; place levels at decreasing rank (l1/l2 highest z .. l5/s1 lowest)
    n = 10
    logp = np.full((n, 5), -10.0)
    placement = {0: 8, 1: 6, 2: 4, 3: 2, 4: 0}  # level_idx -> slice rank
    for lvl, sidx in placement.items():
        logp[sidx, lvl] = 0.0
    assign = al.assign_levels_monotonic(logp)
    assert assign == placement


def test_assign_levels_monotonic_enforces_ordering():
    # Even with noisy peaks, the assignment must be strictly decreasing rank by level
    rng = np.random.default_rng(0)
    logp = rng.normal(size=(12, 5))
    assign = al.assign_levels_monotonic(logp)
    ranks = [assign[i] for i in range(5)]
    # level order l1/l2..l5/s1 maps to strictly DECREASING slice rank
    assert all(ranks[i] > ranks[i + 1] for i in range(4))


def test_scorer_forward_shape():
    model = build_axial_level_scorer().eval()
    img = torch.randn(3, 1, 128, 128)
    z = torch.rand(3, 1)
    out = model(img, z)
    assert out.shape == (3, 5)
