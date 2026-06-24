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


def test_assign_levels_monotonic_prior_recovers_positions():
    # ambiguous scorer logits; the train-derived positional prior should pull each level to
    # its typical normalized-z (l1/l2 high z .. l5/s1 low z) under the monotonic constraint
    n = 10
    logp = np.zeros((n, 5))  # uninformative -> prior decides
    norm_zs = np.array([r / (n - 1) for r in range(n)])
    prior = {0: (0.82, 0.11), 1: (0.65, 0.12), 2: (0.48, 0.13), 3: (0.30, 0.08), 4: (0.13, 0.07)}
    assign = al.assign_levels_monotonic_prior(logp, norm_zs, prior, beta=1.0)
    # monotonic order preserved
    ranks = [assign[i] for i in range(5)]
    assert all(ranks[i] > ranks[i + 1] for i in range(4))
    # each level lands near its prior mean (within 1 rank of round(mean*(n-1)))
    for li in range(5):
        expected_rank = round(prior[li][0] * (n - 1))
        assert abs(assign[li] - expected_rank) <= 1


def test_assign_levels_monotonic_prior_beta0_matches_base():
    rng = np.random.default_rng(1)
    logp = rng.normal(size=(12, 5))
    norm_zs = np.linspace(0, 1, 12)
    prior = dict.fromkeys(range(5), (0.5, 0.2))
    a0 = al.assign_levels_monotonic_prior(logp, norm_zs, prior, beta=0.0)
    base = al.assign_levels_monotonic(logp)
    assert a0 == base  # beta=0 -> identical to the original decoder


def test_scorer_forward_shape():
    model = build_axial_level_scorer().eval()
    img = torch.randn(3, 1, 128, 128)
    z = torch.rand(3, 1)
    out = model(img, z)
    assert out.shape == (3, 5)
