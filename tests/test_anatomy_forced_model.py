"""Integrity tests for the anatomy-forced region-pooling classifier.

These assert that anatomy MATERIALLY affects the forward path — they fail if the
model can ignore anatomy (the failure mode of the v0.4 concat model).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spinescoutx.config import ModelConfig
from spinescoutx.constants import CONDITIONS, split_condition
from spinescoutx.models.anatomy_forced_classifier import (
    _REGION_INDEX,
    build_anatomy_forced_classifier,
)


def _model(global_dropout: float = 0.5):
    cfg = ModelConfig(
        kind="anatomy_forced_classifier",
        backbone="small_cnn",
        pretrained=False,
        in_chans=3,
        anatomy_in_chans=3,
        num_classes=3,
        embed_dim=8,
        global_feature_dropout=global_dropout,
    )
    return build_anatomy_forced_classifier(cfg)


def _data(b: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(b, 3, 64, 64, generator=g)
    anat = torch.zeros(b, 3, 64, 64)
    for i in range(b):  # a DIFFERENT mask per sample (so shuffling actually changes inputs)
        anat[i, 0, 10 + i : 25 + i, 10 + i : 25 + i] = 1  # disc
        anat[i, 1, 28:34, 28 + i : 34 + i] = 1  # canal
        anat[i, 2, 5:55, 5:55] = 1  # vertebra
    lvl = torch.zeros(b, dtype=torch.long)
    cond = torch.arange(b) % len(CONDITIONS)
    return img, anat, lvl, cond


def test_zero_mask_region_features_collapse() -> None:
    m = _model().eval()
    img, anat, _, _ = _data()
    feat = m.encoder(img)
    h, w = feat.shape[2:]
    rf = m.region_pool(feat, F.interpolate(anat, size=(h, w), mode="area"))
    rf0 = m.region_pool(feat, torch.zeros(anat.shape[0], 3, h, w))
    assert torch.allclose(rf0, torch.zeros_like(rf0))  # empty region -> 0 feature
    assert not torch.allclose(rf, rf0)


def test_output_depends_on_anatomy() -> None:
    """Decisive: the model output must change when anatomy is zeroed or shuffled."""
    m = _model().eval()
    img, anat, lvl, cond = _data()
    out_correct = m(img, anat, lvl, cond)
    out_zero = m(img, torch.zeros_like(anat), lvl, cond)
    out_shuf = m(img, torch.roll(anat, 1, 0), lvl, cond)
    assert (out_correct - out_zero).abs().max() > 1e-4
    assert (out_correct - out_shuf).abs().max() > 1e-4


def test_empty_mask_no_nan() -> None:
    m = _model().eval()
    img, anat, lvl, cond = _data()
    out = m(img, torch.zeros_like(anat), lvl, cond)
    assert torch.isfinite(out).all()


def test_region_pool_numerically_stable() -> None:
    m = _model().eval()
    img, _, _, _ = _data()
    feat = m.encoder(img)
    h, w = feat.shape[2:]
    near_empty = torch.zeros(2, 3, h, w)
    near_empty[:, 1, 0, 0] = 1e-6
    rf = m.region_pool(feat[:2], near_empty)
    assert torch.isfinite(rf).all()


def test_condition_side_deterministic() -> None:
    m = _model()
    expect = {None: 0, "left": 1, "right": 2}
    for i, c in enumerate(CONDITIONS):
        _, side = split_condition(c)
        assert int(m.condition_side[i]) == expect[side]


def test_canal_condition_attends_canal() -> None:
    m = _model()
    i = CONDITIONS.index("spinal_canal_stenosis")
    attn = torch.softmax(m.region_attn[i], dim=-1)
    assert int(attn.argmax()) == _REGION_INDEX["spinal_canal"]


def test_anatomy_path_receives_gradient() -> None:
    """region_attn (the anatomy selector) must get gradient -> anatomy is in the graph."""
    m = _model().train()
    img, anat, lvl, cond = _data()
    out = m(img, anat, lvl, cond)
    F.cross_entropy(out, torch.zeros(out.shape[0], dtype=torch.long)).backward()
    assert m.region_attn.grad is not None
    assert m.region_attn.grad.abs().sum() > 0
