"""Tests for the E3 study-level multi-view anatomy-graph reasoner."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spinescoutx.config import ModelConfig  # noqa: E402
from spinescoutx.models.multiview_graph import build_multiview_graph  # noqa: E402


def _cfg(**kw) -> ModelConfig:
    base = {
        "kind": "multiview_graph",
        "backbone": "small_cnn",
        "pretrained": False,
        "graph_d_model": 32,
        "graph_heads": 4,
        "graph_layers": 2,
    }
    base.update(kw)
    return ModelConfig(**base)


def _batch(b=2, n=5, size=64):
    images = torch.randn(b, n, 3, size, size)
    anatomy = torch.randn(b, n, 3, size, size)
    morph = torch.randn(b, n, 14)
    level_idx = torch.arange(n).unsqueeze(0).repeat(b, 1)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[1, 3:] = False  # second study only has 3 present levels
    return images, anatomy, morph, level_idx, mask


def test_forward_shape_and_finite() -> None:
    m = build_multiview_graph(_cfg())
    out = m(*_batch())
    assert out.shape == (2, 5, 3)
    assert torch.isfinite(out).all()


def test_masked_loss_backprops() -> None:
    m = build_multiview_graph(_cfg())
    images, anatomy, morph, level_idx, mask = _batch()
    out = m(images, anatomy, morph, level_idx, mask)
    tgt = torch.randint(0, 3, (2, 5))
    tgt[~mask] = -100
    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(out.reshape(-1, 3), tgt.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in m.head.parameters())


def test_absent_levels_do_not_change_present_logits() -> None:
    """Cross-attention must ignore padded levels: present-level logits are
    invariant to the *content* placed at masked positions."""
    m = build_multiview_graph(_cfg()).eval()
    images, anatomy, morph, level_idx, mask = _batch()
    with torch.no_grad():
        out_a = m(images, anatomy, morph, level_idx, mask)
        images2 = images.clone()
        images2[1, 3:] = torch.randn_like(images2[1, 3:]) * 100  # garbage in padded slots
        out_b = m(images2, anatomy, morph, level_idx, mask)
    # logits at present positions of study 1 must be unchanged
    assert torch.allclose(out_a[1, :3], out_b[1, :3], atol=1e-4)


def test_stream_ablations_build_and_run() -> None:
    for kw in (
        {"use_image": True, "use_anatomy": False, "use_morphology": False},
        {"use_image": False, "use_anatomy": True, "use_morphology": False},
        {"use_image": False, "use_anatomy": False, "use_morphology": True},
        {"graph_layers": 0},  # no cross-level attention (per-level baseline)
    ):
        m = build_multiview_graph(_cfg(**kw))
        out = m(*_batch())
        assert out.shape == (2, 5, 3)
        assert torch.isfinite(out).all()


def test_requires_at_least_one_stream() -> None:
    with pytest.raises(ValueError):
        build_multiview_graph(_cfg(use_image=False, use_anatomy=False, use_morphology=False))


def test_morphology_vector_matches_engine() -> None:
    from spinescoutx.data.study_dataset import _morph_vector
    from spinescoutx.features.morphology import NUM_FEATURES

    mask = np.zeros((3, 64, 64), dtype=np.float32)
    mask[1, 10:50, 28:36] = 1.0
    vec = _morph_vector(mask)
    assert vec.shape == (NUM_FEATURES,)
    assert np.isfinite(vec).all()
