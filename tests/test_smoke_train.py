"""Fast synthetic smoke tests for the training and ablation entrypoints.

These exercise the real training loops end-to-end on tiny synthetic data
(1 epoch, a couple of capped steps, small_cnn/unet, CPU) so they finish in a few
seconds with no real data and no downloads.
"""

from __future__ import annotations

import pytest

from spinescoutx.config import Config, DataConfig, ModelConfig, TrainConfig

pytestmark = pytest.mark.slow


def _classify_config(task: str, backbone: str = "small_cnn") -> Config:
    return Config(
        name="smoke_classify",
        seed=1337,
        task=task,
        output_root="runs",
        data=DataConfig(
            crop_size=32,
            use_25d=True,
            num_workers=0,
            synthetic=True,
            synthetic_n=16,
            val_fraction=0.25,
        ),
        model=ModelConfig(
            kind="image_classifier" if task == "classify" else "anatomy_guided_classifier",
            backbone=backbone,
            pretrained=False,
            in_chans=3,
            anatomy_in_chans=3,
            num_classes=3,
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=4,
            amp=False,
            freeze_backbone_epochs=0,
            early_stop_patience=2,
            max_steps=2,
            device="cpu",
        ),
    )


def _segment_config() -> Config:
    return Config(
        name="smoke_segment",
        seed=1337,
        task="segment",
        output_root="runs",
        data=DataConfig(
            crop_size=32,
            use_25d=False,
            num_workers=0,
            synthetic=True,
            synthetic_n=16,
            val_fraction=0.25,
        ),
        model=ModelConfig(
            kind="anatomy_segmenter",
            backbone="unet",
            pretrained=False,
            in_chans=1,
            num_anatomy_classes=4,
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=4,
            amp=False,
            early_stop_patience=2,
            loss="dice_ce",
            monitor="val_mean_dice",
            monitor_mode="max",
            max_steps=2,
            device="cpu",
        ),
    )


def test_smoke_train_classifier(tmp_path) -> None:
    from spinescoutx.training.train_classifier import train_classifier

    cfg = _classify_config("classify")
    result = train_classifier(cfg, tmp_path)
    assert isinstance(result, dict)
    assert "best" in result
    assert isinstance(result["best"], dict)
    checkpoint = result.get("checkpoint")
    assert checkpoint is not None
    from pathlib import Path

    assert Path(checkpoint).exists()


def test_smoke_train_segmenter(tmp_path) -> None:
    from spinescoutx.training.train_segmenter import train_segmenter

    cfg = _segment_config()
    result = train_segmenter(cfg, tmp_path)
    assert isinstance(result, dict)
    best = result.get("best", {})
    # mean_dice appears somewhere in the reported metrics.
    flat_keys = set(best.keys()) if isinstance(best, dict) else set()
    assert any("dice" in str(k) for k in flat_keys) or "history" in result


def test_smoke_ablation(tmp_path) -> None:
    from spinescoutx.evaluation.ablation import compare_ablations, run_ablation

    cfg = _classify_config("anatomy_guided")
    cfg.ablation = {"modes": ["correct", "zero", "shuffled"], "split": "val"}
    results = run_ablation(cfg, tmp_path)
    assert isinstance(results, dict)
    assert "correct" in results
    deltas = compare_ablations(results)
    assert isinstance(deltas, dict)


def test_perturb_anatomy_modes() -> None:
    import torch

    from spinescoutx.evaluation.ablation import perturb_anatomy

    anatomy = torch.rand(4, 3, 16, 16)
    assert torch.equal(perturb_anatomy(anatomy, "correct"), anatomy)
    assert torch.count_nonzero(perturb_anatomy(anatomy, "zero")) == 0
    shuffled = perturb_anatomy(anatomy, "shuffled")
    assert shuffled.shape == anatomy.shape
    assert not torch.equal(shuffled, anatomy)
    noise = perturb_anatomy(anatomy, "noise", seed=1337)
    noise2 = perturb_anatomy(anatomy, "noise", seed=1337)
    assert torch.equal(noise, noise2)
