"""End-to-end smoke of the REAL RSNA code path on synthetic DICOM fixtures.

Exercises prepare_rsna -> train E0 (RsnaCropDataset) -> anatomy priors -> train E1
(guided, with priors) so the credential-gated real run has no untested code left.
These are NOT real metrics (synthetic pixels); they prove the path executes.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pydicom")
torch = pytest.importorskip("torch")

from conftest import make_synthetic_rsna_root  # noqa: E402
from spinescoutx.config import Config, DataConfig, ModelConfig, TrainConfig  # noqa: E402
from spinescoutx.data.anatomy_priors import generate_anatomy_priors  # noqa: E402
from spinescoutx.data.rsna_prepare import prepare_rsna  # noqa: E402
from spinescoutx.models.anatomy_segmenter import build_segmenter  # noqa: E402
from spinescoutx.training.train_classifier import (  # noqa: E402
    evaluate_classifier,
    train_classifier,
)


def _seg_run(run_dir) -> None:
    run_dir.mkdir(parents=True)
    mc = ModelConfig(kind="anatomy_segmenter", backbone="unet", in_chans=1, num_anatomy_classes=4)
    torch.save({"state_dict": build_segmenter(mc).state_dict()}, run_dir / "best.pt")
    (run_dir / "config.json").write_text(
        json.dumps({"model": mc.__dict__, "data": {"crop_size": 64}, "train": {"device": "cpu"}})
    )


def _classify_cfg(cache, task, anatomy_cache="") -> Config:
    guided = task == "anatomy_guided"
    return Config(
        name="real_smoke",
        seed=1,
        task=task,
        output_root=str(cache.parent / "runs"),
        data=DataConfig(
            rsna_cache=str(cache),
            anatomy_cache=anatomy_cache,
            crop_size=32,
            use_25d=True,
            num_workers=0,
            synthetic=False,
        ),
        model=ModelConfig(
            kind="anatomy_guided_classifier" if guided else "image_classifier",
            backbone="small_cnn",
            pretrained=False,
            in_chans=3,
            anatomy_in_chans=3,
            num_classes=3,
            embed_dim=8,
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=2,
            amp=False,
            device="cpu",
            freeze_backbone_epochs=0,
            max_steps=2,
            monitor="val_weighted_logloss",
            monitor_mode="min",
            early_stop_patience=3,
        ),
    )


@pytest.mark.slow
def test_real_rsna_path_e0_then_e1(tmp_path) -> None:
    root = tmp_path / "rsna"
    root.mkdir()
    make_synthetic_rsna_root(root)
    cache = tmp_path / "cache"
    prepare_rsna(root, cache, crop_size=32, use_25d=True, val_fraction=0.5, seed=1)

    # E0 — image-only on the real manifest path.
    e0 = _classify_cfg(cache, "classify")
    run0 = tmp_path / "runs" / "e0"
    res0 = train_classifier(e0, run0)
    assert "weighted_logloss" in res0["best"]
    ev0 = evaluate_classifier(e0, run0, split="val")
    assert (run0 / "predictions.json").exists()
    assert "macro_f1" in ev0

    # Anatomy priors from a (toy) segmenter, then E1 — guided on the real path.
    seg_run = tmp_path / "seg"
    _seg_run(seg_run)
    priors = tmp_path / "priors"
    generate_anatomy_priors(cache, seg_run, priors, device="cpu")

    e1 = _classify_cfg(cache, "anatomy_guided", anatomy_cache=str(priors))
    run1 = tmp_path / "runs" / "e1"
    res1 = train_classifier(e1, run1)
    assert "weighted_logloss" in res1["best"]
    assert (run1 / "best.pt").exists()
