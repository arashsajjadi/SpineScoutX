"""Regression test for SPIDER->RSNA anatomy-prior generation.

Builds synthetic RSNA crops, trains nothing (uses a freshly-built segmenter saved
as a run), generates anatomy priors, and confirms they are 3-channel, crop-aligned,
and loadable by RsnaCropDataset in guided mode. Skipped if no pydicom.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pydicom")
torch = pytest.importorskip("torch")

from conftest import make_synthetic_rsna_root  # noqa: E402
from spinescoutx.config import ModelConfig  # noqa: E402
from spinescoutx.constants import NUM_ANATOMY_PRIOR_CHANNELS  # noqa: E402
from spinescoutx.data.anatomy_priors import generate_anatomy_priors  # noqa: E402
from spinescoutx.data.datasets import RsnaCropDataset  # noqa: E402
from spinescoutx.data.rsna_prepare import prepare_rsna  # noqa: E402
from spinescoutx.models.anatomy_segmenter import build_segmenter  # noqa: E402


def _make_segmenter_run(run_dir) -> None:
    """Save a minimal (untrained) segmenter run dir: config.json + best.pt."""
    run_dir.mkdir(parents=True)
    model_cfg = ModelConfig(
        kind="anatomy_segmenter", backbone="unet", in_chans=1, num_anatomy_classes=4
    )
    model = build_segmenter(model_cfg)
    cfg = {"model": model_cfg.__dict__, "data": {"crop_size": 64}, "train": {"device": "cpu"}}
    (run_dir / "config.json").write_text(json.dumps(cfg))
    torch.save({"state_dict": model.state_dict()}, run_dir / "best.pt")


def test_generate_anatomy_priors(tmp_path) -> None:
    root = tmp_path / "rsna"
    root.mkdir()
    make_synthetic_rsna_root(root)
    cache = tmp_path / "cache"
    prepare_rsna(root, cache, crop_size=32, use_25d=True, val_fraction=0.5, seed=1)

    seg_run = tmp_path / "seg_run"
    _make_segmenter_run(seg_run)
    priors = tmp_path / "priors"

    summary = generate_anatomy_priors(cache, seg_run, priors, device="cpu")
    assert summary["n_crops"] == 4
    assert summary["priors_written"] == 4
    assert summary["skipped"] == 0
    # Canal target region is real anatomy; foraminal is approximate.
    rv = summary["region_validity"]
    assert rv["spinal_canal_stenosis"]["region_source"] == "anatomy"
    assert rv["left_neural_foraminal_narrowing"]["region_source"] == "approximate"

    # Priors are 3-channel, crop-aligned, and loadable in guided mode.
    mpath = cache / "manifest.parquet"
    manifest = pd.read_parquet(mpath) if mpath.exists() else pd.read_csv(cache / "manifest.csv")
    prior = np.load(priors / manifest.iloc[0]["crop_path"])
    assert prior.shape == (NUM_ANATOMY_PRIOR_CHANNELS, 32, 32)

    ds = RsnaCropDataset(
        manifest, cache, crop_size=32, use_25d=True, guided=True, anatomy_cache_root=str(priors)
    )
    item = ds[0]
    assert item["anatomy"].shape == (NUM_ANATOMY_PRIOR_CHANNELS, 32, 32)
    assert (priors / "anatomy_prior_manifest.csv").exists()
