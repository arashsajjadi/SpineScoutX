"""Tests for typed config loading from the bundled YAML files."""

from __future__ import annotations

from pathlib import Path

import pytest

from spinescoutx.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_from_dict,
    load_config,
)


def _configs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


def _config_files() -> list[Path]:
    return sorted(_configs_dir().glob("*.yaml"))


def test_config_files_exist() -> None:
    assert _config_files(), "expected at least one YAML config"


@pytest.mark.parametrize("path", _config_files(), ids=lambda p: p.name)
def test_load_each_config(path: Path) -> None:
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert isinstance(cfg.name, str) and cfg.name
    assert cfg.task in {"classify", "segment", "anatomy_guided", "ablate"}
    assert isinstance(cfg.seed, int)
    assert cfg.data.crop_size > 0
    assert cfg.model.num_classes >= 1 or cfg.model.num_anatomy_classes >= 1


def test_baseline_typed_fields() -> None:
    cfg = load_config(_configs_dir() / "baseline_image_only.yaml")
    assert cfg.task == "classify"
    assert cfg.model.kind == "image_classifier"
    assert cfg.model.in_chans == 3
    assert cfg.train.monitor_mode == "min"


def test_unknown_keys_are_tolerated() -> None:
    raw = {
        "name": "exp",
        "totally_unknown_top_key": 1,
        "data": {"crop_size": 32, "bogus": True},
        "model": {"backbone": "small_cnn", "nonsense": "x"},
        "train": {"epochs": 1, "made_up": 5},
    }
    cfg = config_from_dict(raw)
    assert cfg.name == "exp"
    assert cfg.data.crop_size == 32
    assert cfg.model.backbone == "small_cnn"
    assert cfg.train.epochs == 1


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(_configs_dir() / "does_not_exist.yaml")
