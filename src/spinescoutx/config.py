"""Typed, explicit experiment configuration loaded from simple YAML files.

A config has four nested sections (``data``, ``model``, ``train``, plus optional
``ablation``/``eval`` dicts). Unknown keys are tolerated but warned about, so a
typo in a YAML file never silently changes behaviour.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .utils.logging import get_logger

log = get_logger()

T = TypeVar("T")


def _from_dict(cls: type[T], d: dict[str, Any]) -> T:
    """Build a dataclass from a dict, ignoring (but warning on) unknown keys."""
    if d is None:
        d = {}
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(d) - known
    if unknown:
        log.warning("Ignoring unknown config keys for %s: %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in d.items() if k in known})  # type: ignore[call-arg]


@dataclass
class DataConfig:
    rsna_cache: str = "data/cache/rsna"
    spider_cache: str = "data/cache/spider"
    anatomy_cache: str = ""  # predicted-anatomy mask cache for guided RSNA crops (real data)
    crop_size: int = 224
    use_25d: bool = True  # stack previous/center/next slice into 3 channels
    num_workers: int = 4
    split_seed: int = 1337
    val_fraction: float = 0.2
    conditions: list[str] | None = None  # subset of CONDITIONS; None = all
    synthetic: bool = False  # use in-memory synthetic data (smoke tests / CI)
    synthetic_n: int = 64


@dataclass
class ModelConfig:
    kind: str = "image_classifier"  # image_classifier|anatomy_segmenter|anatomy_guided_classifier
    backbone: str = "convnext_tiny"  # any timm name, or "small_cnn" for offline tests
    pretrained: bool = True
    in_chans: int = 3
    num_classes: int = 3
    anatomy_in_chans: int = 3  # disc/canal/vertebra prior channels (guided model)
    num_anatomy_classes: int = 4  # background + vertebra/disc/canal (segmenter)
    use_level_embedding: bool = True
    use_condition_embedding: bool = True
    embed_dim: int = 16
    dropout: float = 0.2
    fusion: str = "concat"


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    freeze_backbone_epochs: int = 2
    # LR multiplier applied when the backbone unfreezes (gentle fine-tuning of the
    # pretrained backbone; avoids the unfreeze "shock" of training it at full lr).
    backbone_unfreeze_lr_scale: float = 0.2
    early_stop_patience: int = 5
    class_weighted_loss: bool = True
    weighted_sampler: bool = False
    loss: str = "weighted_ce"  # weighted_ce | focal | dice_ce | dice_focal (seg)
    grad_accum: int = 1
    monitor: str = "val_weighted_logloss"
    monitor_mode: str = "min"  # min|max
    max_steps: int | None = None  # cap steps/epoch (smoke tests)
    device: str = "auto"  # auto|cpu|cuda


@dataclass
class Config:
    name: str = "experiment"
    seed: int = 1337
    task: str = "classify"  # classify | segment | anatomy_guided | ablate
    output_root: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ablation: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = dict(raw or {})
    return Config(
        name=raw.get("name", "experiment"),
        seed=int(raw.get("seed", 1337)),
        task=raw.get("task", "classify"),
        output_root=raw.get("output_root", "runs"),
        data=_from_dict(DataConfig, raw.get("data", {})),
        model=_from_dict(ModelConfig, raw.get("model", {})),
        train=_from_dict(TrainConfig, raw.get("train", {})),
        ablation=raw.get("ablation", {}) or {},
        eval=raw.get("eval", {}) or {},
        notes=raw.get("notes", ""),
    )


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a typed :class:`Config`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {p} must be a YAML mapping, got {type(raw).__name__}")
    return config_from_dict(raw)
