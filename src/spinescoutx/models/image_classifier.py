"""Image classification backbones and head for SpineScoutX.

Provides a tiny offline ``SmallCNN`` backbone, a ``build_backbone`` factory that
also wraps ``timm`` models (offline-safe), and an ``ImageClassifier`` that fuses
backbone features with optional level/condition embeddings before a linear head.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from ..utils.logging import get_logger

log = get_logger()


class SmallCNN(nn.Module):
    """A tiny convolutional backbone usable offline without pretrained weights.

    ``forward`` returns global-average-pooled features of shape ``(B, feat_dim)``.
    The last convolution is exposed via :meth:`last_conv` for Grad-CAM.
    """

    def __init__(self, in_chans: int, feat_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim: int = feat_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def last_conv(self) -> nn.Module:
        """Return the final convolutional layer (Grad-CAM target)."""
        return self.block2[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.block2(self.block1(self.stem(x)))
        pooled = self.pool(feats)
        return torch.flatten(pooled, 1)


def build_backbone(name: str, in_chans: int, pretrained: bool) -> tuple[nn.Module, int]:
    """Build a feature-extracting backbone and return ``(module, num_features)``.

    ``name == "small_cnn"`` builds :class:`SmallCNN`; otherwise a ``timm`` model
    is created with ``num_classes=0`` and average global pooling. ``timm`` honours
    ``pretrained`` and stays offline-safe when ``pretrained=False``.
    """
    if name == "small_cnn":
        module = SmallCNN(in_chans=in_chans)
        return module, module.feature_dim

    import timm

    module = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
        in_chans=in_chans,
    )
    num_features = int(module.num_features)
    return module, num_features


def _last_conv_in_module(module: nn.Module) -> nn.Module | None:
    """Return the last ``nn.Conv2d`` found in ``module`` (depth-first), or None."""
    last: nn.Module | None = None
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            last = layer
    return last


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional layer of ``model`` for Grad-CAM targeting."""
    if isinstance(model, SmallCNN):
        return model.last_conv()
    conv = _last_conv_in_module(model)
    if conv is None:
        raise ValueError("No convolutional layer found for Grad-CAM target.")
    return conv


class ImageClassifier(nn.Module):
    """Backbone + optional level/condition embeddings + linear severity head."""

    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        use_level_embedding: bool,
        use_condition_embedding: bool,
        num_levels: int = 5,
        num_conditions: int = 5,
        embed_dim: int = 16,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.use_level_embedding = use_level_embedding
        self.use_condition_embedding = use_condition_embedding

        self.encoder, feat_dim = build_backbone(backbone, in_chans, pretrained)

        fused_dim = feat_dim
        if use_level_embedding:
            self.level_embedding = nn.Embedding(num_levels, embed_dim)
            fused_dim += embed_dim
        else:
            self.level_embedding = None
        if use_condition_embedding:
            self.condition_embedding = nn.Embedding(num_conditions, embed_dim)
            fused_dim += embed_dim
        else:
            self.condition_embedding = None

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(fused_dim, num_classes)

    def set_backbone_trainable(self, flag: bool) -> None:
        """Freeze (``flag=False``) or unfreeze (``flag=True``) the backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = flag

    def gradcam_target_layer(self) -> nn.Module:
        """Return the backbone's last conv layer for Grad-CAM."""
        return last_conv_layer(self.encoder)

    def forward(
        self,
        image: torch.Tensor,
        level_idx: torch.Tensor | None = None,
        condition_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.encoder(image)
        parts = [feats]
        if self.level_embedding is not None:
            if level_idx is None:
                raise ValueError("level_idx is required when use_level_embedding is True.")
            parts.append(self.level_embedding(level_idx))
        if self.condition_embedding is not None:
            if condition_idx is None:
                raise ValueError("condition_idx is required when use_condition_embedding is True.")
            parts.append(self.condition_embedding(condition_idx))
        fused = torch.cat(parts, dim=1) if len(parts) > 1 else feats
        return self.head(self.dropout(fused))


def build_image_classifier(model_cfg: ModelConfig) -> ImageClassifier:
    """Construct an :class:`ImageClassifier` from a :class:`ModelConfig`."""
    return ImageClassifier(
        backbone=model_cfg.backbone,
        in_chans=model_cfg.in_chans,
        num_classes=model_cfg.num_classes,
        use_level_embedding=model_cfg.use_level_embedding,
        use_condition_embedding=model_cfg.use_condition_embedding,
        embed_dim=model_cfg.embed_dim,
        dropout=model_cfg.dropout,
        pretrained=model_cfg.pretrained,
    )
