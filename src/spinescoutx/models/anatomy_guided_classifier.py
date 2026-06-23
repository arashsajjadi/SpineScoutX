"""Anatomy-guided severity classifier for SpineScoutX.

Fuses an image-encoder backbone (shared with :mod:`image_classifier`) with a
small anatomy-prior encoder and optional level/condition embeddings, then applies
a linear severity head.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from ..utils.logging import get_logger
from .image_classifier import build_backbone, last_conv_layer

log = get_logger()


class AnatomyEncoder(nn.Module):
    """A small CNN over anatomy-prior mask channels -> ``(B, feat_dim)``."""

    def __init__(self, in_chans: int = 3, feat_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim: int = feat_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x)
        return torch.flatten(self.pool(feats), 1)


class AnatomyGuidedClassifier(nn.Module):
    """Image encoder + anatomy encoder + embeddings, concat-fused into a head."""

    def __init__(
        self,
        backbone: str,
        in_chans: int = 3,
        anatomy_in_chans: int = 3,
        num_classes: int = 3,
        use_level_embedding: bool = True,
        use_condition_embedding: bool = True,
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

        self.image_encoder, image_feat_dim = build_backbone(backbone, in_chans, pretrained)
        self.anatomy_encoder = AnatomyEncoder(in_chans=anatomy_in_chans)

        fused_dim = image_feat_dim + self.anatomy_encoder.feature_dim
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
        """Freeze (``flag=False``) or unfreeze (``flag=True``) the image encoder."""
        for param in self.image_encoder.parameters():
            param.requires_grad = flag

    def gradcam_target_layer(self) -> nn.Module:
        """Return the image encoder's last conv layer for Grad-CAM."""
        return last_conv_layer(self.image_encoder)

    def forward(
        self,
        image: torch.Tensor,
        anatomy: torch.Tensor,
        level_idx: torch.Tensor | None = None,
        condition_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_feats = self.image_encoder(image)
        anatomy_feats = self.anatomy_encoder(anatomy)
        parts = [image_feats, anatomy_feats]
        if self.level_embedding is not None:
            if level_idx is None:
                raise ValueError("level_idx is required when use_level_embedding is True.")
            parts.append(self.level_embedding(level_idx))
        if self.condition_embedding is not None:
            if condition_idx is None:
                raise ValueError("condition_idx is required when use_condition_embedding is True.")
            parts.append(self.condition_embedding(condition_idx))
        fused = torch.cat(parts, dim=1)
        return self.head(self.dropout(fused))


def build_anatomy_guided_classifier(model_cfg: ModelConfig) -> AnatomyGuidedClassifier:
    """Construct an :class:`AnatomyGuidedClassifier` from a :class:`ModelConfig`."""
    return AnatomyGuidedClassifier(
        backbone=model_cfg.backbone,
        in_chans=model_cfg.in_chans,
        anatomy_in_chans=model_cfg.anatomy_in_chans,
        num_classes=model_cfg.num_classes,
        use_level_embedding=model_cfg.use_level_embedding,
        use_condition_embedding=model_cfg.use_condition_embedding,
        embed_dim=model_cfg.embed_dim,
        dropout=model_cfg.dropout,
        pretrained=model_cfg.pretrained,
    )
