"""Disc-level keypoint localizer model (heatmap regression via a 2D U-Net).

Predicts one heatmap channel per lumbar disc level (L1/L2…L5/S1) on a sagittal
slice. Peaks are read off the (sigmoid) heatmaps as predicted level points.
Research-only.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from ..constants import LEVELS
from .anatomy_segmenter import UNet2D


class DiscLevelLocalizer(nn.Module):
    """U-Net heatmap regressor: ``[B, 1, H, W] -> [B, num_levels, H, W]`` logits."""

    def __init__(self, in_chans: int = 1, num_levels: int = len(LEVELS), base: int = 32) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.unet = UNet2D(in_chans=in_chans, num_classes=num_levels, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    def heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid heatmaps in [0, 1]."""
        return torch.sigmoid(self.forward(x))


def build_disc_localizer(model_cfg: ModelConfig) -> DiscLevelLocalizer:
    base = int(getattr(model_cfg, "embed_dim", 16)) * 2 or 32
    return DiscLevelLocalizer(
        in_chans=model_cfg.in_chans, num_levels=len(LEVELS), base=max(base, 32)
    )
