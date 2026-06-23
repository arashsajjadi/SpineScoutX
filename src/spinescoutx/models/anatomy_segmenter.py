"""2D anatomy segmentation model for SpineScoutX.

Provides a lightweight ``UNet2D`` and a ``build_segmenter`` factory that can use a
MONAI U-Net when requested and available, falling back to ``UNet2D`` otherwise.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from ..utils.logging import get_logger

log = get_logger()


class _DoubleConv(nn.Module):
    """Two 3x3 conv -> BatchNorm -> ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """A small 2D U-Net with two down/up stages and skip connections."""

    def __init__(self, in_chans: int = 1, num_classes: int = 4, base: int = 32) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes

        self.enc1 = _DoubleConv(in_chans, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


def build_segmenter(model_cfg: ModelConfig) -> nn.Module:
    """Build a segmentation model from a :class:`ModelConfig`.

    When ``backbone == "monai_unet"`` and MONAI is importable, build a MONAI
    U-Net; otherwise build :class:`UNet2D` (logging a warning on fallback).
    """
    if model_cfg.backbone == "monai_unet":
        try:
            from monai.networks.nets import UNet as MonaiUNet
        except ImportError:
            log.warning(
                "MONAI not available for backbone 'monai_unet'; "
                "falling back to UNet2D. Install with: pip install spinescoutx[monai]"
            )
        else:
            return MonaiUNet(
                spatial_dims=2,
                in_channels=model_cfg.in_chans,
                out_channels=model_cfg.num_anatomy_classes,
                channels=(32, 64, 128, 256),
                strides=(2, 2, 2),
                num_res_units=2,
            )
    return UNet2D(
        in_chans=model_cfg.in_chans,
        num_classes=model_cfg.num_anatomy_classes,
    )
