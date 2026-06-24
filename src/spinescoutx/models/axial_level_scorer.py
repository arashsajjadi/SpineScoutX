"""Coordinate-supervised axial level scorer.

Pure DICOM-geometry level matching (sagittal-disc-z → axial-slice-z) is unreliable
(~27% within ±1 axial slice) because of a large systematic per-level bias plus noise.
But two signals are strong: (a) each lumbar level sits at a fairly consistent **normalized
z-position** within the axial stack (l1/l2 high, l5/s1 low), and (b) the slice **appearance**
changes by level. This scorer fuses both: a small CNN over the axial slice ⊕ the slice's
normalized z-rank → per-level logits. Trained on GT-labelled subarticular axial slices
(supervision only); at inference it is run over every slice and decoded with a monotonic
level-ordering constraint — no GT at auto inference.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import torch
from torch import nn


class AxialLevelScorer(nn.Module):
    """``(axial slice [B,1,H,W], norm_z [B,1]) -> level logits [B, num_levels]``."""

    def __init__(self, num_levels: int = 5, in_chans: int = 1, base: int = 32) -> None:
        super().__init__()
        self.num_levels = int(num_levels)

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_chans, base),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(base * 4 + 1, base * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base * 2, num_levels),
        )

    def forward(self, img: torch.Tensor, norm_z: torch.Tensor) -> torch.Tensor:
        f = self.pool(self.features(img)).flatten(1)
        z = norm_z.reshape(f.shape[0], 1).to(dtype=f.dtype)
        return self.head(torch.cat([f, z], dim=1))


def build_axial_level_scorer(
    num_levels: int = 5, in_chans: int = 1, base: int = 32
) -> AxialLevelScorer:
    return AxialLevelScorer(num_levels=num_levels, in_chans=in_chans, base=base)
