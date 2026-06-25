"""BiGRU axial level-sequence refiner (v1.5).

The deployed ``axial_level_scorer`` classifies each axial slice's lumbar level *independently*.
This refiner reads the whole stack as an ordered sequence (z-ascending) and refines the per-slice
level posteriors with bidirectional context: input per slice = the scorer's 5 level log-probs +
normalized z-rank; a BiGRU emits refined 5-class level logits per slice. The same monotonic decode
(`assign_levels_monotonic`) then maps levels -> slices. Tests whether sequence context improves
axial level localization over the independent per-slice scorer. Research-only. Not diagnostic.
"""

from __future__ import annotations

import torch
from torch import nn


class AxialSeqRefiner(nn.Module):
    """``(feats [B,T,in_dim], lengths [B]) -> level logits [B,T,5]`` via a BiGRU."""

    def __init__(
        self,
        in_dim: int = 6,
        hidden: int = 64,
        num_layers: int = 1,
        num_levels: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, num_levels)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=feats.size(1))
        return self.head(self.drop(out))


def build_axial_seq_refiner(**kwargs) -> AxialSeqRefiner:
    return AxialSeqRefiner(**kwargs)
