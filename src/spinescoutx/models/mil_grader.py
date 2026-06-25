"""Multi-instance (MIL) severity grader over K candidate crops (v1.5).

A shared CNN encoder embeds each of the K candidate crops in a bag; the instance embeddings are
aggregated (max / attention / gated-attention) into a bag embedding, fused with level + condition
embeddings, and mapped to 3 severity logits. This lets the grader use several plausible localized
crops instead of trusting a single auto crop. The encoder can be warm-started from a deployed
single-crop grader. Research-only. Not diagnostic.
"""

from __future__ import annotations

import torch
from torch import nn

from .image_classifier import build_backbone


class MILGrader(nn.Module):
    """``(bags [B,K,3,H,W], level_idx, condition_idx) -> logits [B,3]``."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        in_chans: int = 3,
        num_classes: int = 3,
        pooling: str = "attention",
        num_levels: int = 5,
        num_conditions: int = 5,
        embed_dim: int = 16,
        attn_dim: int = 128,
        dropout: float = 0.2,
        instance_dropout: float = 0.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if pooling not in ("max", "mean", "attention", "gated"):
            raise ValueError(f"unknown pooling {pooling}")
        self.pooling = pooling
        self.instance_dropout = float(instance_dropout)
        self.encoder, feat = build_backbone(backbone, in_chans, pretrained)
        self.feat = feat
        if pooling in ("attention", "gated"):
            self.attn_V = nn.Linear(feat, attn_dim)
            self.attn_U = nn.Linear(feat, attn_dim) if pooling == "gated" else None
            self.attn_w = nn.Linear(attn_dim, 1)
        self.level_embedding = nn.Embedding(num_levels, embed_dim)
        self.condition_embedding = nn.Embedding(num_conditions, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat + 2 * embed_dim, num_classes)

    def _pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, K, feat); mask: (B, K) bool (True = valid instance)
        neg = torch.finfo(h.dtype).min  # dtype-safe -inf (works under AMP float16)
        if self.pooling in ("max", "mean"):
            hm = h.masked_fill(~mask.unsqueeze(-1), neg if self.pooling == "max" else 0.0)
            if self.pooling == "max":
                return hm.max(dim=1).values
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            return hm.sum(1) / denom
        a = torch.tanh(self.attn_V(h))
        if self.attn_U is not None:  # gated
            a = a * torch.sigmoid(self.attn_U(h))
        scores = self.attn_w(a).squeeze(-1)  # (B, K)
        if self.training and self.instance_dropout > 0:
            drop = (torch.rand_like(scores) < self.instance_dropout) & mask
            mask = mask & ~drop
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, K, 1)
        return (w * h).sum(dim=1)

    def forward(
        self,
        bags: torch.Tensor,
        level_idx: torch.Tensor,
        condition_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, k = bags.shape[0], bags.shape[1]
        if mask is None:
            mask = torch.ones(b, k, dtype=torch.bool, device=bags.device)
        feats = self.encoder(bags.reshape(b * k, *bags.shape[2:])).reshape(b, k, self.feat)
        pooled = self._pool(feats, mask)
        parts = [pooled, self.level_embedding(level_idx), self.condition_embedding(condition_idx)]
        return self.head(self.dropout(torch.cat(parts, dim=1)))

    def load_encoder_from(self, state_dict: dict) -> int:
        """Warm-start the encoder from a deployed grader checkpoint (encoder.* keys)."""
        enc = {k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        missing = self.encoder.load_state_dict(enc, strict=False)
        return len(enc) - len(getattr(missing, "missing_keys", []))


def build_mil_grader(**kwargs) -> MILGrader:
    return MILGrader(**kwargs)
