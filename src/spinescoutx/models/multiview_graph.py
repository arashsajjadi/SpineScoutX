"""E3: study-level multi-view anatomy-graph severity reasoner.

A study is a small graph of lumbar disc levels (L1/L2 … L5/S1). Each level is a
node carrying up to three evidence *streams* (views):

  * **image**   — the 2.5D sagittal-T2 crop (CNN backbone token),
  * **anatomy** — the SPIDER disc/canal/vertebra prior masks (small-CNN token),
  * **morphology** — the 14 structured geometry features (MLP token).

The streams are fused into one node token, a level embedding is added, and a
Transformer encoder performs **cross-level attention** over the (≤5) present nodes,
masked by a level-availability mask. A shared head then predicts a 3-class severity
per level. Toggling ``use_image/use_anatomy/use_morphology`` gives the Phase-9
stream ablations; ``graph_layers=0`` disables cross-level attention (per-level
baseline) for the anatomy-graph ablation.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .image_classifier import SmallCNN, build_backbone

NUM_LEVELS = 5


class MultiViewAnatomyGraphClassifier(nn.Module):
    """Per-level multi-stream tokens + cross-level attention + per-level head."""

    def __init__(
        self,
        *,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 3,
        num_levels: int = NUM_LEVELS,
        use_image: bool = True,
        use_anatomy: bool = True,
        use_morphology: bool = True,
        morph_dim: int = 14,
        d_model: int = 256,
        graph_layers: int = 2,
        graph_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not (use_image or use_anatomy or use_morphology):
            raise ValueError("at least one of use_image/use_anatomy/use_morphology must be True")
        self.use_image = use_image
        self.use_anatomy = use_anatomy
        self.use_morphology = use_morphology
        self.num_levels = num_levels

        token_dim = 0
        if use_image:
            self.image_encoder, d_img = build_backbone(backbone, 3, pretrained)
            token_dim += d_img
        if use_anatomy:
            # anatomy masks are simple binary channels -> a tiny offline CNN suffices
            self.anatomy_encoder = SmallCNN(in_chans=3, feat_dim=128)
            token_dim += 128
        if use_morphology:
            self.morph_mlp = nn.Sequential(
                nn.Linear(morph_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 64)
            )
            token_dim += 64

        self.token_proj = nn.Linear(token_dim, d_model)
        self.level_embedding = nn.Embedding(num_levels, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.graph_layers = graph_layers
        if graph_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=graph_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=graph_layers)
        else:
            self.encoder = None

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def set_backbone_trainable(self, flag: bool) -> None:
        """Freeze/unfreeze the image backbone (mirrors ImageClassifier API)."""
        if self.use_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = flag

    def _node_tokens(
        self, images: torch.Tensor, anatomy: torch.Tensor, morph: torch.Tensor
    ) -> torch.Tensor:
        """Fuse the enabled streams into ``(B, L, d_model)`` node tokens."""
        b, n = images.shape[:2]
        parts: list[torch.Tensor] = []
        if self.use_image:
            f = self.image_encoder(images.reshape(b * n, *images.shape[2:]))
            parts.append(f.reshape(b, n, -1))
        if self.use_anatomy:
            f = self.anatomy_encoder(anatomy.reshape(b * n, *anatomy.shape[2:]))
            parts.append(f.reshape(b, n, -1))
        if self.use_morphology:
            parts.append(self.morph_mlp(morph))
        fused = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.token_proj(fused)

    def forward(
        self,
        images: torch.Tensor,  # (B, L, 3, H, W)
        anatomy: torch.Tensor,  # (B, L, 3, H, W)
        morph: torch.Tensor,  # (B, L, morph_dim)
        level_idx: torch.Tensor,  # (B, L) long
        mask: torch.Tensor,  # (B, L) bool: True = level present
    ) -> torch.Tensor:
        """Return per-level severity logits ``(B, L, num_classes)``."""
        tokens = self._node_tokens(images, anatomy, morph)
        tokens = tokens + self.level_embedding(level_idx.clamp(min=0))
        tokens = self.input_norm(tokens)
        if self.encoder is not None:
            # key_padding_mask: True where a position should be IGNORED (absent level)
            tokens = self.encoder(tokens, src_key_padding_mask=~mask)
        return self.head(self.dropout(tokens))


def build_multiview_graph(model_cfg: ModelConfig) -> MultiViewAnatomyGraphClassifier:
    """Construct the E3 reasoner from a :class:`ModelConfig`."""
    return MultiViewAnatomyGraphClassifier(
        backbone=model_cfg.backbone,
        pretrained=model_cfg.pretrained,
        num_classes=model_cfg.num_classes,
        use_image=model_cfg.use_image,
        use_anatomy=model_cfg.use_anatomy,
        use_morphology=model_cfg.use_morphology,
        morph_dim=model_cfg.morph_dim,
        d_model=model_cfg.graph_d_model,
        graph_layers=model_cfg.graph_layers,
        graph_heads=model_cfg.graph_heads,
        dropout=model_cfg.dropout,
    )
