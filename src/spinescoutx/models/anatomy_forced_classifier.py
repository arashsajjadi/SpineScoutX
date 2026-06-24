"""Anatomy-forced region-pooling classifier (SpineScoutX v0.5).

The v0.4 concat-fusion model could ignore anatomy (its ablation showed
correct ≈ shuffled ≈ zero). This model is structurally different: the anatomy
masks DEFINE regions, and features are masked-pooled over those regions. The
region features therefore depend on the mask geometry — zeroing or shuffling the
mask changes them by construction, so the model cannot trivially ignore anatomy.

Pipeline:  image -> feature map ;  masks -> region definitions ;
region pooling (disc/canal/vertebra) ;  condition-specific target-region
attention ;  global feature (with dropout to force region reliance) ;  fuse with
level/condition/side embeddings ;  severity head.

Research-only. Not diagnostic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from ..constants import (
    ANATOMY_PRIOR_CHANNELS,
    CONDITIONS,
    NUM_ANATOMY_PRIOR_CHANNELS,
    split_condition,
)

# Region index (in ANATOMY_PRIOR_CHANNELS order: disc=0, spinal_canal=1, vertebra=2).
_REGION_INDEX = {name: i for i, name in enumerate(ANATOMY_PRIOR_CHANNELS)}
# Side ids: 0 = none, 1 = left, 2 = right.
_SIDE_ID = {None: 0, "left": 1, "right": 2}
# Per-base-condition dominant anatomy region (an interpretable init prior; learned thereafter).
_DOMINANT_REGION = {
    "spinal_canal_stenosis": "spinal_canal",
    "neural_foraminal_narrowing": "disc",
    "subarticular_stenosis": "spinal_canal",
}


class SmallCNNFeatures(nn.Module):
    """Tiny offline feature-map encoder (for tests / no-internet use)."""

    def __init__(self, in_chans: int = 3, width: int = 64) -> None:
        super().__init__()
        self.feature_dim = width
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 2, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, width, 3, 2, 1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_feature_encoder(name: str, in_chans: int, pretrained: bool) -> tuple[nn.Module, int]:
    """Return a (feature-map encoder, channel dim). Encoder output is ``[B, C, h, w]``."""
    if name == "small_cnn":
        enc = SmallCNNFeatures(in_chans)
        return enc, enc.feature_dim
    import timm

    model = timm.create_model(
        name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=in_chans
    )
    return model, int(model.num_features)


def _condition_side_ids() -> torch.Tensor:
    return torch.tensor([_SIDE_ID[split_condition(c)[1]] for c in CONDITIONS], dtype=torch.long)


def _dominant_region_ids() -> torch.Tensor:
    ids = []
    for c in CONDITIONS:
        base, _ = split_condition(c)
        ids.append(_REGION_INDEX[_DOMINANT_REGION[base]])
    return torch.tensor(ids, dtype=torch.long)


class AnatomyForcedRegionClassifier(nn.Module):
    """Severity classifier whose features are region-pooled from anatomy masks."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        in_chans: int = 3,
        anatomy_in_chans: int = NUM_ANATOMY_PRIOR_CHANNELS,
        num_classes: int = 3,
        num_levels: int = 5,
        num_conditions: int = len(CONDITIONS),
        embed_dim: int = 16,
        dropout: float = 0.2,
        global_feature_dropout: float = 0.5,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.anatomy_in_chans = int(anatomy_in_chans)
        self.global_feature_dropout = float(global_feature_dropout)
        self.encoder, feat_dim = build_feature_encoder(backbone, in_chans, pretrained)
        self.feat_dim = feat_dim

        # Condition -> region attention over the anatomy regions; initialized with an
        # interpretable prior (dominant region) but learned thereafter.
        attn_init = torch.zeros(num_conditions, anatomy_in_chans)
        dom = _dominant_region_ids()
        attn_init[torch.arange(num_conditions), dom] = 2.0  # softmax ~0.8 on the dominant region
        self.region_attn = nn.Parameter(attn_init)

        self.level_embedding = nn.Embedding(num_levels, embed_dim)
        self.condition_embedding = nn.Embedding(num_conditions, embed_dim)
        self.side_embedding = nn.Embedding(3, embed_dim)
        self.register_buffer("condition_side", _condition_side_ids(), persistent=False)

        fuse_dim = feat_dim * 2 + embed_dim * 3  # global + target-region + 3 embeddings
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def set_backbone_trainable(self, flag: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = flag

    def gradcam_target_layer(self) -> nn.Module:
        last_conv = None
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
        return last_conv

    def region_pool(self, feat: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Masked average pool of ``feat`` ([B,C,h,w]) per region -> ``[B,R,C]``."""
        denom = masks.sum(dim=(2, 3)).clamp_min(1e-6)  # [B,R]
        pooled = torch.einsum("bchw,brhw->brc", feat, masks)  # [B,R,C]
        return pooled / denom.unsqueeze(-1)

    def forward(
        self,
        image: torch.Tensor,
        anatomy: torch.Tensor,
        level_idx: torch.Tensor | None = None,
        condition_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition_idx is None:
            raise ValueError("condition_idx is required for the anatomy-forced classifier.")
        feat = self.encoder(image)  # [B, C, h, w]
        b, _, h, w = feat.shape
        masks = F.interpolate(anatomy.float(), size=(h, w), mode="area")  # [B, R, h, w]

        region_feats = self.region_pool(feat, masks)  # [B, R, C]
        attn = torch.softmax(self.region_attn[condition_idx], dim=-1)  # [B, R]
        target_region_feat = torch.einsum("br,brc->bc", attn, region_feats)  # [B, C]

        global_feat = feat.mean(dim=(2, 3))  # [B, C]
        if self.training and self.global_feature_dropout > 0:
            keep = (torch.rand(b, 1, device=feat.device) > self.global_feature_dropout).float()
            global_feat = global_feat * keep

        parts = [global_feat, target_region_feat]
        if level_idx is not None:
            parts.append(self.level_embedding(level_idx))
        parts.append(self.condition_embedding(condition_idx))
        side_idx = self.condition_side[condition_idx]
        parts.append(self.side_embedding(side_idx))
        return self.head(torch.cat(parts, dim=1))


def build_anatomy_forced_classifier(model_cfg: ModelConfig) -> AnatomyForcedRegionClassifier:
    """Construct an :class:`AnatomyForcedRegionClassifier` from a model config."""
    global_dropout = float(getattr(model_cfg, "global_feature_dropout", 0.5))
    return AnatomyForcedRegionClassifier(
        backbone=model_cfg.backbone,
        in_chans=model_cfg.in_chans,
        anatomy_in_chans=model_cfg.anatomy_in_chans,
        num_classes=model_cfg.num_classes,
        embed_dim=model_cfg.embed_dim,
        dropout=model_cfg.dropout,
        global_feature_dropout=global_dropout,
        pretrained=model_cfg.pretrained,
    )
