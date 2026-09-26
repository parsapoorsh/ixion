"""Rotation model: EfficientNet-B0 backbone + binary upright head.

A single output neuron predicts whether the image is upright (0°). Trained
from scratch (no ImageNet weights). The backbone returns a spatial feature
map; either spatial attention or global average pooling aggregates it
before LayerNorm -> Dropout -> Linear.
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class SpatialAttentionPool(nn.Module):
    """Learned spatial weights over a feature map, then weighted sum.

    A 1x1 conv produces a single score per location; softmax over HxW turns
    that into a soft mask so the model can emphasize upright cues (horizon,
    faces, text) and downweight bland regions (sky, grass).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C)
        weights = self.score(x).flatten(2).softmax(dim=-1)
        return (x.flatten(2) * weights).sum(dim=-1)


class RotationModel(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
        pooling: str = "avg",
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )
        feat_dim: int = self.backbone.num_features  # 1280 # pyright: ignore[reportAssignmentType]
        if pooling == "attention":
            self.pool: nn.Module = SpatialAttentionPool(feat_dim)
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.pooling == "avg":
            pooled = self.pool(feats).flatten(1)
        else:
            pooled = self.pool(feats)
        return self.head(pooled).squeeze(-1)
