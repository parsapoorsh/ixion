"""Rotation model: EfficientNet-B0 backbone + binary upright head.

A single output neuron predicts whether the image is upright (0°). Trained
from scratch (no ImageNet weights). The backbone is built via timm; the head
is AdaptiveAvgPool (already inside timm with ``global_pool='avg'``) →
LayerNorm → Dropout → Linear.
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class RotationModel(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim: int = self.backbone.num_features  # 1280 # pyright: ignore[reportAssignmentType]
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats).squeeze(-1)
