"""Arch-agnostic rotation wrapper: backbone + 4-class head.

The backbone is any ``Backbone`` instance (see ``models.base``). The
head is shared across architectures so the forward signature stays
identical regardless of which backbone is in use.
"""

from __future__ import annotations

import torch
from torch import nn

import constants
from models.base import Backbone


class RotationModel(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        pretrained: bool,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = backbone.build(pretrained=pretrained)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, constants.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)
