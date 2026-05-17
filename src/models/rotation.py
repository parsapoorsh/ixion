"""Arch-agnostic rotation wrapper: backbone + 4-class head + temperature.

The backbone is any ``Backbone`` instance (see ``models.base``). The
head and the post-hoc temperature buffer are shared across architectures
so calibrated inference and the eventual ONNX export keep a single
forward signature regardless of which backbone is in use.
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
        pretrained: bool = False,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = backbone.build(pretrained=pretrained)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, constants.NUM_CLASSES),
        )
        # Temperature for post-hoc calibration. Not learned during training.
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def calibrated_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x) / self.temperature
