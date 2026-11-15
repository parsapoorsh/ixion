"""Learned spatial attention pooling for convolutional feature maps."""

from __future__ import annotations

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
