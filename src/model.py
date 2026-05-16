"""ConvNeXt-V2-Tiny backbone + 4-class rotation head.

Trained from scratch — ``pretrained=False`` is the default. The timm model
id still references the ``.fcmae_ft_in22k_in1k`` variant because that's the
architecture we want; the flag just skips loading the weights. Backbone and
head learn at the same rate (no layer-wise LR split, since both are
randomly initialised).

The temperature parameter is stored on the module so calibrated inference
and the eventual ONNX export can use a single forward signature.
Temperature is fit post-hoc on the val set (frozen during training).
"""

import timm
import torch
from torch import nn

import constants


class RotationModel(nn.Module):
    def __init__(self, pretrained: bool = False, drop_rate: float = 0.2) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features  # 768 for ConvNeXt-V2-Tiny
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
