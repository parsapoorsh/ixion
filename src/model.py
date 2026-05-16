"""ConvNeXt-V2-Tiny backbone + 4-class rotation head.

The temperature parameter is stored on the module so calibrated inference and
the ONNX export can use a single forward signature. Temperature is fit
post-hoc (frozen during training).
"""

import timm
import torch
from torch import nn

import config


class RotationModel(nn.Module):
    def __init__(self, pretrained: bool = True, drop_rate: float = 0.2) -> None:
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
            nn.Linear(feat_dim, config.NUM_CLASSES),
        )
        # Temperature for post-hoc calibration. Not learned during training.
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def calibrated_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x) / self.temperature

    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float):
        """Layer-wise LR: backbone gets a lower rate than the freshly initialised head."""
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ]
