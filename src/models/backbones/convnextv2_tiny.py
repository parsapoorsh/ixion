"""ConvNeXt-V2-Tiny backbone.

Trained from scratch. The timm id keeps the ``.fcmae_ft_in22k_in1k``
variant because that's the architecture we want; ``pretrained=False``
just skips loading the weights.
"""

from typing import ClassVar

import timm
from torch import nn

from models.base import Backbone, TrainDefaults


class ConvNeXtV2Tiny(Backbone):
    name: ClassVar[str] = "convnextv2_tiny"
    train_defaults: ClassVar[TrainDefaults] = TrainDefaults(
        epochs=100,
        lr=1e-3,
        weight_decay=0.05,
        label_smoothing=0.1,
        warmup_epochs=5,
        grad_clip=1.0,
        ema_decay=0.9999,
        input_size=224,
        resize_size=232,
    )

    def build(self, *, pretrained: bool) -> tuple[nn.Module, int]:
        model = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        return model, model.num_features  # 768
