"""EfficientNet-B0 backbone.

~5.3M params, ~0.4 GFLOPs at 224² — roughly 10× less compute than
ConvNeXt-V2-Tiny. Picked when the loop is GPU-bound, not I/O-bound.
Trained from scratch.
"""

from typing import ClassVar

import timm
from torch import nn

from models.base import Backbone, TrainDefaults


class EfficientNetB0(Backbone):
    name: ClassVar[str] = "efficientnet_b0"
    # ~7× smaller activations than ConvNeXt-V2-Tiny → much larger
    # batch fits the same card. Other recipe knobs are conservative
    # shared defaults; tune once a real run shows what wants to move.
    train_defaults: ClassVar[TrainDefaults] = TrainDefaults(
        epochs=100,
        batch_size=256,
        lr=1e-3,
        weight_decay=0.05,
        label_smoothing=0.1,
        warmup_epochs=5,
        grad_clip=1.0,
        ema_decay=0.9999,
    )

    def build(self, *, pretrained: bool) -> tuple[nn.Module, int]:
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        return model, model.num_features  # 1280
