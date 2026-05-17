"""ConvNeXt-V2-Tiny backbone factory.

Trained from scratch — ``pretrained=False`` is the default at the
``RotationModel`` level. The timm model id still references the
``.fcmae_ft_in22k_in1k`` variant because that's the architecture we
want; the flag just skips loading the weights.
"""

import timm
from torch import nn


def build(*, pretrained: bool) -> tuple[nn.Module, int]:
    model = timm.create_model(
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    return model, model.num_features  # 768 for ConvNeXt-V2-Tiny
