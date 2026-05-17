"""Backbone implementations.

One ``Backbone`` subclass per architecture, each in its own module.
Add a new arch by dropping a sibling module and appending the class
to the re-exports below and to ``models.BACKBONES``.
"""

from models.backbones.convnextv2_tiny import ConvNeXtV2Tiny
from models.backbones.efficientnet_b0 import EfficientNetB0

__all__ = ["ConvNeXtV2Tiny", "EfficientNetB0"]
