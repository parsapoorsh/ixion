"""Backbone implementations.

One ``Backbone`` subclass per architecture, each in its own module.
Re-exported here so callers can ``from models.backbones import
EfficientNetB0`` without reaching into the per-arch module.
"""

from models.backbones.efficientnet_b0 import EfficientNetB0

__all__ = ["EfficientNetB0"]
