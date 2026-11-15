"""Model building blocks.

``RotationModel`` (backbone + binary upright head) lives in ``rotation.py``,
its pooling layer in ``attention.py``, and the training-side wrappers
(``ModelEMA``, ``SAM``) alongside them.
"""

from model.attention import SpatialAttentionPool
from model.ema import ModelEMA
from model.rotation import RotationModel
from model.sam import SAM

__all__ = [
    "ModelEMA",
    "RotationModel",
    "SAM",
    "SpatialAttentionPool",
]
