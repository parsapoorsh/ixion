"""Model package.

``RotationModel`` (backbone + binary upright head) lives in ``rotation.py``;
``ModelEMA`` lives in ``ema.py``.
"""

from models.ema import ModelEMA
from models.rotation import RotationModel

__all__ = ["RotationModel", "ModelEMA"]
