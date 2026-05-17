"""Model package.

Generic rotation wrapper (head + temperature) lives in ``rotation.py``.
Backbone implementations live under ``models.backbones`` — one
``Backbone`` subclass per architecture.

To add a new architecture: create ``models/backbones/<name>.py`` with
a class ``<Name>(Backbone)`` (set ``name``, ``train_defaults``,
implement ``build``) and re-export it from
``models/backbones/__init__.py``.
"""

from models.base import Backbone, TrainDefaults
from models.rotation import RotationModel

__all__ = ["RotationModel", "Backbone", "TrainDefaults"]
