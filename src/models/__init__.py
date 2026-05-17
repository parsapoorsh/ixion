"""Model package.

Generic rotation wrapper (head + temperature) lives in ``rotation.py``.
Each backbone architecture is its own module here, registered in
``BACKBONES`` so ``RotationModel(backbone="<name>")`` can resolve it.

To add a new architecture: create ``models/<name>.py`` exposing
``build(*, pretrained: bool) -> tuple[nn.Module, int]`` (the backbone
and its output feature dimension), then register it below.
"""

from models import convnextv2_tiny
from models.rotation import RotationModel

BACKBONES = {
    "convnextv2_tiny": convnextv2_tiny.build,
}

__all__ = ["RotationModel", "BACKBONES"]
