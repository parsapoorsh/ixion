"""Shared types for the models package.

Lives in its own module so each backbone file can import ``Backbone``
and ``TrainDefaults`` without creating a cycle against
``models/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class TrainDefaults:
    """Per-backbone training recipe defaults.

    Used as the field set ``TrainConfig`` inherits from, so adding a
    recipe knob is a one-place change.
    """
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    label_smoothing: float
    warmup_epochs: int
    grad_clip: float
    ema_decay: float
    input_size: int  # crop fed to the model; arch's native size (timm convention)
    resize_size: int  # pre-crop resize; ~1.03–1.14× input_size for RandomCrop wiggle room


class Backbone(ABC):
    """One architecture. Subclass per arch, set ``name`` and
    ``train_defaults``, implement ``build``. Instances are stateless;
    the registry in ``models/__init__.py`` constructs one per arch.
    """
    name: ClassVar[str]
    train_defaults: ClassVar[TrainDefaults]

    @abstractmethod
    def build(self, *, pretrained: bool) -> tuple[nn.Module, int]:
        """Return the backbone module and its output feature dim."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
