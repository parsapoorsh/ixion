"""Dataset spec dataclass — just the shape, no instances.

Concrete dataset instances live in config.py so all on-disk paths are
configured in one place.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """A train/val pair on disk, with the normalization stats that go with it."""
    name: str
    train: Path
    val: Path
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
