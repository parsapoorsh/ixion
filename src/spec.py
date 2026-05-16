"""Dataset spec dataclass — just the shape, no instances.

Concrete dataset instances live in config.py so all on-disk paths are
configured in one place.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """A train/val pair on disk, with the normalization stats that go with it.

    mean/std may be None — in that case stats.get_stats samples the train
    split and caches the result. Set them explicitly to pin to fixed
    constants (e.g. ImageNet stats) and skip the scan.
    """
    name: str
    train: Path
    val: Path
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None
