"""DatasetSpec, image loading, the rotation pair wrapper, and the
concatenated train/val dataset builders.

Image opening is delegated to image_io.open_rgb so the JXL handler and the
truncated-image tolerance are registered in exactly one place.

Each base image yields four training samples - one per rotation in
{0, 90, 180, 270}. The sample index is (image_idx, rotation_idx). The label
is binary: 1.0 when the view is upright (rotation 0), else 0.0 - so each
base image contributes one positive and three negatives. Filesystem scanning
lives in utils.index; transform recipes live in builders.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Sequence

    from PIL import Image

import constants
from utils.image_io import open_rgb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """A train/val pair on disk, with the normalization stats that go with it.

    mean/std default to None - in that case stats.get_stats samples the
    train split and caches the result. Set them explicitly to pin to fixed
    constants (e.g. ImageNet stats) and skip the scan.
    """
    name: str
    train: list[Path]
    val: list[Path]
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None


class ImageLoader(Dataset):
    """Loads images by index from a path source, optionally applying a transform.

    Without a transform, returns a ``PIL.Image`` (used by ``RotationPairDataset``
    which applies its own transform after rotation). With a transform, returns the
    transformed tensor (used by the stats DataLoader directly).

    ``files`` only needs to support ``__len__`` and ``__getitem__(idx)`` -
    anything sequence-like works (a list, an ``index.PathList`` view over a
    trie, etc.). We don't require a materialised list so the caller can
    keep the index in a compact representation.
    """

    def __init__(self, files: Sequence[Path], transform: Callable | None = None) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Image.Image | torch.Tensor:
        img = open_rgb(self.files[idx])
        if self.transform is not None:
            return self.transform(img)
        return img


class RotationPairDataset(Dataset):
    """Wraps a base image source so each item is (image, is_upright_label).

    For a base of N images, len = 4 * N. Sample i decomposes to
    img_idx = i // 4 and rot_idx = i % 4. The label is 1.0 when rot_idx == 0
    (upright), else 0.0.
    """

    def __init__(self, base: ImageLoader, transform: Callable) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base) * len(constants.ROTATIONS)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        img_idx, rot_idx = divmod(idx, len(constants.ROTATIONS))
        n = len(self.base)
        for offset in range(n):
            probe = (img_idx + offset) % n
            try:
                img = self.base[probe]
                break
            except (UnidentifiedImageError, OSError) as exc:
                logger.warning(
                    "skipping unreadable image %s: %s",
                    self.base.files[probe], exc,
                )
        else:
            raise RuntimeError("no readable images remain in the dataset")
        tensor = self.transform(img)
        if rot_idx != 0:
            tensor = TF.rotate(tensor, constants.ROTATIONS[rot_idx], expand=True)

        prob = 1.0 if rot_idx == 0 else 0.0
        return tensor, prob
