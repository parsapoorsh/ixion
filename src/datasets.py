"""Image loading and the rotation pair wrapper.

Image opening is delegated to image_io.open_rgb so the JXL handler and the
truncated-image tolerance are registered in exactly one place.

Each base image yields four training samples — one per rotation in
{0, 90, 180, 270}. The sample index is (image_idx, rotation_idx), so each
epoch sees a perfectly class-balanced view of the data with full coverage of
every rotation. Filesystem scanning lives in index.py; transforms and dataset
wiring live in builders.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

import constants
from utils.image_io import open_rgb


@dataclass(frozen=True)
class DatasetSpec:
    """A train/val pair on disk, with the normalization stats that go with it.

    mean/std default to None — in that case stats.get_stats samples the
    train split and caches the result. Set them explicitly to pin to fixed
    constants (e.g. ImageNet stats) and skip the scan.
    """
    name: str
    train: Path
    val: Path
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None


class ImageLoader(Dataset):
    """Returns PIL.Image (RGB) by index from a path source.

    ``files`` only needs to support ``__len__`` and ``__getitem__(idx)`` —
    anything sequence-like works (a list, an ``index.PathList`` view over a
    trie, etc.). We don't require a materialised list so the caller can
    keep the index in a compact representation.
    """

    def __init__(self, files: Sequence[Path]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Image.Image:
        return open_rgb(self.files[idx])


class RotationPairDataset(Dataset):
    """Wraps a base image source so each item is (image, rotation_label).

    For a base of N images, len = 4 * N. Sample i decomposes to
    img_idx = i // 4 and rot_idx = i % 4. Rotation label is rot_idx.
    """

    def __init__(self, base: ImageLoader, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base) * constants.NUM_CLASSES

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_idx, rot_idx = divmod(idx, constants.NUM_CLASSES)
        img = self.base[img_idx]
        if rot_idx != 0:
            img = TF.rotate(img, constants.ROTATIONS[rot_idx], expand=False)
        return self.transform(img), rot_idx
