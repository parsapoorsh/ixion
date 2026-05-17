"""DatasetSpec, image loading, the rotation pair wrapper, and the
concatenated train/val dataset builders.

Image opening is delegated to image_io.open_rgb so the JXL handler and the
truncated-image tolerance are registered in exactly one place.

Each base image yields four training samples — one per rotation in
{0, 90, 180, 270}. The sample index is (image_idx, rotation_idx), so each
epoch sees a perfectly class-balanced view of the data with full coverage of
every rotation. Filesystem scanning lives in utils.index; transform recipes
live in builders.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from PIL import UnidentifiedImageError
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import functional as TF

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Sequence

    from PIL import Image

import constants
import stats
from builders import build_eval_transform, build_train_transform
from utils.image_io import open_rgb
from utils.index import index_files

logger = logging.getLogger(__name__)


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
    """Loads images by index from a path source, optionally applying a transform.

    Without a transform, returns a ``PIL.Image`` (used by ``RotationPairDataset``
    which applies its own transform after rotation). With a transform, returns the
    transformed tensor (used by the stats DataLoader directly).

    ``files`` only needs to support ``__len__`` and ``__getitem__(idx)`` —
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
    """Wraps a base image source so each item is (image, rotation_label).

    For a base of N images, len = 4 * N. Sample i decomposes to
    img_idx = i // 4 and rot_idx = i % 4. Rotation label is rot_idx.
    """

    def __init__(self, base: ImageLoader, transform: Callable) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base) * constants.NUM_CLASSES

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_idx, rot_idx = divmod(idx, constants.NUM_CLASSES)
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
            tensor = TF.rotate(tensor, constants.ROTATIONS[rot_idx])
        return tensor, rot_idx


def build_train_dataset(
    specs: Sequence[DatasetSpec],
    input_size: int,
    resize_size: int,
) -> ConcatDataset:
    """One RotationPairDataset per source dataset, normalized with that source's
    own mean/std, then concatenated. Each image is normalized by the stats it
    was registered with, which keeps the input distribution to the model
    consistent in mean=0/std=1 space even when sources differ.
    """
    parts: list[RotationPairDataset] = []
    for spec in specs:
        files = index_files(f"train_{spec.name}", [spec.train])
        channel_stats = stats.get_stats(
            spec,
            input_size=input_size,
            resize_size=resize_size,
        )
        transform = build_train_transform(
            mean=channel_stats.mean,
            std=channel_stats.std,
            input_size=input_size,
            resize_size=resize_size,
        )
        parts.append(RotationPairDataset(ImageLoader(files), transform))
    return ConcatDataset(parts)


def build_val_dataset(
    specs: Sequence[DatasetSpec],
    input_size: int,
    resize_size: int,
) -> ConcatDataset:
    parts: list[RotationPairDataset] = []
    for spec in specs:
        files = index_files(f"val_{spec.name}", [spec.val])
        channel_stats = stats.get_stats(
            spec,
            input_size=input_size,
            resize_size=resize_size,
        )
        transform = build_eval_transform(
            mean=channel_stats.mean,
            std=channel_stats.std,
            input_size=input_size,
            resize_size=resize_size,
        )
        parts.append(RotationPairDataset(ImageLoader(files), transform))
    return ConcatDataset(parts)
