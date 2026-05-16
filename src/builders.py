"""Factories for transforms and concatenated rotation-pair datasets.

Kept separate from the raw Dataset/transform primitives in datasets.py so the
"how the pieces wire together" lives in one place.
"""

from typing import Sequence

from torch.utils.data import ConcatDataset
from torchvision import transforms as T

import config
from datasets import ImageLoader, RotationPairDataset
from index import index_files
from spec import DatasetSpec
from stats import get_stats


def build_train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    """Photometric-only augmentation. Any spatial flip/rotate would corrupt the label."""
    return T.Compose([
        T.Resize(config.RESIZE_SIZE),
        T.RandomCrop(config.INPUT_SIZE),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def build_eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    return T.Compose([
        T.Resize(config.RESIZE_SIZE),
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_train_dataset(datasets: Sequence[DatasetSpec]) -> ConcatDataset:
    """One RotationPairDataset per source dataset, normalized with that source's
    own mean/std, then concatenated. Each image is normalized by the stats it
    was registered with, which keeps the input distribution to the model
    consistent in mean=0/std=1 space even when sources differ.
    """
    parts: list[RotationPairDataset] = []
    for spec in datasets:
        files = index_files(f"train_{spec.name}", [spec.train])
        stats = get_stats(spec)
        transform = build_train_transform(stats.mean, stats.std)
        parts.append(RotationPairDataset(ImageLoader(files), transform))
    return ConcatDataset(parts)


def build_val_dataset(datasets: Sequence[DatasetSpec]) -> ConcatDataset:
    parts: list[RotationPairDataset] = []
    for spec in datasets:
        files = index_files(f"val_{spec.name}", [spec.val])
        stats = get_stats(spec)
        transform = build_eval_transform(stats.mean, stats.std)
        parts.append(RotationPairDataset(ImageLoader(files), transform))
    return ConcatDataset(parts)
