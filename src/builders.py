"""Factories for concatenated rotation-pair datasets.

Transform builders live in config.py alongside the input sizes and
augmentation knobs they parameterise. This file just wires datasets
together using those transforms.
"""

from typing import Sequence

from torch.utils.data import ConcatDataset

from config import build_eval_transform, build_train_transform
from datasets import ImageLoader, RotationPairDataset
from index import index_files
from spec import DatasetSpec
from stats import get_stats


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
