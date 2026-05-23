"""Transform factories and dataset builders."""

from __future__ import annotations

from typing import Literal

import torch
from torchvision import transforms as T

import config
import constants
from datasets import ImageLoader, RotationPairDataset
from stats import compute_stats
from utils.index import index_files


def build_train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    """Photometric-only augmentation. Any spatial flip/rotate would corrupt the label."""
    return T.Compose([
        T.Resize(constants.RESIZE_SIZE),
        T.RandomCrop(constants.INPUT_SIZE),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


eval_preprocess = T.Compose([
    T.Resize(constants.RESIZE_SIZE),
    T.CenterCrop(constants.INPUT_SIZE),
    T.ToTensor(),
])


def build_eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    return T.Compose([
        *eval_preprocess.transforms,
        T.Normalize(mean, std),
    ])


def build_dataset(
    split: Literal["train", "val"],
    device: torch.device,
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    invalidate_cache: bool = False,
) -> RotationPairDataset:
    roots = [
        root
        for ds in config.DATASETS
        for root in (ds.train if split == "train" else ds.val)
    ]
    files = index_files(
        name=split,
        roots=roots,
        cache_dir=constants.INDEX_CACHE_DIR,
        invalidate_cache=invalidate_cache,
    )

    mean, std = compute_stats(
        files,
        device=device,
        num_workers=num_workers,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
    )

    transform = (
        build_train_transform(mean=mean, std=std)
        if split == "train"
        else build_eval_transform(mean=mean, std=std)
    )
    return RotationPairDataset(
        base=ImageLoader(files=files),
        transform=transform,
    )
