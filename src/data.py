"""Transforms, the rotation dataset, and DataLoader construction."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import torch
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import constants
from config import DatasetSpec
from stats import MeanStd, compute_stats
from utils.image_io import open_rgb
from utils.index import cache_fingerprint, index_files

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    erasing_p: float = 0.25,
    crop_policy: str = "random",
) -> T.Compose:
    """Label-safe augmentation only. No spatial flip/rotate (would corrupt labels)."""
    if crop_policy == "none":
        spatial = [T.Resize(constants.INPUT_SIZE)]
    elif crop_policy == "random":
        spatial = [T.Resize(constants.RESIZE_SIZE), T.RandomCrop(constants.INPUT_SIZE)]
    elif crop_policy == "center":
        spatial = [T.Resize(constants.RESIZE_SIZE), T.CenterCrop(constants.INPUT_SIZE)]
    else:
        raise ValueError(f"unknown crop policy: {crop_policy}")

    transforms = [
        *spatial,
        T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.05,
        ),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
    if erasing_p > 0:
        # After normalize so erased patches are mean-gray in model space.
        transforms.append(
            T.RandomErasing(
                p=erasing_p,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0,
            )
        )
    return T.Compose(transforms)


display_preprocess = T.Compose(
    [
        T.Resize(constants.RESIZE_SIZE),
        T.CenterCrop(constants.INPUT_SIZE),
    ]
)
eval_preprocess = T.Compose(
    [
        *display_preprocess.transforms,
        T.ToTensor(),
    ]
)


def build_eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    return T.Compose(
        [
            *eval_preprocess.transforms,
            T.Normalize(mean, std),
        ]
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class RotationPairDataset(Dataset[tuple[torch.Tensor, float]]):
    """Wraps a file list so each item is (image, is_upright_label).

    For N images, len = 4 * N. Sample i decomposes to img_idx = i // 4 and
    rot_idx = i % 4. The label is 1.0 when rot_idx == 0 (upright), else 0.0.
    """

    def __init__(self, files: Sequence[Path], transform: Callable) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files) * len(constants.ROTATIONS)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        img_idx, rot_idx = divmod(idx, len(constants.ROTATIONS))
        # Fail loudly on unreadable files: wrapping around (the previous
        # behaviour) duplicated samples and skewed validation metrics.
        try:
            img = open_rgb(self.files[img_idx])
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"unreadable image {self.files[img_idx]}: {exc}") from exc
        tensor = self.transform(img)
        if rot_idx != 0:
            tensor = TF.rotate(
                tensor,
                constants.ROTATIONS[rot_idx],
                expand=True,
            )

        prob = 1.0 if rot_idx == 0 else 0.0
        return tensor, prob


def build_dataset(
    split: Literal["train", "val"],
    datasets: Sequence[DatasetSpec],
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    mean_std: MeanStd | None = None,
    invalidate_cache: bool = False,
    erasing_p: float = 0.25,
    crop_policy: str = "random",
) -> tuple[RotationPairDataset, MeanStd]:
    roots = [root for ds in datasets for root in (ds["train"] if split == "train" else ds["val"])]
    files = index_files(
        name=split,
        roots=roots,
        cache_dir=constants.INDEX_CACHE_DIR,
        invalidate_cache=invalidate_cache,
    )

    if mean_std is None:
        stats_cache = constants.INDEX_CACHE_DIR / f"stats_{split}.json"
        stats_fingerprint = cache_fingerprint(
            "stats",
            sorted(str(root) for root in roots),
            constants.INPUT_SIZE,
            constants.RESIZE_SIZE,
            constants.STATS_SAMPLE_FRACTION,
            constants.STATS_SAMPLE_SEED,
        )
        sample_count = min(
            len(files),
            max(1, int(len(files) * constants.STATS_SAMPLE_FRACTION)),
        )
        if sample_count >= len(files):
            sample_indices = None
        else:
            rng = random.Random(constants.STATS_SAMPLE_SEED)
            sample_indices = sorted(rng.sample(range(len(files)), sample_count))
        logger.info(
            "computing dataset stats for %s split on %d/%d files",
            split,
            sample_count,
            len(files),
        )
        mean_std = compute_stats(
            files=files,
            num_workers=num_workers,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            sample_indices=sample_indices,
            cache_path=stats_cache,
            invalidate_cache=invalidate_cache,
            fingerprint=stats_fingerprint,
        )
        logger.info("stats - mean: %s | std: %s", *mean_std)

    mean, std = mean_std
    transform = (
        build_train_transform(
            mean=mean,
            std=std,
            erasing_p=erasing_p,
            crop_policy=crop_policy,
        )
        if split == "train"
        else build_eval_transform(mean=mean, std=std)
    )
    return RotationPairDataset(files=files, transform=transform), mean_std


def build_loader(
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
) -> DataLoader:
    """DataLoader with the project's worker and pinning defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=device.type == "cuda",
        shuffle=shuffle,
    )
