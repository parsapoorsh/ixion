"""Transform factories for training and evaluation.

Transforms read INPUT_SIZE/RESIZE_SIZE from config. Dataset wiring (the
build_*_dataset functions that actually use these transforms) lives in
datasets.py so this module has no dependency on stats or filesystem
indexing, keeping the import graph acyclic.
"""

from torchvision import transforms as T

import config


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


def build_eval_preprocess() -> T.Compose:
    """Resize → CenterCrop → ToTensor — the pre-Normalize half of the eval
    transform. Shared with stats computation, which needs the same shape
    and dtype pipeline but cannot normalise (it's computing the normaliser).
    """
    return T.Compose([
        T.Resize(config.RESIZE_SIZE),
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
    ])


def build_eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    return T.Compose([
        build_eval_preprocess(),
        T.Normalize(mean, std),
    ])
