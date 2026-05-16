"""Static paths and hyperparameters for the rotation detector."""

from pathlib import Path

import torch
from torchvision import transforms as T

from spec import DatasetSpec


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ImageNet normalization constants — left here as a convenience for anyone
# who wants to pin a DatasetSpec to fixed stats instead of computing them.
# Defaults below pass None so stats.get_stats derives mean/std from the
# train split and caches the result.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASETS = [
    DatasetSpec(
        name="imagenet",
        train=Path("/mnt/archive/2-stuff/datasets/ILSVRC2017/Data/CLS-LOC/train"),
        val=Path("/mnt/archive/2-stuff/datasets/ILSVRC2017/Data/CLS-LOC/val"),
        mean=None,
        std=None,
    ),
    DatasetSpec(
        name="coco",
        train=Path("/mnt/archive/2-stuff/datasets/coco/train2017"),
        val=Path("/mnt/archive/2-stuff/datasets/coco/val2017"),
        mean=None,
        std=None,
    ),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INDEX_CACHE_DIR = PROJECT_ROOT / "cache"

INPUT_SIZE = 224
RESIZE_SIZE = 232

ROTATIONS = (0, 90, 180, 270)
NUM_CLASSES = len(ROTATIONS)


def build_train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    """Photometric-only augmentation. Any spatial flip/rotate would corrupt the label."""
    return T.Compose([
        T.Resize(RESIZE_SIZE),
        T.RandomCrop(INPUT_SIZE),
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
        T.Resize(RESIZE_SIZE),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
