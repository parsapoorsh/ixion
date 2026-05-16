"""Static paths and hyperparameters for the rotation detector."""

from pathlib import Path

import torch

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

# stats.get_stats samples this fraction of each dataset's train split.
# Changing this requires --invalidate-cache to take effect; cached stats
# don't track which fraction produced them.
STATS_SAMPLE_FRACTION = 0.05
STATS_SAMPLE_SEED = 0
