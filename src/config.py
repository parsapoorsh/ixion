"""Static paths and hyperparameters for the rotation detector."""

from pathlib import Path

import torch

from spec import DatasetSpec


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ImageNet normalization is the canonical choice for any model fine-tuned from
# ImageNet-pretrained weights. Both datasets use it because the backbone is
# pretrained on ImageNet; override per-dataset if you swap to a backbone
# pretrained on a different distribution.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASETS = [
    DatasetSpec(
        name="imagenet",
        train=Path("/mnt/archive/2-stuff/datasets/ILSVRC2017/Data/CLS-LOC/train"),
        val=Path("/mnt/archive/2-stuff/datasets/ILSVRC2017/Data/CLS-LOC/val"),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
    DatasetSpec(
        name="coco",
        train=Path("/mnt/archive/2-stuff/datasets/coco/train2017"),
        val=Path("/mnt/archive/2-stuff/datasets/coco/val2017"),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INDEX_CACHE_DIR = PROJECT_ROOT / "cache"

INPUT_SIZE = 224
RESIZE_SIZE = 232
ROTATIONS = (0, 90, 180, 270)
NUM_CLASSES = len(ROTATIONS)
