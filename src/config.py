"""Static paths and hyperparameters for the rotation detector."""

from pathlib import Path

import torch

from datasets import DatasetSpec


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

# stats.get_stats samples this fraction of each dataset's train split.
# Changing this requires --invalidate-cache to take effect; cached stats
# don't track which fraction produced them.
STATS_SAMPLE_FRACTION = 0.05
STATS_SAMPLE_SEED = 0


# DataLoader worker processes for image decoding. Tune to the machine —
# more workers help when decode is the bottleneck, but too many concurrent
# random seeks make HDD-backed datasets slower, not faster. Each CLI also
# exposes --num-workers to override per-run.
NUM_WORKERS = 8


# Image size fed to the model. ConvNeXt-V2-Tiny ends in AdaptiveAvgPool so
# the architecture itself accepts any size; 224 is the convention from the
# pretrained checkpoint. Training from scratch leaves this as a free knob —
# bumping to 256/288 trades compute for more orientation cues.
INPUT_SIZE = 224

# Pre-crop resize. Kept slightly larger than INPUT_SIZE so RandomCrop has
# pixels to choose from; standard ConvNeXt eval recipes pair 232 with 224.
# If you change INPUT_SIZE, scale this by ~1.03–1.14×.
RESIZE_SIZE = 232
