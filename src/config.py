"""Static paths and hyperparameters for the rotation detector."""

from pathlib import Path

from datasets import DatasetSpec


# ImageNet normalization constants - left here as a convenience for anyone
# who wants to pin a DatasetSpec to fixed stats instead of computing them.
# Defaults below pass None so stats.get_stats derives mean/std from the
# train split and caches the result.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATH_TO_DATASETS = Path("/media/poweruser/data/2-stuff/datasets")

DATASETS = [
    DatasetSpec(
        name="imagenet",
        train=[
            PATH_TO_DATASETS / Path("ILSVRC2017/Data/CLS-LOC/train"),
        ],
        val=[
            PATH_TO_DATASETS / Path("ILSVRC2017/Data/CLS-LOC/val"),
        ],
        mean=None,
        std=None,
    ),
    DatasetSpec(
        name="coco",
        train=[
            PATH_TO_DATASETS / Path("coco/train2017"),
        ],
        val=[
            PATH_TO_DATASETS / Path("coco/val2017"),
        ],
        mean=None,
        std=None,
    ),
]

# stats.get_stats samples this fraction of each dataset's train split.
# Changing this requires --invalidate-cache to take effect; cached stats
# don't track which fraction produced them.
STATS_SAMPLE_FRACTION = 0.20
STATS_SAMPLE_SEED = 0
