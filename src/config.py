"""Static paths and hyperparameters for the rotation detector."""

from pathlib import Path

from datasets import DatasetSpec

# ImageNet normalization constants - left here as a convenience for anyone
# who wants to pin a DatasetSpec to fixed stats instead of computing them.
# Defaults below pass None so stats.get_stats derives mean/std from the
# train split and caches the result.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATH_TO_DATASETS = Path("/media/poweruser/data/2-datasets")

DATASETS = [
    DatasetSpec(
        name="imagenet",
        train=[
            PATH_TO_DATASETS / Path("ILSVRC2017/Data/CLS-LOC/train"),
            PATH_TO_DATASETS / Path("ILSVRC2017/Data/CLS-LOC/test"),
        ],
        val=[
            PATH_TO_DATASETS / Path("ILSVRC2017/Data/CLS-LOC/val"),
        ],
    ),
    DatasetSpec(
        name="coco",
        train=[
            PATH_TO_DATASETS / Path("coco/train2017"),
            PATH_TO_DATASETS / Path("coco/test2017"),
        ],
        val=[
            PATH_TO_DATASETS / Path("coco/val2017"),
        ],
    ),
]

# Stats sampling constants live in constants.py (datasets.py needs them and
# can't import config without a cycle). See constants.STATS_SAMPLE_FRACTION.
