"""Dataset layout helpers for the rotation detector.

Train/val roots are not pinned here - pass the datasets root at the CLI
(``--data-root``) and ``build_datasets`` expands the standard ImageNet 2017
+ COCO 2017 layout under it.
"""

from pathlib import Path

from datasets import DatasetSpec

# ImageNet normalization constants - left here as a convenience for anyone
# who wants to pin a DatasetSpec to fixed stats instead of computing them.
# Defaults pass None so stats.get_stats derives mean/std from the train
# split and caches the result.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_datasets(root: Path) -> list[DatasetSpec]:
    """Build the default ImageNet + COCO specs under *root*.

    Expected layout::

        root/
          ILSVRC2017/Data/CLS-LOC/{train,test,val}/
          coco/{train2017,test2017,val2017}/
    """
    root = Path(root)
    return [
        DatasetSpec(
            name="imagenet",
            train=[
                root / "ILSVRC2017/Data/CLS-LOC/train",
                root / "ILSVRC2017/Data/CLS-LOC/test",
            ],
            val=[
                root / "ILSVRC2017/Data/CLS-LOC/val",
            ],
        ),
        DatasetSpec(
            name="coco",
            train=[
                root / "coco/train2017",
                root / "coco/test2017",
            ],
            val=[
                root / "coco/val2017",
            ],
        ),
    ]


# Stats sampling constants live in constants.py (datasets.py needs them and
# can't import config without a cycle). See constants.STATS_SAMPLE_FRACTION.
