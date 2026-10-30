"""Dataset layout helpers for the rotation detector.

Train/val roots are not pinned here - pass the datasets root at the CLI
(``--data-root``) and ``build_datasets`` expands the standard ImageNet 2017
+ COCO 2017 layout under it. Each corpus is included only when its top-level
directory exists.
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
    """Build ImageNet / COCO specs under *root* for corpora that are present.

    Expected layout (each top-level dir is optional)::

        root/
          ILSVRC2017/Data/CLS-LOC/{train,test,val}/
          coco/{train2017,test2017,val2017}/
    """
    root = Path(root)
    result: list[DatasetSpec] = []

    imagenet_path = root / "ILSVRC2017"
    if imagenet_path.exists():
        result.append(
            DatasetSpec(
                name="imagenet",
                train=[
                    imagenet_path / "Data" / "CLS-LOC" / "train",
                    imagenet_path / "Data" / "CLS-LOC" / "test",
                ],
                val=[
                    imagenet_path / "Data" / "CLS-LOC" / "val",
                ],
            )
        )

    coco_path = root / "coco"
    if coco_path.exists():
        result.append(
            DatasetSpec(
                name="coco",
                train=[
                    coco_path / "train2017",
                    coco_path / "test2017",
                ],
                val=[
                    coco_path / "val2017",
                ],
            )
        )

    if not result:
        raise FileNotFoundError(
            f"no known datasets under {root} "
            "(expected ILSVRC2017/ and/or coco/)"
        )
    return result


# Stats sampling constants live in constants.py (datasets.py needs them and
# can't import config without a cycle). See constants.STATS_SAMPLE_FRACTION.
