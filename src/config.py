"""Dataset layout helpers for the rotation detector.

Train/val roots are not pinned here - pass the datasets root at the CLI
(``--data-root``) and ``build_dataset_specs`` expands the standard ImageNet 2017
+ COCO 2017 layout under it. Each corpus is included only when its top-level
directory exists.
"""

from logging import getLogger
from pathlib import Path
from typing import TypedDict

logger = getLogger(__name__)


class DatasetSpec(TypedDict):
    """A train/val pair on disk.

    Normalization stats are not pinned here - ``core.build_dataset``
    samples the train split and caches the result unless explicit
    mean/std are passed.
    """

    name: str
    train: list[Path]
    val: list[Path]


def build_dataset_specs(root: Path) -> list[DatasetSpec]:
    """Build ImageNet / COCO specs under *root* for corpora that are present.

    Expected layout (each top-level dir is optional)::

        root/
          ILSVRC2017/Data/CLS-LOC/{train,test,val}/
          coco/{train2017,test2017,val2017}/
    """
    result: list[DatasetSpec] = []

    imagenet_path = root / "ILSVRC2017"
    if imagenet_path.exists():
        logger.info("found ImageNet2017 at: %s", imagenet_path)
        result.append(
            {
                "name": "imagenet",
                "train": [
                    imagenet_path / "Data" / "CLS-LOC" / "train",
                    imagenet_path / "Data" / "CLS-LOC" / "test",
                ],
                "val": [
                    imagenet_path / "Data" / "CLS-LOC" / "val",
                ],
            }
        )

    coco_path = root / "coco"
    if coco_path.exists():
        logger.info("found coco at: %s", coco_path)
        result.append(
            {
                "name": "coco",
                "train": [
                    coco_path / "train2017",
                    coco_path / "test2017",
                ],
                "val": [
                    coco_path / "val2017",
                ],
            }
        )

    if not result:
        raise FileNotFoundError(
            f"no known datasets under {root} "
            "(expected ILSVRC2017/ and/or coco/)"
        )
    return result


# Stats sampling constants live in constants.py so the data pipeline and
# model code can share them without import cycles. See
# constants.STATS_SAMPLE_FRACTION.
