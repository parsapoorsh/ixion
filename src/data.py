"""Image loading, transforms, and the rotation pair wrapper.

PIL is used for decoding; pillow-jxl-plugin auto-registers on import so
PIL.Image.open transparently handles JPEG-XL alongside the formats PIL
already supports.

The training dataset is built from ImageNet + COCO. Each base image yields four
training samples — one per rotation in {0, 90, 180, 270}. The sample index is
(image_idx, rotation_idx), so each epoch sees a perfectly class-balanced view
of the data with full coverage of every rotation. One RotationPairDataset is
constructed per source so each image is normalized with its source's stats.
"""

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import pillow_jxl  # noqa: F401  side effect: registers JXL handler with PIL
import torch
from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import config
from datasets import DatasetSpec

ImageFile.LOAD_TRUNCATED_IMAGES = True
log = logging.getLogger(__name__)


def scan_images(roots: Sequence[Path]) -> list[Path]:
    """Recursively gather every regular file under the given roots.

    No extension filter — the dataset directories are expected to contain only
    images, and PIL will raise at load time on anything it can't decode.
    """
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for entry in root.rglob("*"):
            if entry.is_file():
                out.append(entry)
    return out


def load_or_build_index(
    name: str,
    roots: Sequence[Path],
) -> list[Path]:
    """Build (and cache to disk) the file list for a dataset split.

    Scanning 1.2M files takes tens of seconds; caching avoids repeating it.
    """
    config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.INDEX_CACHE_DIR / f"index_{name}.json"
    if cache.exists():
        with cache.open() as f:
            return [Path(p) for p in json.load(f)]

    t0 = time.time()
    files = scan_images(roots)
    dt = time.time() - t0
    log.info("scanned %s: %d files in %.1fs", name, len(files), dt)

    with cache.open("w") as f:
        json.dump([str(p) for p in files], f)
    return files


class ImageList(Dataset):
    """Returns PIL.Image (RGB) by index from a flat list of image files."""

    def __init__(self, files: list[Path]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Image.Image:
        return Image.open(self.files[idx]).convert("RGB")


class RotationPairDataset(Dataset):
    """Wraps a base image source so each item is (image, rotation_label).

    For a base of N images, len = 4 * N. Sample i decomposes to
    img_idx = i // 4 and rot_idx = i % 4. Rotation label is rot_idx.
    """

    def __init__(self, base: ImageList, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base) * config.NUM_CLASSES

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_idx, rot_idx = divmod(idx, config.NUM_CLASSES)
        img = self.base[img_idx]
        if rot_idx != 0:
            img = TF.rotate(img, config.ROTATIONS[rot_idx], expand=False)
        return self.transform(img), rot_idx


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


def build_eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    return T.Compose([
        T.Resize(config.RESIZE_SIZE),
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_train_dataset(datasets: Sequence[DatasetSpec]) -> ConcatDataset:
    """One RotationPairDataset per source dataset, normalized with that source's
    own mean/std, then concatenated. Each image is normalized by the stats it
    was registered with, which keeps the input distribution to the model
    consistent in mean=0/std=1 space even when sources differ.
    """
    parts: list[RotationPairDataset] = []
    for spec in datasets:
        files = load_or_build_index(f"train_{spec.name}", [spec.train])
        transform = build_train_transform(spec.mean, spec.std)
        parts.append(RotationPairDataset(ImageList(files), transform))
    return ConcatDataset(parts)


def build_val_dataset(datasets: Sequence[DatasetSpec]) -> ConcatDataset:
    parts: list[RotationPairDataset] = []
    for spec in datasets:
        files = load_or_build_index(f"val_{spec.name}", [spec.val])
        transform = build_eval_transform(spec.mean, spec.std)
        parts.append(RotationPairDataset(ImageList(files), transform))
    return ConcatDataset(parts)
