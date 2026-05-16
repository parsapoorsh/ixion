"""Image loading and the rotation pair wrapper.

Image opening is delegated to image_io.open_rgb so the JXL handler and the
truncated-image tolerance are registered in exactly one place.

Each base image yields four training samples — one per rotation in
{0, 90, 180, 270}. The sample index is (image_idx, rotation_idx), so each
epoch sees a perfectly class-balanced view of the data with full coverage of
every rotation. Filesystem scanning lives in index.py; transforms and dataset
wiring live in builders.py.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

import config
from image_io import open_rgb


class ImageLoader(Dataset):
    """Returns PIL.Image (RGB) by index from a flat list of image files."""

    def __init__(self, files: list[Path]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Image.Image:
        return open_rgb(self.files[idx])


class RotationPairDataset(Dataset):
    """Wraps a base image source so each item is (image, rotation_label).

    For a base of N images, len = 4 * N. Sample i decomposes to
    img_idx = i // 4 and rot_idx = i % 4. Rotation label is rot_idx.
    """

    def __init__(self, base: ImageLoader, transform) -> None:
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
