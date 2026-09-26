"""Single entry point for opening images.

Centralises two easy-to-forget side effects that bit us before:

  * ``import pillow_jxl`` registers the JXL handler with PIL. Without it,
    every ``.jxl`` file fails ``Image.open``.
  * ``ImageFile.LOAD_TRUNCATED_IMAGES = True`` keeps PIL from raising on
    images whose last few bytes are missing, which happens occasionally in
    the wild.

Any module that needs to read an image should import ``open_rgb`` from
here instead of calling ``PIL.Image.open`` directly.
"""

from pathlib import Path

import pillow_jxl  # noqa: F401  side effect: registers JXL handler with PIL
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_rgb(
    path: str | Path,
    preview_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Open ``path``, apply its EXIF orientation, and return it as RGB.

    Camera files frequently store pixels sideways and rely on EXIF for
    display; transposing here means the model scores the orientation a
    user or Immich actually sees.

    ``preview_size`` asks the decoder for a reduced-resolution image
    roughly this size (JPEG drafts via DCT scaling; a no-op for other
    formats). The stats pass uses it because mean/std only need eval-size
    pixels; training and inference always pass full resolution.
    """
    img = Image.open(path)
    if preview_size is not None:
        img.draft("RGB", preview_size)
    return ImageOps.exif_transpose(img).convert("RGB")
