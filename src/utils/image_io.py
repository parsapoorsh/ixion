"""Single entry point for opening images.

Centralises two easy-to-forget side effects that bit us before:

  * ``import pillow_jxl`` registers the JXL handler with PIL. Without it,
    every ``.jxl`` file silently fails ``Image.open``.
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


def open_rgb(path: str | Path) -> Image.Image:
    """Open ``path``, apply its EXIF orientation, and return it as RGB.

    Camera files frequently store pixels sideways and rely on EXIF for
    display; transposing here means the model scores the orientation a
    user or Immich actually sees.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")
