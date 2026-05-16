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
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_rgb(path: str | Path) -> Image.Image:
    """Open ``path`` and return it as an RGB ``PIL.Image``."""
    return Image.open(path).convert("RGB")
