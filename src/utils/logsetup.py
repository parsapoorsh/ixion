"""Tiny logging setup shared by every CLI entry point.

Importing this module is enough to configure the root logger consistently.
Kept in its own file so changing the format/level happens once.
"""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
