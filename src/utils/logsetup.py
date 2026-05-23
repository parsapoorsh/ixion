"""Tiny logging setup shared by every CLI entry point.

Importing this module is enough to configure the root logger consistently.
Kept in its own file so changing the format/level happens once.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so an active progress bar
    isn't clobbered by interleaved log lines."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup_logging(
    logs_dir: Path,
    level: int = logging.INFO,
) -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
