"""Tiny logging setup shared by every CLI entry point.

Importing this module is enough to configure the root logger consistently.
Kept in its own file so changing the format/level happens once.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

import constants


class TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so an active progress bar
    isn't clobbered by interleaved log lines."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup_logging(
    level: int = logging.INFO,
    log_path: Path | None = None,
    console: bool = True,
) -> Path:
    """Configure the root logger. Returns the file path being written to so
    a parent process can hand it to a child (mp.Process target) and have
    both write to the same file. FileHandler uses append mode and log lines
    are well under PIPE_BUF (4KB), so concurrent appenders on POSIX are
    safe.

    Pass ``console=False`` from a child process that shares stdout with a
    parent running a tqdm progress bar — otherwise the child's log writes
    will land mid-bar and scramble the display. The shared log file still
    captures everything; tail it in another terminal to watch the child.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    constants.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = constants.LOGS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    if console:
        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
    root.addHandler(file_handler)
    return log_path
