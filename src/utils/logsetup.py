"""Tiny logging setup shared by every CLI entry point.

Importing this module is enough to configure the root logger consistently.
Kept in its own file so changing the format/level happens once.
"""

import logging
import sys

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so an active progress bar
    isn't clobbered by interleaved log lines."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO) -> None:
    handler = TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
