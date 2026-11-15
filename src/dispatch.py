"""Subcommand dispatch, imported by cli.py only after parsing."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import browse
import constants
import dfr
import export
import gradcam
import training
import validation
from utils.logsetup import setup_logging

if TYPE_CHECKING:
    import argparse

HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "train": training.run_train,
    "finetune": training.run_train,
    "validate": validation.run_validate,
    "browse": browse.run_browse,
    "gradcam": gradcam.run_gradcam,
    "dfr": dfr.run_dfr,
    "export": export.run_export,
}


def run(args: argparse.Namespace) -> None:
    """Configure logging and dispatch one parsed subcommand."""
    setup_logging(constants.LOGS_DIR)
    HANDLERS[args.command](args)
