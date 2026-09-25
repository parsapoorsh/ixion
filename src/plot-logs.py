#!/usr/bin/env python3
"""
Parse training log lines and plot metrics (loss, group4_acc, lr, grad_norm).

Usage:
  python plot-logs.py train.log
  python plot-logs.py train.log --epoch 0
  python plot-logs.py train.log -e 0 --save metrics.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Matches lines like:
# 2026-09-25 02:29:09 INFO __main__: epoch 0 step 450/11550930 loss=1.52132 group4_acc=0.24612 lr=7.01e-08 grad_norm=62.26751
LINE_RE = re.compile(
    r"epoch\s+(\d+)\s+step\s+(\d+)/(\d+)\s+"
    r"loss=([0-9.eE+-]+)\s+"
    r"group4_acc=([0-9.eE+-]+)\s+"
    r"lr=([0-9.eE+-]+)\s+"
    r"grad_norm=([0-9.eE+-]+)"
)

METRICS = ("loss", "group4_acc", "lr", "grad_norm")


@dataclass(frozen=True)
class Record:
    epoch: int
    step: int
    total_steps: int
    loss: float
    group4_acc: float
    lr: float
    grad_norm: float


def parse_log(path: Path, epoch: Optional[int] = None) -> list[Record]:
    """Return list of Records, optionally filtered to a single epoch."""
    records: list[Record] = []
    with path.open() as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            ep, step, total, loss, acc, lr, grad = m.groups()
            ep_i = int(ep)
            if epoch is not None and ep_i != epoch:
                continue
            records.append(
                Record(
                    epoch=ep_i,
                    step=int(step),
                    total_steps=int(total),
                    loss=float(loss),
                    group4_acc=float(acc),
                    lr=float(lr),
                    grad_norm=float(grad),
                )
            )
    if not records:
        if epoch is not None:
            # Re-scan for available epochs so the error message is helpful
            available: set[int] = set()
            with path.open() as f:
                for line in f:
                    m = LINE_RE.search(line)
                    if m:
                        available.add(int(m.group(1)))
            raise SystemExit(
                f"No data for epoch {epoch}. Available epochs: {sorted(available)}"
            )
        raise SystemExit(f"No matching log lines found in {path}")
    return records


def plot(
    records: list[Record],
    epoch: Optional[int],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    x = np.arange(len(records))  # always continuous

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    title = (
        f"Training metrics - epoch {epoch}"
        if epoch is not None
        else "Training metrics - all epochs"
    )
    fig.suptitle(title, fontsize=14)

    for ax, metric in zip(axes, METRICS):
        y = np.array([getattr(r, metric) for r in records])
        ax.plot(x, y, linewidth=1.2)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if metric == "lr":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Mark epoch boundaries when showing all epochs
    if epoch is None and len(records) > 1:
        epochs = np.array([r.epoch for r in records])
        changes = np.where(np.diff(epochs) != 0)[0] + 1
        for ax in axes:
            for c in changes:
                ax.axvline(c, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training metrics from log file."
    )
    parser.add_argument(
        "logfile",
        type=Path,
        help="Path to the training log file",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=None,
        help="Epoch to plot (default: all epochs, continuous)",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=Path,
        default=None,
        help="Save figure to this path instead of (or in addition to) showing",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful with --save)",
    )
    args = parser.parse_args()

    if not args.logfile.is_file():
        raise SystemExit(f"File not found: {args.logfile}")

    records = parse_log(args.logfile, epoch=args.epoch)

    plot(
        records,
        epoch=args.epoch,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
