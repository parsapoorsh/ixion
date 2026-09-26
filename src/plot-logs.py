#!/usr/bin/env python3
"""
Parse training log lines and plot metrics (loss, acc, lr, grad_norm).

Usage:
  python plot-logs.py train.log
  python plot-logs.py train.log --epoch 0
  python plot-logs.py train.log -e 0 --save metrics.png
  python plot-logs.py train.log --watch
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Matches lines like:
# 2026-09-25 02:29:09 INFO __main__: epoch 0 step 450/11550930 loss=1.52132 acc=0.24612 lr=7.01e-08 grad_norm=62.26751
LINE_RE = re.compile(
    r"epoch\s+(\d+)\s+step\s+(\d+)/(\d+)\s+"
    r"loss=([0-9.eE+-]+)\s+"
    r"(?:group4_)?acc=([0-9.eE+-]+)\s+"
    r"lr=([0-9.eE+-]+)\s+"
    r"grad_norm=([0-9.eE+-]+)"
)

METRICS = ("loss", "acc", "lr", "grad_norm")

@dataclass(frozen=True)
class Record:
    epoch: int
    step: int
    total_steps: int
    loss: float
    acc: float
    lr: float
    grad_norm: float


def parse_log(
    path: Path,
    epoch: int | None = None,
    allow_empty: bool = False,
) -> list[Record]:
    """Return list of Records, optionally filtered to a single epoch.

    With ``allow_empty`` an unparseable (or not-yet-populated) log returns an
    empty list instead of exiting, which lets a watch loop keep polling.
    """
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
                    acc=float(acc),
                    lr=float(lr),
                    grad_norm=float(grad),
                )
            )
    if not records:
        if allow_empty:
            return records
        if epoch is not None:
            # Re-scan for available epochs so the error message is helpful
            available: set[int] = set()
            with path.open() as f:
                for line in f:
                    m = LINE_RE.search(line)
                    if m:
                        available.add(int(m.group(1)))
            raise SystemExit(f"No data for epoch {epoch}. Available epochs: {sorted(available)}")
        raise SystemExit(f"No matching log lines found in {path}")
    return records


def smooth_trend(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean for a smooth trend overlay.

    A linear fit would ignore the curved shape of loss/acc/grad_norm; a
    rolling mean follows the actual trend. The window is forced odd so the
    mean stays centered on each sample, and edges are replicated so the
    trend spans the full range.
    """
    n = y.size
    if n < 3:
        return y.astype(float).copy()
    window = min(window, n)
    if window < 3:
        window = min(3, n)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y.astype(float).copy()
    half = window // 2
    padded = np.pad(y, (half, half), mode="edge")
    cs = np.cumsum(np.insert(padded, 0, 0.0))
    return (cs[window:] - cs[:-window]) / window


def make_focus_scale(ceiling, spread_top):
    """Build a (forward, inverse) pair that zooms the y axis into one end."""
    p = 2

    def forward(values):
        step = np.clip(np.asarray(values, dtype=float) / ceiling, 0.0, 1.0)
        if spread_top:
            return 1.0 - np.power(1.0 - step, 1.0 / p)
        return np.power(step, 1.0 / p)

    def inverse(steps):
        step = np.clip(np.asarray(steps, dtype=float), 0.0, 1.0)
        if spread_top:
            step = 1.0 - np.power(1.0 - step, p)
        else:
            step = np.power(step, p)
        return step * ceiling

    return forward, inverse


def log_forward(v):
    """Map v > 0 to log2(v), stretching values near 0."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(np.maximum(v, 1e-6))


def log_inverse(u):
    """Inverse of log_forward: u -> 2**u."""
    return np.power(2.0, np.asarray(u, dtype=float))


def draw_axes(
    fig: plt.Figure,
    axes: np.ndarray,
    records: list[Record],
    epoch: int | None,
    trend_window: int = 0,
) -> None:
    """Draw all panels into ``axes`` so a watch loop can redraw in place."""
    x = np.arange(len(records))  # always continuous

    axes = axes.ravel()

    title = f"Training metrics - epoch {epoch}" if epoch is not None else "Training metrics - all epochs"
    fig.suptitle(title, fontsize=14)

    auto_window = max(15, (len(records) // 40) | 1)
    window = trend_window if trend_window > 0 else auto_window

    for ax, metric in zip(axes, METRICS, strict=True):
        y = np.array([getattr(r, metric) for r in records])
        trend = smooth_trend(y, window)

        ax.plot(
            x,
            y,
            color="C0",
            linewidth=1.0,
            label="raw",
        )
        ax.plot(
            x,
            trend,
            color="C1",
            linewidth=2.0,
            label="trend",
        )
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if metric == "lr":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        else:
            # Zoom into the crowded end: acc piles up near its max, loss near
            # zero, and a linear axis would waste most of its height on the
            # empty end (see FOCUS). The axis spans exactly this run's own
            # range - the peak sits on the top line rather than in a margin -
            # and the transform is built around that range, because auto-limits
            # collapse on a function scale.
            forward, inverse = make_focus_scale(
                float(y.max()),
                metric == "acc",
            )
            ax.set_yscale(
                "function",
                functions=(forward, inverse),
            )
            ax.set_ylim(
                float(y.min()),
                float(y.max()),
            )

    axes[0].legend(loc="upper right", fontsize=8)

    # Mark epoch boundaries when showing all epochs
    if epoch is None and len(records) > 1:
        epochs = np.array([r.epoch for r in records])
        changes = np.where(np.diff(epochs) != 0)[0] + 1
        for ax in axes:
            for c in changes:
                ax.axvline(c, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.tight_layout()


def plot(
    records: list[Record],
    epoch: int | None,
    save_path: Path | None = None,
    show: bool = True,
    trend_window: int = 0,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    draw_axes(fig, axes, records, epoch, trend_window)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def watch_log(
    path: Path,
    epoch: int | None,
    save_path: Path | None,
    show: bool,
    trend_window: int,
    interval: float,
) -> None:
    """Re-read ``path`` on change and redraw; exits on KeyboardInterrupt.

    Change is detected by mtime (nanosecond) and size, so the plot follows a
    log that is still being written by a running training job.
    """
    if show:
        plt.ion()
    fig: plt.Figure | None = None
    axes: np.ndarray | None = None
    last_mtime: int | None = None
    last_size: int | None = None
    while True:
        try:
            stat = path.stat()
        except FileNotFoundError:
            time.sleep(interval)
            continue
        changed = (
            last_mtime is None
            or stat.st_mtime_ns != last_mtime
            or stat.st_size != last_size
        )
        if changed:
            last_mtime = stat.st_mtime_ns
            last_size = stat.st_size
            records = parse_log(path, epoch=epoch, allow_empty=True)
            if not records:
                # Nothing parseable yet (or after the log was truncated); keep
                # waiting for the first batch of step lines.
                if show:
                    plt.pause(interval)
                else:
                    time.sleep(interval)
                continue
            if fig is None:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            assert axes is not None
            for ax in axes.ravel():
                ax.clear()
            draw_axes(fig, axes, records, epoch, trend_window)
            fig.canvas.draw_idle()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved figure to {save_path}")
        if show:
            plt.pause(interval)
        else:
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics from log file.")
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
    parser.add_argument(
        "--trend-window",
        type=int,
        default=0,
        help="rolling-mean window for the trend overlay (0 = auto, ~2.5%% of points)",
    )
    parser.add_argument(
        "-w",
        "--watch",
        action="store_true",
        help="keep re-reading the log and redrawing the plot as it grows",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=2.0,
        help="seconds between re-reads in --watch mode",
    )
    args = parser.parse_args()

    # Watch mode tolerates a log that doesn't exist yet (e.g. training starts
    # after the watcher); a one-shot run still requires an existing file.
    if not args.watch and not args.logfile.is_file():
        raise SystemExit(f"File not found: {args.logfile}")

    if args.watch:
        watch_log(
            args.logfile,
            epoch=args.epoch,
            save_path=args.save,
            show=not args.no_show,
            trend_window=args.trend_window,
            interval=args.watch_interval,
        )
        return

    records = parse_log(args.logfile, epoch=args.epoch)

    plot(
        records,
        epoch=args.epoch,
        save_path=args.save,
        show=not args.no_show,
        trend_window=args.trend_window,
    )


if __name__ == "__main__":
    main()
