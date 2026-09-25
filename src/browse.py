#!/usr/bin/env python3
"""Interactive rotation-correction browser.

Scans a directory, runs predict_rotations on every image in a background
prefetch thread (queue depth: PREFETCH_SIZE), and displays each result in
a blocking tkinter window - close the window to advance to the next image.
A tqdm bar in the terminal tracks inference progress independently of display.

Usage:
    venv/bin/python src/browse.py \\
        --dir /path/to/photos \\
        --checkpoint checkpoints/epoch_030.pt \\
        --device cuda
"""

from __future__ import annotations

import argparse
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from constants import INDEX_CACHE_DIR
from infer import RotationVote, predict_rotations
from models import RotationModel
from stats import NormStats, load_stats_cache
from utils.image_io import open_rgb


IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff", ".tif",
    ".jxl",
})
PREFETCH_SIZE = 10
_DONE = object()  # sentinel pushed by worker when exhausted


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InferResult:
    path: Path
    original: Image.Image
    corrected: Image.Image
    vote: RotationVote
    best_deg: int
    abstained: bool = False
    error: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint: Path,
    device: torch.device,
    weights: str,
) -> tuple[RotationModel, NormStats]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    state_key = {"ema": "ema", "raw": "model"}[weights]
    if state_key not in payload:
        raise KeyError(
            f"checkpoint missing '{state_key}' key; available={list(payload)}"
        )

    raw_stats = payload.get("stats")
    if raw_stats:
        stats = NormStats(
            mean=tuple(raw_stats["mean"]),  # type: ignore[arg-type]
            std=tuple(raw_stats["std"]),    # type: ignore[arg-type]
        )
    else:
        stats = load_stats_cache(INDEX_CACHE_DIR / "stats_train.json")

    model = RotationModel(
        pretrained=False,
        pooling=payload.get("pooling", "attention"),
    ).to(device)
    model.load_state_dict(payload[state_key])
    model.eval()
    return model, stats


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

def scan_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# ---------------------------------------------------------------------------
# Background prefetch worker
# ---------------------------------------------------------------------------

def prefetch_worker(
    paths: list[Path],
    model: RotationModel,
    device: torch.device,
    stats: NormStats,
    out: queue.Queue,
    bar: tqdm,
    abstain_margin: float | None,
) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)

    for path in paths:
        try:
            original = open_rgb(path)
            vote = predict_rotations(
                model=model,
                img=original,
                device=device,
                mean=stats.mean,
                std=stats.std,
            )
            best_deg = vote.best_deg
            abstained = (
                abstain_margin is not None
                and vote.should_abstain(abstain_margin)
            )
            # Low margin: keep original (no auto-rotate).
            if abstained:
                corrected = original
            else:
                corrected = original.rotate(best_deg, expand=True)
            result: InferResult = InferResult(
                path=path,
                original=original,
                corrected=corrected,
                vote=vote,
                best_deg=best_deg,
                abstained=abstained,
            )
        except Exception as exc:
            blank = Image.new("RGB", (224, 224), color=(30, 30, 30))
            result = InferResult(
                path=path,
                original=blank,
                corrected=blank,
                vote=RotationVote(
                    probs={0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0},
                    normalized_probs={0: 0.25, 90: 0.25, 180: 0.25, 270: 0.25},
                    best_deg=0,
                    margin=0.0,
                    angle=0.0,
                ),
                best_deg=0,
                error=str(exc),
            )

        out.put(result)
        bar.update(1)

    out.put(_DONE)


# ---------------------------------------------------------------------------
# Display (blocking tkinter window)
# ---------------------------------------------------------------------------

def show_result(result: InferResult, index: int, total: int) -> None:
    """Open a blocking tkinter window for *result*; returns when it is closed."""
    import tkinter as tk
    from PIL import ImageTk

    prob_str = "  ".join(
        f"{deg}°: {prob:.3f}" for deg, prob in sorted(result.vote.normalized_probs.items())
    )
    if result.abstained:
        decision = f"ABSTAIN (margin={result.vote.margin:.3f})"
    else:
        decision = f"correction: {result.best_deg}° (margin={result.vote.margin:.3f})"
    title = (
        f"[{index}/{total}]  {result.path.name}"
        f"  ->  {decision}"
        f"  |  {prob_str}"
    )

    root = tk.Tk()
    root.title(title)
    root.resizable(True, True)
    root.bind("<q>", lambda _e: root.destroy())

    img = result.corrected.copy()
    img.thumbnail((1280, 900), Image.LANCZOS)

    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=photo, bg="black")
    img_label.pack(fill=tk.BOTH, expand=True)

    if result.error:
        err_label = tk.Label(
            root,
            text=f"⚠  {result.error}",
            fg="#ff6060",
            bg="black",
            font=("Helvetica", 11),
        )
        err_label.pack()

    info = tk.Label(
        root,
        text=prob_str,
        fg="#cccccc",
        bg="#1a1a1a",
        font=("Courier", 10),
        pady=4,
    )
    info.pack(fill=tk.X)

    root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    print(f"Loading {args.checkpoint}  ({args.weights} weights)...")
    model, stats = load_model(args.checkpoint, device, args.weights)
    print("Model ready.\n")

    root_dir = Path(args.dir)
    print(f"Scanning {root_dir}...")
    paths = scan_images(root_dir)
    if not paths:
        print("No images found.")
        return
    print(f"Found {len(paths)} image(s).\n")

    result_queue: queue.Queue[InferResult | object] = queue.Queue(maxsize=PREFETCH_SIZE)
    bar = tqdm(total=len(paths), desc="Inferring", unit="img", dynamic_ncols=True)

    worker = threading.Thread(
        target=prefetch_worker,
        args=(
            paths,
            model,
            device,
            stats,
            result_queue,
            bar,
            args.abstain_margin,
        ),
        daemon=True,
    )
    worker.start()

    shown = 0
    skipped = 0
    abstained = 0
    processed = 0
    while True:
        item = result_queue.get()
        if item is _DONE:
            break

        assert isinstance(item, InferResult)
        processed += 1

        best_norm = max(item.vote.normalized_probs.values())
        if not item.error and best_norm >= args.conf_threshold:
            skipped += 1
            tqdm.write(
                f"[{processed}/{len(paths)}]  {item.path.name}"
                f"  skipped (conf={best_norm:.3f} >= {args.conf_threshold})"
            )
            continue

        shown += 1
        if item.abstained:
            abstained += 1

        if item.error:
            tqdm.write(f"[{processed}/{len(paths)}]  {item.path}  ERROR: {item.error}")
        else:
            status = "abstain" if item.abstained else f"best={item.best_deg}°"
            tqdm.write(
                f"[{processed}/{len(paths)}]  {item.path.name}"
                f"  {status}"
                f"  margin={item.vote.margin:.3f}"
                f"  conf={best_norm:.3f}"
                f"  norm={{{', '.join(f'{d}°:{p:.3f}' for d, p in sorted(item.vote.normalized_probs.items()))}}}"
            )

        show_result(item, shown, len(paths))

    bar.close()
    worker.join()
    print(
        f"\nDone - displayed {shown} / {len(paths)} image(s)"
        f"  ({skipped} skipped conf >= {args.conf_threshold},"
        f" {abstained} abstained on margin)."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Browse a directory with live rotation predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="root directory to scan recursively for images",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint .pt file to load",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="torch device string",
    )
    p.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="which weights from the checkpoint to use",
    )
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.95,
        dest="conf_threshold",
        help="skip display when max four-view softmax probability exceeds this",
    )
    p.add_argument(
        "--abstain-margin",
        type=float,
        default=None,
        dest="abstain_margin",
        help=(
            "if set, leave image unrotated when the top1-top2 softmax "
            "margin is below this (e.g. 0.15); default: always apply argmax"
        ),
    )
    main(p.parse_args())
