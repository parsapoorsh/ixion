"""Tkinter browser with live rotation predictions."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

import constants
from checkpoint import load_checkpoint_model
from data import build_eval_transform
from inference import RotationVote, predict_rotations
from model import RotationModel
from utils.image_io import open_rgb

if TYPE_CHECKING:
    import argparse

# ---------------------------------------------------------------------------

IMAGE_EXTS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".jxl",
    }
)
PREFETCH_SIZE = 10
DONE = object()  # sentinel pushed by worker when exhausted


class InferResult(TypedDict):
    path: Path
    corrected: Image.Image
    vote: RotationVote
    abstained: bool
    error: str | None


def prefetch_worker(
    paths: list[Path],
    model: RotationModel,
    device: torch.device,
    preprocess: T.Compose,
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
                preprocess=preprocess,
            )
            abstained = abstain_margin is not None and vote["margin"] < abstain_margin
            # Low margin: keep original (no auto-rotate).
            if abstained:
                corrected = original
            else:
                corrected = original.rotate(vote["best_deg"], expand=True)
            result: InferResult = {
                "path": path,
                "corrected": corrected,
                "vote": vote,
                "abstained": abstained,
                "error": None,
            }
        except Exception as exc:
            blank = Image.new("RGB", (constants.INPUT_SIZE, constants.INPUT_SIZE), color=(30, 30, 30))
            result = {
                "path": path,
                "corrected": blank,
                "vote": {
                    "probs": dict.fromkeys(constants.ROTATIONS, 0.25),
                    "best_deg": 0,
                    "margin": 0.0,
                },
                "abstained": False,
                "error": str(exc),
            }

        out.put(result)
        bar.update(1)

    out.put(DONE)


def show_result(result: InferResult, index: int, total: int) -> None:
    """Open a blocking tkinter window for *result*; returns when it is closed."""
    import tkinter as tk

    from PIL import ImageTk

    prob_str = "  ".join(f"{deg}°: {prob:.3f}" for deg, prob in sorted(result["vote"]["probs"].items()))
    if result["abstained"]:
        decision = f"ABSTAIN (margin={result['vote']['margin']:.3f})"
    else:
        decision = f"correction: {result['vote']['best_deg']}° (margin={result['vote']['margin']:.3f})"
    title = f"[{index}/{total}]  {result['path'].name}  ->  {decision}  |  {prob_str}"

    root = tk.Tk()
    root.title(title)
    root.resizable(True, True)
    root.bind("<q>", lambda _e: root.destroy())

    img = result["corrected"].copy()
    img.thumbnail((1280, 900), Image.Resampling.LANCZOS)

    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=photo, bg="black")
    img_label.pack(fill=tk.BOTH, expand=True)

    if result["error"]:
        err_label = tk.Label(
            root,
            text=f"⚠  {result['error']}",
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


def run_browse(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    print(f"Loading {args.checkpoint}  ({args.weights} weights)...")
    _, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    preprocess = build_eval_transform(mean=mean, std=std)
    print("Model ready.\n")

    root_dir = Path(args.dir)
    print(f"Scanning {root_dir}...")
    paths = sorted(p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
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
            preprocess,
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
        if item is DONE:
            break

        result = cast(InferResult, item)
        processed += 1

        best_norm = max(result["vote"]["probs"].values())
        if not result["error"] and best_norm >= args.conf_threshold:
            skipped += 1
            tqdm.write(
                f"[{processed}/{len(paths)}]  {result['path'].name}"
                f"  skipped (conf={best_norm:.3f} >= {args.conf_threshold})"
            )
            continue

        shown += 1
        if result["abstained"]:
            abstained += 1

        if result["error"]:
            tqdm.write(f"[{processed}/{len(paths)}]  {result['path']}  ERROR: {result['error']}")
        else:
            status = "abstain" if result["abstained"] else f"best={result['vote']['best_deg']}°"
            norm_str = ", ".join(f"{d}°:{p:.3f}" for d, p in sorted(result["vote"]["probs"].items()))
            tqdm.write(
                f"[{processed}/{len(paths)}]  {result['path'].name}"
                f"  {status}"
                f"  margin={result['vote']['margin']:.3f}"
                f"  conf={best_norm:.3f}"
                f"  norm={{{norm_str}}}"
            )

        show_result(result, shown, len(paths))

    bar.close()
    worker.join()
    print(
        f"\nDone - displayed {shown} / {len(paths)} image(s)"
        f"  ({skipped} skipped conf >= {args.conf_threshold},"
        f" {abstained} abstained on margin)."
    )
