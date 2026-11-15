"""Rotation detector core: data pipeline, training, evaluation, and tools.

Everything that is not a model building block or a file-system helper lives
here: transform factories, datasets, norm stats, the training loop,
checkpoint loading, validation, four-view inference, GradCAM, the browse
browser, DFR head retraining, ONNX export, and the subcommand bodies that
``cli.py`` dispatches after parsing.

Keeping the tools in one module lets them share a single checkpoint loader,
stats decoder, and vote construction instead of each carrying a copy.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

import constants
from checkpoint import load_checkpoint_model
from data import (
    RotationPairDataset,
    build_eval_transform,
    build_loader,
    build_train_transform,
    display_preprocess,
)
from inference import (
    RotationVote,
    circular_mean_angle,
    predict_rotations,
    stack_rotation_views,
)
from model import RotationModel
from training import move_batch, run_train
from utils.image_io import open_rgb
from utils.index import index_files
from utils.logsetup import setup_logging
from validation import run_validate

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

N_ROTS = len(constants.ROTATIONS)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Deep Feature Reweighting (DFR)
# ---------------------------------------------------------------------------


def retrain_head(
    model: RotationModel,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> None:
    """Retrain the head only, with the backbone frozen in eval mode.

    Same binary BCE objective as the main training loop.
    """
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.head.train()
        epoch_start = time.monotonic()
        loss_sum = 0.0
        correct = 0
        seen = 0

        pbar = tqdm(
            loader,
            desc=f"dfr epoch {epoch}",
            unit="batch",
            leave=False,
            file=sys.stdout,
        )
        for step, (imgs, labels) in enumerate(pbar):
            imgs, labels = move_batch(imgs, labels, device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
            optimizer.step()

            n = imgs.size(0)
            loss_sum += float(loss.detach()) * n
            preds = (logits.detach() > 0).float()
            correct += int((preds == labels).sum())
            seen += n

            if step % constants.LOG_STEP == 0:
                pbar.set_postfix(
                    acc=f"{correct / max(1, seen):.5f}",
                    loss=f"{loss_sum / max(1, seen):.5f}",
                )

        dt = time.monotonic() - epoch_start
        logger.info(
            "dfr epoch %d done in %.0fs: loss=%.5f acc=%.5f",
            epoch,
            dt,
            loss_sum / max(1, seen),
            correct / max(1, seen),
        )


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------


class GradCAM:
    """GradCAM and guided-GradCAM heatmap computation for one loaded model."""

    def __init__(
        self,
        model: RotationModel,
        device: torch.device,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.model = model
        self.device = device
        self.preprocess = build_eval_transform(mean=mean, std=std)

    def compute(self, tensor: torch.Tensor) -> tuple[list[np.ndarray], list[float]]:
        """Run one forward+backward and return per-sample GradCAM heatmaps
        (each in [0, 1]) plus per-sample upright logits."""
        target_layer: nn.Module = self.model.backbone.conv_head  # type: ignore[assignment]

        saved_features: list[torch.Tensor] = []
        saved_grads: list[torch.Tensor] = []

        def fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
            saved_features.append(out)

        def bwd(_m: nn.Module, _gin: object, gout: tuple[torch.Tensor, ...]) -> None:
            saved_grads.append(gout[0])

        fh = target_layer.register_forward_hook(fwd)
        bh = target_layer.register_full_backward_hook(bwd)  # type: ignore[arg-type]

        try:
            output = self.model(tensor)  # (B,) or (B, 1)
            self.model.zero_grad()
            output.sum().backward()  # per-sample grads are independent
        finally:
            fh.remove()
            bh.remove()

        feat = saved_features[0]  # (B, C, H, W)
        grad = saved_grads[0]  # (B, C, H, W)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = nn.functional.relu((weights * feat).sum(dim=1))  # (B, H, W)
        cam_max = cam.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        cam_np = (cam / cam_max).cpu().detach().float().numpy()

        logits = output.detach().cpu().float().reshape(-1).tolist()
        return list(cam_np), logits

    def guided(self, tensor: torch.Tensor, cams: list[np.ndarray]) -> list[np.ndarray]:
        """Guided backprop gated by the supplied GradCAM heatmaps.

        Produces sharp, class-discriminative, edge-like maps at full input
        resolution. SiLU stand-in for ReLU in the guided-BP gate: only pass
        gradient where the SiLU output was positive AND the incoming gradient
        is positive. Runs a single batched forward+backward over the batch.
        """
        silu_outputs: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_fwd(idx: int):
            def fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
                silu_outputs[idx] = out

            return fwd

        def make_bwd(idx: int):
            def bwd(
                _m: nn.Module,
                _gin: object,
                gout: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, ...]:
                out = silu_outputs[idx]
                new_grad = gout[0].clone()
                new_grad[out <= 0] = 0
                new_grad[new_grad < 0] = 0
                return (new_grad,)

            return bwd

        silu_modules = [m for m in self.model.modules() if isinstance(m, nn.SiLU)]
        original_inplace = [sm.inplace for sm in silu_modules]
        for sm in silu_modules:
            sm.inplace = False  # backward hooks reject view+inplace combo
        for i, sm in enumerate(silu_modules):
            handles.append(sm.register_forward_hook(make_fwd(i)))
            handles.append(sm.register_full_backward_hook(make_bwd(i)))  # type: ignore[arg-type]

        try:
            x = tensor.detach().clone().requires_grad_(True)
            output = self.model(x)
            self.model.zero_grad()
            output.sum().backward()
        finally:
            for h in handles:
                h.remove()
            for sm, was_inplace in zip(silu_modules, original_inplace, strict=True):
                sm.inplace = was_inplace

        grad = x.grad.abs().sum(dim=1)  # type: ignore[union-attr]  # (B, H, W)
        grad_np = grad.cpu().detach().float().numpy()

        guided_out: list[np.ndarray] = []
        for i, cam in enumerate(cams):
            h_in, w_in = grad_np[i].shape
            cam_up = (
                np.array(
                    Image.fromarray((cam * 255).astype(np.uint8)).resize((w_in, h_in), Image.Resampling.BICUBIC),
                ).astype(np.float32)
                / 255.0
            )

            guided = grad_np[i] * cam_up
            p99 = float(np.percentile(guided, 99))
            if p99 > 0:
                guided = np.clip(guided / p99, 0.0, 1.0)
            guided_out.append(guided)
        return guided_out

    def rotations(
        self,
        pil_img: Image.Image,
    ) -> tuple[list[Image.Image], list[float], list[np.ndarray], torch.Tensor]:
        # Display images: resize + center-crop, then PIL-rotate each view.
        crop = display_preprocess(pil_img)
        displays = [crop.rotate(a) if a != 0 else crop for a in constants.ROTATIONS]

        # Model input: transform once (resize/crop/to_tensor/normalize), then
        # rotate the tensor - same geometry as train/infer (RotationPairDataset,
        # predict_rotations), not PIL-rotate-then-transform.
        base = self.preprocess(pil_img)
        tensors = stack_rotation_views(base, self.device)

        cams, logits = self.compute(tensors)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0).tolist()
        return displays, probs, cams, tensors


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def overlay_heatmap(
    pil_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.65,
    colormap: str = "jet",
) -> Image.Image:
    cam = np.sqrt(cam)  # spread activation - avoids tiny concentrated hot spots
    heatmap = mpl.colormaps[colormap](cam)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_img.size, Image.Resampling.BICUBIC)
    blended = np.array(pil_img, dtype=np.float32) * (1 - alpha) + np.array(heatmap_pil, dtype=np.float32) * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def render_rotation_grid(
    displays: list[Image.Image],
    probs: list[float],
    heatmaps: list[np.ndarray],
    title: str,
    out_path: Path,
    colormap: str = "jet",
) -> None:
    """Render a 2x2 rotation grid with `heatmaps[i]` overlaid on `displays[i]`."""
    pred = int(np.argmax(probs))

    fig = Figure(figsize=(10, 10))
    axes = fig.subplots(2, 2)

    for idx, (angle, heat, prob, disp) in enumerate(
        zip(constants.ROTATIONS, heatmaps, probs, displays, strict=True)
    ):
        img_overlay = overlay_heatmap(disp, heat, colormap=colormap)
        ax = axes[idx // 2, idx % 2]
        marker = " *" if idx == pred else ""
        ax.imshow(img_overlay)
        ax.set_title(f"{angle}°{marker}  softmax={prob:.0%}", fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Browse (tkinter)
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


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def export_onnx(
    model: RotationModel,
    out_path: Path,
    device: torch.device,
    opset: int,
    dynamo: bool,
    verify: bool,
) -> None:
    """Trace ``model`` to ONNX with a dynamic batch axis."""
    dummy = torch.randn(1, 3, constants.INPUT_SIZE, constants.INPUT_SIZE, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs: dict = {
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": {
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
        "dynamo": dynamo,
    }

    logger.info(
        "exporting ONNX -> %s (opset=%d, dynamo=%s, device=%s)",
        out_path,
        opset,
        dynamo,
        device,
    )

    # Legacy exporter writes ``f`` directly and returns None. The dynamo
    # exporter returns an ONNXProgram that must be saved when f is None.
    try:
        result = torch.onnx.export(
            model,
            (dummy,),
            str(out_path) if not dynamo else None,
            **export_kwargs,
        )
    except ModuleNotFoundError as exc:
        if dynamo and "onnxscript" in str(exc):
            raise RuntimeError(
                "dynamo export needs the onnxscript package "
                "(pip install onnxscript); or omit --dynamo for the legacy exporter"
            ) from exc
        raise

    if result is not None:
        result.save(str(out_path))

    if not out_path.is_file():
        raise RuntimeError(f"ONNX export produced no file at {out_path}")

    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("verify requires onnxruntime") from exc

        with torch.no_grad():
            torch_out = model(dummy).detach().float().cpu().numpy()

        session = ort.InferenceSession(
            str(out_path),
            providers=["CPUExecutionProvider"],
        )
        ort_out = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: dummy.detach().float().cpu().numpy()},
        )[0]

        max_abs = float(abs(torch_out.reshape(-1) - ort_out.reshape(-1)).max())
        logger.info("verify max|torch-ort|=%.6g on dummy batch", max_abs)
        if max_abs > 1e-4:
            logger.warning("torch/ort diverge more than 1e-4; check opset / dynamo flags")


# ---------------------------------------------------------------------------
# CLI subcommand bodies
# ---------------------------------------------------------------------------


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


def run_gradcam(args: argparse.Namespace) -> None:
    images = [p for p in args.images if p.is_file()]

    device = torch.device(args.device)
    _, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    gradcam = GradCAM(model, device, mean, std)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        pil = open_rgb(img_path)
        displays, probs, cams, tensors = gradcam.rotations(pil)
        angle = circular_mean_angle(probs)

        render_rotation_grid(
            displays,
            probs,
            cams,
            f"{img_path.name} - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_gradcam.png",
            colormap="jet",
        )
        print(f"{img_path.name} - predicted angle {angle:.1f}°")

        guided_cams = gradcam.guided(tensors, cams)
        render_rotation_grid(
            displays,
            probs,
            guided_cams,
            f"{img_path.name} - guided GradCAM - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_guidedcam.png",
            colormap="inferno",
        )


def run_dfr(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    _, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
        strict_stats=True,
    )
    model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

    # Freeze everything outside the head; the backbone also stays in eval mode
    # so its BatchNorm running stats remain at the full model's values.
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    model.backbone.eval()

    if not args.no_reinit_head:
        for module in model.head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore[operator]
        logger.info("reinitialised head for a fresh DFR fit")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable params: %d | frozen params: %d", trainable, total - trainable)

    # Indexed under its own cache name so it never collides with the main
    # train/val file indexes.
    files = index_files(
        name="dfr",
        roots=list(args.data_root),
        cache_dir=constants.INDEX_CACHE_DIR,
        invalidate_cache=args.invalidate_cache,
    )
    dataset = RotationPairDataset(
        files=files,
        transform=build_train_transform(mean=mean, std=std),
    )
    logger.info("DFR reweighting set: %d files -> %d pairs", len(files), len(dataset))
    loader = build_loader(
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
    )

    retrain_head(
        model=model,
        loader=loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    state = model.state_dict()
    payload = {
        "raw": state,
        "ema": state,
        "stats": {"mean": list(mean), "std": list(std)},
        "epoch": "dfr",
        "pooling": model.pooling,
        "dfr_source": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    logger.info("saved DFR checkpoint %s", args.output)


def run_export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    payload, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    meta = {
        "checkpoint": str(args.checkpoint),
        "epoch": payload.get("epoch"),
        "weights": args.weights,
        "pooling": payload.get("pooling", "avg"),
        "pretrained": payload.get("pretrained"),
        "mean": list(mean),
        "std": list(std),
        "input_size": constants.INPUT_SIZE,
        "resize_size": constants.RESIZE_SIZE,
        "rotations": list(constants.ROTATIONS),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
    }
    out_path = args.out or args.checkpoint.with_suffix(".onnx")

    export_onnx(
        model=model,
        out_path=out_path,
        device=device,
        opset=args.opset,
        dynamo=args.dynamo,
        verify=args.verify,
    )

    meta_path = out_path.with_suffix(".json")
    meta_payload = {
        **meta,
        "onnx": str(out_path),
        "opset": args.opset,
        "dynamo": args.dynamo,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n")
    logger.info("wrote metadata %s", meta_path)
    logger.info("done: %s (%.1f MiB)", out_path, out_path.stat().st_size / (1024 * 1024))


HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "train": run_train,
    "finetune": run_train,
    "validate": run_validate,
    "browse": run_browse,
    "gradcam": run_gradcam,
    "dfr": run_dfr,
    "export": run_export,
}


def run(args: argparse.Namespace) -> None:
    """Configure logging and dispatch one parsed subcommand."""
    setup_logging(constants.LOGS_DIR)
    HANDLERS[args.command](args)
