#!/usr/bin/env python3
"""Command-line entry point for the rotation detector.

Subcommands:
    train     full training run
    finetune  short ImageNet fine-tune preset
    validate  evaluate a checkpoint on the val split
    browse    tkinter browser over a directory of photos
    gradcam   GradCAM visualisation for a few images
    dfr       retrain the head on frozen features
    export    export a checkpoint to ONNX
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
from pathlib import Path

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import constants
import core
from utils.image_io import open_rgb
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)

# Short-schedule preset applied by the finetune subcommand; explicit flags
# still win. Lower backbone LR and weight decay keep the pretrained features
# close to where they started while the new head catches up.
FINETUNE_DEFAULTS = {
    "pretrain": True,
    "epochs": 5,
    "lr": 1e-3,
    "backbone_lr_mult": 0.1,
    "warmup_epochs": 0.5,
    "weight_decay": 0.05,
}


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=("root directory that contains the ImageNet 2017 and COCO 2017 layout (see config.build_datasets)"),
    )
    p.add_argument(
        "--device",
        type=str,
        required=True,
        help="torch device string (e.g. 'cuda', 'cuda:0', 'cpu')",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="images per step",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="DataLoader worker processes for image decoding",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        required=True,
        help="batches each worker keeps queued ahead of the consumer",
    )


def add_checkpoint_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to an epoch_NNN.pt checkpoint",
    )
    p.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="which weights from the checkpoint to use: EMA shadow (production) or raw",
    )


def add_training_args(p: argparse.ArgumentParser) -> None:
    add_data_args(p)
    p.add_argument(
        "--seed",
        type=int,
        required=True,
        help="torch RNG seed for the training loop",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="total number of training epochs",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="peak learning rate (after warmup, before cosine decay)",
    )
    p.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=1.0,
        help="backbone LR as a fraction of --lr (head always uses the full --lr)",
    )
    p.add_argument(
        "--warmup-epochs",
        type=float,
        default=5,
        help="linear LR warmup length in epochs (fractions allowed)",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="head dropout rate",
    )
    p.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        help="backbone stochastic depth rate",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=20.0,
        help="max total gradient norm before clipping (safety net for spikes; typical norm is ~6-16)",
    )
    p.add_argument(
        "--ema-horizon-epochs",
        type=float,
        default=0.4,
        help="EMA averaging horizon in epochs (sets the per-step EMA decay)",
    )
    p.add_argument(
        "--erasing-p",
        type=float,
        default=0.25,
        help="random erasing probability (0 disables)",
    )
    p.add_argument(
        "--crop-policy",
        choices=["random", "center", "none"],
        default="random",
        help="train spatial preprocessing: resize+random crop, resize+center crop, or direct resize",
    )
    p.add_argument(
        "--pooling",
        choices=["attention", "avg"],
        default="attention",
        help="how to aggregate backbone features before the head",
    )
    p.add_argument(
        "--sam",
        action="store_true",
        help="use Sharpness-Aware Minimization over AdamW (2x passes/step, flatter minima)",
    )
    p.add_argument(
        "--sam-rho",
        type=float,
        default=0.05,
        help="SAM neighbourhood radius for the worst-case ascent step",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached file-index and per-dataset stats JSONs before running",
    )
    p.add_argument(
        "--pretrain",
        action="store_true",
        help="initialize the backbone from timm's ImageNet weights and use ImageNet mean/std",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile even when build dependencies are available",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to an epoch_NNN.pt checkpoint to resume from (model, EMA, optimizer, and saved schedule restored)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Train, validate, inspect, and export the rotation detector.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{train,finetune,validate,browse,gradcam,dfr,export}",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="full training run",
        description="Train the rotation detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_training_args(train_parser)
    train_parser.set_defaults(handler=run_train)

    finetune_parser = subparsers.add_parser(
        "finetune",
        help="short ImageNet fine-tune preset",
        description=(
            "Fine-tune from ImageNet weights: "
            + ", ".join(f"{k}={v}" for k, v in FINETUNE_DEFAULTS.items() if k != "pretrain")
            + "; explicit flags override"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_training_args(finetune_parser)
    finetune_parser.set_defaults(**FINETUNE_DEFAULTS)
    finetune_parser.set_defaults(handler=run_train)

    validate_parser = subparsers.add_parser(
        "validate",
        help="evaluate a checkpoint on the val split",
        description="Validate a single checkpoint against the val split under --data-root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_args(validate_parser)
    add_checkpoint_args(validate_parser)
    validate_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="stop after N batches (quick smoke test; default: full val set)",
    )
    validate_parser.set_defaults(handler=run_validate)

    browse_parser = subparsers.add_parser(
        "browse",
        help="browse a directory with live rotation predictions",
        description="Browse a directory with live rotation predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    browse_parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="root directory to scan recursively for images",
    )
    add_checkpoint_args(browse_parser)
    browse_parser.add_argument(
        "--device",
        default="cuda",
        help="torch device string",
    )
    browse_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.95,
        dest="conf_threshold",
        help="skip display when max four-view softmax probability exceeds this",
    )
    browse_parser.add_argument(
        "--abstain-margin",
        type=float,
        default=None,
        dest="abstain_margin",
        help=(
            "if set, leave image unrotated when the top1-top2 softmax "
            "margin is below this (e.g. 0.15); default: always apply argmax"
        ),
    )
    browse_parser.set_defaults(handler=run_browse)

    gradcam_parser = subparsers.add_parser(
        "gradcam",
        help="GradCAM visualisation for a few images",
        description="Show the image at all four rotations with GradCAM overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gradcam_parser.add_argument(
        "--device",
        default="cuda",
        help="torch device string",
    )
    gradcam_parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="image files to visualise",
    )
    gradcam_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for the rendered PNG grids",
    )
    add_checkpoint_args(gradcam_parser)
    gradcam_parser.set_defaults(handler=run_gradcam)

    dfr_parser = subparsers.add_parser(
        "dfr",
        help="retrain the head on frozen features",
        description="Deep Feature Reweighting: retrain only the head on frozen features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_checkpoint_args(dfr_parser)
    dfr_parser.add_argument(
        "--data-root",
        type=Path,
        nargs="+",
        required=True,
        help="one or more dirs of hard / in-domain images to refit the head on",
    )
    dfr_parser.add_argument(
        "--output",
        type=Path,
        default=constants.CHECKPOINT_DIR / "dfr_head.pt",
        help="where to write the DFR-retrained checkpoint",
    )
    dfr_parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="torch device string (e.g. 'cuda', 'cuda:0', 'cpu')",
    )
    dfr_parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="images per step",
    )
    dfr_parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="DataLoader worker processes for image decoding",
    )
    dfr_parser.add_argument(
        "--prefetch-factor",
        type=int,
        required=True,
        help="batches each worker keeps queued ahead of the consumer",
    )
    dfr_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="head-retraining epochs (DFR converges fast on frozen features)",
    )
    dfr_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="head learning rate (constant)",
    )
    dfr_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="head weight decay",
    )
    dfr_parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="max grad norm for the head",
    )
    dfr_parser.add_argument(
        "--no-reinit-head",
        action="store_true",
        help="fine-tune the existing head instead of refitting it from scratch",
    )
    dfr_parser.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached file indexes before scanning the reweighting roots",
    )
    dfr_parser.set_defaults(handler=run_dfr)

    export_parser = subparsers.add_parser(
        "export",
        help="export a checkpoint to ONNX",
        description="Export a rotation-detector checkpoint to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_checkpoint_args(export_parser)
    export_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .onnx path (default: checkpoint with .onnx suffix)",
    )
    export_parser.add_argument(
        "--device",
        default="cpu",
        help="torch device used for the export trace",
    )
    export_parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    export_parser.add_argument(
        "--dynamo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the torch.export/dynamo ONNX exporter (default: legacy)",
    )
    export_parser.add_argument(
        "--verify",
        action="store_true",
        help="compare torch vs onnxruntime on a dummy batch after export",
    )
    export_parser.set_defaults(handler=run_export)

    return parser


def run_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # Pretrained features expect the ImageNet normalization they were trained
    # with; from-scratch runs compute stats over the train split instead.
    pretrain_stats = (
        core.NormStats(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        if args.pretrain
        else None
    )
    datasets = config.build_datasets(args.data_root)
    ds, stats = core.build_dataset(
        split="train",
        datasets=datasets,
        device=device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        prefetch_factor=args.prefetch_factor,
        stats=pretrain_stats,
        invalidate_cache=args.invalidate_cache,
        erasing_p=args.erasing_p,
        crop_policy=args.crop_policy,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=device.type == "cuda",
        shuffle=True,
    )

    core.train(
        device=device,
        train_loader=dl,
        seed=args.seed,
        checkpoint_dir=constants.CHECKPOINT_DIR,
        stats=stats,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        drop_rate=args.dropout,
        drop_path_rate=args.drop_path,
        pooling=args.pooling,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        ema_horizon_epochs=args.ema_horizon_epochs,
        use_sam=args.sam,
        sam_rho=args.sam_rho,
        backbone_lr_mult=args.backbone_lr_mult,
        pretrained=args.pretrain,
        compile_model=not args.no_compile,
        resume_from=args.resume,
    )


def run_validate(args: argparse.Namespace) -> None:
    core.validate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=torch.device(args.device),
        which_weights=args.weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_batches=args.max_batches,
    )


def run_browse(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    print(f"Loading {args.checkpoint}  ({args.weights} weights)...")
    _, model, stats = core.load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    print("Model ready.\n")

    root_dir = Path(args.dir)
    print(f"Scanning {root_dir}...")
    paths = core.scan_images(root_dir)
    if not paths:
        print("No images found.")
        return
    print(f"Found {len(paths)} image(s).\n")

    result_queue: queue.Queue[core.InferResult | object] = queue.Queue(maxsize=core.PREFETCH_SIZE)
    bar = tqdm(total=len(paths), desc="Inferring", unit="img", dynamic_ncols=True)

    worker = threading.Thread(
        target=core.prefetch_worker,
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
        if item is core.DONE:
            break

        assert isinstance(item, core.InferResult)
        processed += 1

        best_norm = max(item.vote.normalized_probs.values())
        if not item.error and best_norm >= args.conf_threshold:
            skipped += 1
            tqdm.write(
                f"[{processed}/{len(paths)}]  {item.path.name}  skipped (conf={best_norm:.3f} >= {args.conf_threshold})"
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

        core.show_result(item, shown, len(paths))

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
    _, model, stats = core.load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    preprocess = core.build_eval_transform(mean=stats.mean, std=stats.std)

    args.out_dir.mkdir(exist_ok=True)

    for img_path in images:
        pil = open_rgb(img_path)
        displays, probs, cams, tensors = core.compute_rotations(model, pil, preprocess, device)
        angle = core.circular_mean_angle(probs)

        core.visualise(
            displays,
            probs,
            cams,
            f"{img_path.name} - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_gradcam.png",
        )
        print(f"{img_path.name} - predicted angle {angle:.1f}°")

        guided_cams = core.compute_guided_gradcam_batch(model, tensors, cams)
        core.visualise_grid(
            displays,
            probs,
            guided_cams,
            f"{img_path.name} - guided GradCAM - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_guidedcam.png",
            colormap="inferno",
        )


def run_dfr(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    payload, model, stats = core.load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
        strict_stats=True,
    )
    logger.info(
        "loaded %s (epoch=%s, weights=%s)",
        args.checkpoint,
        payload.get("epoch"),
        args.weights,
    )
    model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    core.freeze_backbone(model, reinit_head=not args.no_reinit_head)

    loader = core.build_head_loader(
        data_roots=list(args.data_root),
        stats=stats,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        invalidate_cache=args.invalidate_cache,
    )

    core.retrain_head(
        model=model,
        loader=loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    core.save_dfr_checkpoint(
        path=args.output,
        model=model,
        stats=stats,
        source_checkpoint=args.checkpoint,
    )


def run_export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    payload, model, stats = core.load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    meta = core.build_onnx_meta(
        payload,
        stats,
        checkpoint=args.checkpoint,
        weights=args.weights,
    )
    logger.info(
        "loaded %s (epoch=%s, weights=%s, pooling=%s)",
        args.checkpoint,
        meta.get("epoch"),
        args.weights,
        meta.get("pooling"),
    )

    out_path: Path = args.out
    if out_path is None:
        out_path = args.checkpoint.with_suffix(".onnx")

    core.export_onnx(
        model=model,
        out_path=out_path,
        device=device,
        opset=args.opset,
        dynamo=args.dynamo,
        verify=args.verify,
    )
    core.write_onnx_meta(meta, out_path, opset=args.opset, dynamo=args.dynamo)
    logger.info("done: %s (%.1f MiB)", out_path, out_path.stat().st_size / (1024 * 1024))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(constants.LOGS_DIR)
    args.handler(args)


if __name__ == "__main__":
    main()
