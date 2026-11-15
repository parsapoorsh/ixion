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

This module only parses arguments; the subcommand bodies live in ``core``,
which is imported after parsing so ``cli.py --help`` never loads torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import constants

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


def add_runtime_args(p: argparse.ArgumentParser) -> None:
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


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=("root directory that contains the ImageNet 2017 and COCO 2017 layout (see config.build_dataset_specs)"),
    )
    add_runtime_args(p)


def add_optional_device(
    p: argparse.ArgumentParser,
    default: str = "cuda",
    help_text: str = "torch device string",
) -> None:
    p.add_argument(
        "--device",
        default=default,
        help=help_text,
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
    add_optional_device(browse_parser)
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

    gradcam_parser = subparsers.add_parser(
        "gradcam",
        help="GradCAM visualisation for a few images",
        description="Show the image at all four rotations with GradCAM overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_optional_device(gradcam_parser)
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
    add_runtime_args(dfr_parser)
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
    add_optional_device(export_parser, default="cpu", help_text="torch device used for the export trace")
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

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Import the training stack only when a subcommand runs, so `cli.py --help`
    # stays free of torch/timm.
    import core

    core.run(args)


if __name__ == "__main__":
    main()
