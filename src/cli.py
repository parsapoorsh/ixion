#!/usr/bin/env python3
"""Command-line entry point for training.

Subcommands:
    cli.py train ARGS      - full training run
    cli.py finetune ARGS   - short ImageNet fine-tune preset

The training stack (torch, timm, torchvision) is imported inside ``run`` so
``--help`` answers without loading any of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def add_training_args(p: argparse.ArgumentParser) -> None:
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
        help="images per training step",
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
        description="Train the rotation detector.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{train,finetune}",
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

    return parser


def run(args: argparse.Namespace) -> None:
    # Imported here so `cli.py --help` skips the whole torch/timm import chain.
    import torch
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from torch.utils.data import DataLoader

    import config
    import constants
    import train
    from datasets import build_dataset
    from stats import NormStats
    from utils.logsetup import setup_logging

    setup_logging(constants.LOGS_DIR)

    device = torch.device(args.device)

    # Pretrained features expect the ImageNet normalization they were trained
    # with; from-scratch runs compute stats over the train split instead.
    pretrain_stats = (
        NormStats(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        if args.pretrain
        else None
    )
    datasets = config.build_datasets(args.data_root)
    ds, stats = build_dataset(
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

    train.train(
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


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
