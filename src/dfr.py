"""Deep Feature Reweighting (DFR): retrain only the head on frozen features.

The hard tail of the orientation task is a worst-group / spurious-correlation
failure - the model leans on easy cues (faces, sky/ground, shadows) that are
absent in the difficult 1% (flat-lay, documents, symmetric interiors,
top-down). DFR (Kirichenko et al., 2022) shows the backbone already learns
good features; the problem is that the head *weights* them badly. So we freeze
the backbone, reinitialise the head, and retrain just the head on a small set
balanced toward the hard / in-domain cases.

This is a post-hoc edit of an existing checkpoint - no full retrain. Point
``--data-root`` at in-domain images (e.g. a sample of the Immich library).
The output is a new checkpoint in the same format ``validator.py`` expects
(weights stored under both ``model`` and ``ema``), so you can validate it
the usual way.

Run from the project root:
    venv/bin/python src/dfr.py \\
        --checkpoint checkpoints/epoch_029.pt --weights ema \\
        --data-root /path/to/hard_images \\
        --device cuda --batch-size 256 --num-workers 8 --prefetch-factor 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import constants
from builders import build_train_transform
from datasets import ImageLoader, RotationPairDataset
from models import RotationModel
from stats import NormStats
from utils.index import index_files
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)


def load_checkpoint_model(
    checkpoint_path: Path,
    which_weights: str,
    device: torch.device,
) -> tuple[RotationModel, NormStats]:
    """Load a checkpoint's weights into a fresh RotationModel + its stats."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key = {"ema": "ema", "raw": "model"}[which_weights]
    if state_key not in payload:
        raise KeyError(f"checkpoint missing '{state_key}' state dict; keys={list(payload)}")

    if "stats" not in payload:
        raise KeyError("checkpoint has no 'stats'; DFR needs the normalization the backbone was trained with")
    raw_stats = payload["stats"]
    stats = NormStats(
        mean=tuple(raw_stats["mean"]),  # type: ignore[arg-type]
        std=tuple(raw_stats["std"]),  # type: ignore[arg-type]
    )

    model = RotationModel(
        pretrained=False,
        pooling=payload.get("pooling", "avg"),
    ).to(device)
    model.load_state_dict(payload[state_key])
    logger.info(
        "loaded %s (epoch=%s, weights=%s)",
        checkpoint_path,
        payload.get("epoch"),
        which_weights,
    )
    return model, stats


def freeze_backbone(model: RotationModel, reinit_head: bool) -> None:
    """Freeze the backbone and pooling layer; optionally reinitialise the head.

    True DFR refits the classifier from scratch on the reweighting data, so
    ``reinit_head`` defaults on. Everything outside ``model.head`` is frozen
    (backbone + pooling layer, if it has parameters) and the backbone is put
    in eval mode so its BatchNorm running stats stay fixed at the values the
    full model converged to.
    """
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for p in model.pool.parameters():
        p.requires_grad_(False)
    model.backbone.eval()

    if reinit_head:
        for module in model.head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore[operator]
        logger.info("reinitialised head for a fresh DFR fit")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info("trainable params: %d | frozen params: %d", trainable, frozen)


def build_head_loader(
    data_roots: list[Path],
    stats: NormStats,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    invalidate_cache: bool,
) -> DataLoader:
    """Index the reweighting roots and wrap them as a rotation-pair loader.

    Indexed under its own cache name so it never collides with the main
    train/val file indexes.
    """
    files = index_files(
        name="dfr",
        roots=data_roots,
        cache_dir=constants.INDEX_CACHE_DIR,
        invalidate_cache=invalidate_cache,
    )
    transform = build_train_transform(mean=stats.mean, std=stats.std)
    dataset = RotationPairDataset(
        base=ImageLoader(files=files),
        transform=transform,
    )
    logger.info("DFR reweighting set: %d files -> %d pairs", len(files), len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=device.type == "cuda",
        shuffle=True,
    )


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
            imgs = imgs.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            labels = labels.to(device, non_blocking=True).float()

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


def save_dfr_checkpoint(
    path: Path,
    model: RotationModel,
    stats: NormStats,
    source_checkpoint: Path,
) -> None:
    """Save the DFR-retrained model in the format validator.py expects.

    The retrained weights go into both ``model`` and ``ema`` so validating
    with either ``--weights ema`` or ``--weights raw`` picks them up.
    """
    sd = model.state_dict()
    payload = {
        "model": sd,
        "ema": sd,
        "stats": {"mean": list(stats.mean), "std": list(stats.std)},
        "epoch": "dfr",
        "pooling": model.pooling,
        "dfr_source": str(source_checkpoint),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("saved DFR checkpoint %s", path)


def main(args) -> None:
    setup_logging(constants.LOGS_DIR)
    device = torch.device(args.device)

    model, stats = load_checkpoint_model(
        checkpoint_path=args.checkpoint,
        which_weights=args.weights,
        device=device,
    )
    model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    freeze_backbone(model, reinit_head=not args.no_reinit_head)

    loader = build_head_loader(
        data_roots=list(args.data_root),
        stats=stats,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        invalidate_cache=args.invalidate_cache,
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

    save_dfr_checkpoint(
        path=args.output,
        model=model,
        stats=stats,
        source_checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Deep Feature Reweighting: retrain only the head on frozen features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="source checkpoint whose backbone features are reused",
    )
    p.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="which weights to start from: EMA shadow (production) or raw",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        nargs="+",
        required=True,
        help="one or more dirs of hard / in-domain images to refit the head on",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=constants.CHECKPOINT_DIR / "dfr_head.pt",
        help="where to write the DFR-retrained checkpoint",
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
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="head-retraining epochs (DFR converges fast on frozen features)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="head learning rate (constant)",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="head weight decay",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="max grad norm for the head",
    )
    p.add_argument(
        "--no-reinit-head",
        action="store_true",
        help="fine-tune the existing head instead of refitting it from scratch",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached file indexes before scanning the reweighting roots",
    )
    args = p.parse_args()

    main(args)
