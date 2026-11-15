"""Deep Feature Reweighting: head-only retraining."""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import constants
from checkpoint import load_checkpoint_model
from data import RotationPairDataset, build_loader, build_train_transform
from model import RotationModel
from training import move_batch
from utils.index import index_files

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

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
