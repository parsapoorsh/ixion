"""Single-phase training loop.

Run from the project root:
    venv/bin/python src/train.py --device cuda --batch-size N \\
        --num-workers N --prefetch-factor N

Optimisation knobs are plain kwargs on ``train_one_run`` with the
project's defaults baked in. ``RunConfig`` carries the process/IO
knobs. Validation is a separate process - see ``validator.py``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
import constants
import stats
from datasets import build_train_dataset
from models import ModelEMA, RotationModel
from utils.index import clear_cache
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Process / IO / reproducibility - independent of recipe."""
    device: torch.device
    batch_size: int
    num_workers: int
    prefetch_factor: int
    seed: int
    log_every: int = constants.LOG_STEP


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    extras: dict | None = None,
) -> None:
    # torch.compile() wraps the module and prefixes state_dict keys with
    # "_orig_mod." - unwrap so checkpoints stay portable across compile.
    raw_model = getattr(model, "_orig_mod", model)
    payload: dict = {
        "model": raw_model.state_dict(),
        "ema": ema.state_dict(),
        "ema_step": ema.step,
        "optimizer": optimizer.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }
    if extras:
        payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def cosine_with_warmup(
    step: int,
    total_steps: int,
    warmup: int,
) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def build_train_loader(run: RunConfig) -> DataLoader:
    logger.info("building train dataset...")
    train_ds = build_train_dataset(
        config.DATASETS,
        input_size=constants.INPUT_SIZE,
        resize_size=constants.RESIZE_SIZE,
        num_workers=run.num_workers,
        device=run.device,
    )
    logger.info("train pairs: %d", len(train_ds))
    return DataLoader(
        train_ds,
        batch_size=run.batch_size,
        shuffle=True,
        num_workers=run.num_workers,
        pin_memory=run.device.type == "cuda",
        persistent_workers=run.num_workers > 0,
        prefetch_factor=run.prefetch_factor if run.num_workers > 0 else None,
        drop_last=True,
    )


def train_one_run(
    run: RunConfig,
    epochs: int = 100,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    warmup_epochs: int = 10,
    grad_clip: float = 1.0,
    ema_decay: float = 0.9999,
    equiv_weight: float = 0.5,
    resume_from: Path | None = None,
) -> None:
    torch.manual_seed(run.seed)
    torch.backends.cudnn.benchmark = True
    train_loader = build_train_loader(run)

    model = RotationModel(pretrained=False).to(run.device)
    model = model.to(memory_format=torch.channels_last)  # type: ignore[call-arg]
    ema = ModelEMA(model, decay=ema_decay)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    logger.info(
        "%d epochs x %d steps = %d total (warmup %d steps)",
        epochs, steps_per_epoch, total_steps, warmup_steps,
    )

    start_epoch = 0
    if resume_from is not None:
        payload = torch.load(resume_from, map_location=run.device, weights_only=False)
        model.load_state_dict(payload["model"])
        ema.module.load_state_dict(payload["ema"])
        ema.step = int(payload.get("ema_step", 0))
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "rng_cpu" in payload:
            torch.set_rng_state(payload["rng_cpu"].cpu())
        if payload.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
        start_epoch = int(payload["epoch"]) + 1
        logger.info(
            "resumed from %s (next epoch=%d, ema_step=%d)",
            resume_from, start_epoch, ema.step,
        )

    # Compile after any state_dict load so checkpoints are saved/loaded in
    # uncompiled form (no "_orig_mod." prefix in the on-disk dict).
    model = torch.compile(model)

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.monotonic()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        running_grad_norm = 0.0
        running_grad_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            unit="batch",
            leave=False,
            file=sys.stdout,
        )
        for imgs, labels in pbar:
            imgs = imgs.to(
                run.device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            labels = labels.to(run.device, non_blocking=True)

            scale = cosine_with_warmup(
                global_step,
                total_steps,
                warmup_steps,
            )

            # set lr
            for group in optimizer.param_groups:
                group["lr"] = lr * scale

            with torch.autocast(device_type=run.device.type, dtype=torch.bfloat16):  # pyright: ignore[reportPrivateImportUsage]
                logits = model(imgs)
                ce_loss = loss_fn(logits, labels)
                loss = ce_loss
                if equiv_weight > 0:
                    # Enforce f(rot(x, k*90)) == (f(x) + k) mod 4. Pairs the
                    # same images at a different rotation in one batch so
                    # the gradient directly penalises rotational
                    # inconsistency, rather than relying on shuffled data
                    # to surface that pairing implicitly.
                    k = int(torch.randint(1, 4, (1,)).item())
                    rot_imgs = torch.rot90(
                        imgs, k=k, dims=[-2, -1],
                    ).contiguous(memory_format=torch.channels_last)
                    rot_labels = (labels + k) % 4
                    rot_logits = model(rot_imgs)
                    loss = ce_loss + equiv_weight * loss_fn(rot_logits, rot_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
            optimizer.step()
            ema.update(model)

            running_loss += ce_loss.item() * imgs.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_seen += imgs.size(0)
            running_grad_norm += grad_norm.item()
            running_grad_steps += 1

            lr_now = optimizer.param_groups[0]["lr"]
            acc = running_correct / max(1, running_seen)
            avg_loss = running_loss / max(1, running_seen)
            avg_grad_norm = running_grad_norm / max(1, running_grad_steps)
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{acc:.3f}",
                lr=f"{lr_now:.2e}",
                gn=f"{avg_grad_norm:.2f}",
            )
            if global_step % run.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d loss=%.4f acc=%.3f lr=%.2e grad_norm=%.2f",
                    epoch, global_step, total_steps,
                    avg_loss, acc, lr_now, avg_grad_norm,
                )
            global_step += 1

        dt = time.monotonic() - epoch_start
        logger.info(
            "epoch %d done in %.0fs, throughput %.1f img/s",
            epoch, dt, running_seen / dt,
        )

        ckpt = constants.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            ckpt,
            model,
            ema,
            optimizer,
            extras={"epoch": epoch},
        )
        logger.info("saved %s", ckpt)

    logger.info("done.")


def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser(
        description="Train the rotation detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--invalidate-cache",
        action="store_true",
        help="delete cached file-index and per-dataset stats JSONs before running",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to an epoch_NNN.pt checkpoint to resume from (model, ema, ema_step restored; optimizer re-warms)",
    )
    p.add_argument(
        "--find-lr",
        action="store_true",
        help="run an LR range test (~500 steps) and exit; writes checkpoints/lr_finder.png",
    )
    args = p.parse_args()

    if args.invalidate_cache:
        clear_cache()
        stats.clear_cache()

    run = RunConfig(
        device=torch.device(args.device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )

    if args.find_lr:
        from lr_finder import find_lr
        recommended = find_lr(run)
        logger.info("recommended lr=%.2e", recommended)
        print(f"recommended lr = {recommended:.3e}")
        return

    train_one_run(run, resume_from=args.resume)


if __name__ == "__main__":
    main()
