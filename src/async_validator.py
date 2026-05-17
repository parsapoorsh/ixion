"""Async validator running on a second GPU.

Spawns a child process pinned to ``cuda:<device_index>`` that owns its
own val loader and a copy of the model. The parent calls
``submit(ema_module, epoch, phase)`` at the 50% and 100% marks of each
epoch; the child returns ``ValidationResult`` objects which the parent
drains opportunistically between training steps. The train loop never
blocks on validation.

Dtype is fp16, not bf16 — the second card on this machine is a 1660
SUPER (Turing) which has no BF16 tensor cores.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader

import config  # noqa: F401  # establishes datasets/stats/config load order before `from datasets` below
import constants
from datasets import build_val_dataset
from models import RotationModel
from utils.logsetup import setup_logging

if TYPE_CHECKING:
    from datasets import DatasetSpec
    from models import Backbone, TrainDefaults

logger = logging.getLogger(__name__)

VALIDATOR_WORKERS = 4
SHUTDOWN_TIMEOUT_S = 30.0


@dataclass
class ValidationTask:
    epoch: int
    phase: str  # "mid" or "end"
    state_dict: dict


@dataclass
class ValidationResult:
    epoch: int
    phase: str
    loss: float
    acc: float
    per_class: list[float]
    seconds: float


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, list[float]]:
    model.train(False)
    correct = torch.zeros(constants.NUM_CLASSES, device=device)
    seen = torch.zeros(constants.NUM_CLASSES, device=device)
    total_loss = 0.0
    total_n = 0
    loss_fn = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        for c in range(constants.NUM_CLASSES):
            mask = labels == c
            seen[c] += mask.sum()
            correct[c] += (preds[mask] == c).sum()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

    return (
        total_loss / max(1, total_n),
        (correct.sum() / seen.sum().clamp(min=1)).item(),
        (correct / seen.clamp(min=1)).cpu().tolist(),
    )


def validator_worker(
    task_q: mp.Queue,
    result_q: mp.Queue,
    backbone: Backbone,
    recipe: TrainDefaults,
    dataset_specs: list[DatasetSpec],
    device_index: int,
) -> None:
    setup_logging()
    log = logging.getLogger(__name__)
    log.info("validator worker starting on cuda:%d", device_index)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    val_ds = build_val_dataset(
        dataset_specs,
        input_size=recipe.input_size,
        resize_size=recipe.resize_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=recipe.batch_size,
        shuffle=False,
        num_workers=VALIDATOR_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=constants.PREFETCH_FACTOR,
    )
    log.info("val pairs: %d", len(val_ds))

    model = RotationModel(
        backbone=backbone,
        pretrained=False,
    ).to(device)

    while True:
        task = task_q.get()
        if task is None:
            log.info("validator worker shutting down")
            return
        model.load_state_dict(task.state_dict)
        t0 = time.monotonic()
        loss, acc, per_class = run_validation(
            model,
            val_loader,
            device=device,
            dtype=torch.float16,
        )
        dt = time.monotonic() - t0
        log.info(
            "validation epoch %d phase=%s loss=%.4f acc=%.4f per_class=%s in %.0fs",
            task.epoch, task.phase, loss, acc, per_class, dt,
        )
        result_q.put(
            ValidationResult(
                epoch=task.epoch,
                phase=task.phase,
                loss=loss,
                acc=acc,
                per_class=per_class,
                seconds=dt,
            )
        )


class AsyncValidator:
    def __init__(
        self,
        backbone: Backbone,
        recipe: TrainDefaults,
        dataset_specs: list[DatasetSpec],
        device_index: int = 1,
    ) -> None:
        ctx = mp.get_context("spawn")
        self.task_q: mp.Queue = ctx.Queue(maxsize=1)
        self.result_q: mp.Queue = ctx.Queue()
        self.process = ctx.Process(
            target=validator_worker,
            args=(
                self.task_q,
                self.result_q,
                backbone,
                recipe,
                dataset_specs,
                device_index,
            ),
            daemon=True,
        )
        self.process.start()
        logger.info(
            "spawned async validator on cuda:%d (pid=%d)",
            device_index, self.process.pid,
        )

    def submit(
        self,
        ema_module: nn.Module,
        epoch: int,
        phase: str,
    ) -> None:
        cpu_sd = {k: v.detach().cpu() for k, v in ema_module.state_dict().items()}
        try:
            self.task_q.put_nowait(
                ValidationTask(epoch=epoch, phase=phase, state_dict=cpu_sd),
            )
        except queue.Full:
            logger.warning(
                "skipped validation epoch %d phase=%s: prior val still running",
                epoch, phase,
            )

    def drain(self) -> list[ValidationResult]:
        out: list[ValidationResult] = []
        while True:
            try:
                out.append(self.result_q.get_nowait())
            except queue.Empty:
                return out

    def shutdown(self) -> None:
        try:
            self.task_q.put(None, timeout=5.0)
        except queue.Full:
            pass
        self.process.join(timeout=SHUTDOWN_TIMEOUT_S)
        if self.process.is_alive():
            logger.warning(
                "validator did not exit in %.0fs, terminating",
                SHUTDOWN_TIMEOUT_S,
            )
            self.process.terminate()
            self.process.join(timeout=5.0)
