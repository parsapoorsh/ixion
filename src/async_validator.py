"""Async validator running on a second GPU.

Spawns a child process pinned to ``cuda:<device_index>`` that owns its
own val loader and a copy of the model. The parent calls
``submit(ema_module, epoch, phase)`` at the 50% and 100% marks of each
epoch; the child returns ``ValidationResult`` objects which the parent
drains opportunistically between training steps. The train loop never
blocks on validation.

The same child also handles an "ambiguity scan" task type — walk the
val splits, run each image through all 4 rotations, and write the
paths of images the model finds non-confident (≤3 of 4 views agree)
to ``cache/ambiguity.json``, sorted most-ambiguous first.

Dtype is fp16, not bf16 — the second card on this machine is a 1660
SUPER (Turing) which has no BF16 tensor cores.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from logging.handlers import QueueHandler
from typing import TYPE_CHECKING

import torch
from PIL import UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

import config  # noqa: F401  # establishes datasets/stats/config load order before `from datasets` below
import constants
import stats
from builders import build_eval_transform
from datasets import build_val_dataset
from models import RotationModel
from utils.image_io import open_rgb
from utils.index import index_files

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path  # noqa: F401  # used in type annotations elsewhere

    from datasets import DatasetSpec
    from models import Backbone, TrainDefaults

logger = logging.getLogger(__name__)

VALIDATOR_WORKERS = 8
VALIDATOR_PREFETCH = 6
SHUTDOWN_TIMEOUT_S = 30.0
VALIDATION_LOG_EVERY = 50
SCAN_LOG_EVERY = 50
# Per-image scan batch. Each image expands to 4 rotation views before the
# forward pass, so this value × 4 is what actually hits the model. Sized
# for the GTX 1660 SUPER's ~5.6 GiB usable VRAM at the 4 GiB target —
# measured on EfficientNet-B0 at 224²: peak_reserved=3.92 GiB. Reduce if
# you swap to a heavier backbone (ConvNeXt-V2-Tiny activations are
# ~2-3× B0's).
SCAN_BATCH_SIZE = 128
AMBIGUITY_OUTPUT_PATH = constants.INDEX_CACHE_DIR / "ambiguity.json"


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


@dataclass
class AmbiguityScanTask:
    epoch: int
    state_dict: dict


@dataclass
class AmbiguityScanResult:
    epoch: int
    output_path: Path
    n_total: int
    n_weak: int
    n_ambiguous: int
    seconds: float


class AmbiguityScanDataset(Dataset):
    """Yields (views[4,C,H,W], idx) per image — all four rotations stacked."""

    def __init__(
        self,
        files,
        transform: Callable,
    ) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        n = len(self.files)
        for offset in range(n):
            probe = (idx + offset) % n
            try:
                img = open_rgb(self.files[probe])
                break
            except (UnidentifiedImageError, OSError) as exc:
                logger.warning(
                    "skipping unreadable image %s: %s",
                    self.files[probe], exc,
                )
        else:
            raise RuntimeError("no readable images remain in the dataset")
        views = torch.stack(
            [self.transform(TF.rotate(img, r, expand=False)) for r in constants.ROTATIONS]
        )
        return views, probe


@torch.no_grad()
def scan_dataset(
    model: nn.Module,
    dataset_specs: list[DatasetSpec],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    recipe_input_size: int,
    recipe_resize_size: int,
) -> tuple[list[str], int, int]:
    """Walk the val splits, 4-rotation-score each image, return the paths the
    model finds non-confident (≤3 views agree), sorted ascending by
    rotation-equivariant posterior max. Returns (paths, n_weak, n_ambiguous).
    """
    model.train(False)
    scored: list[tuple[float, str]] = []
    seen_paths: set[str] = set()
    n_weak = 0
    n_ambiguous = 0

    n = constants.NUM_CLASSES
    r_arange = torch.arange(n, device=device)
    posterior_idx = (
        (torch.arange(n, device=device)[None, :] + torch.arange(n, device=device)[:, None]) % n
    )  # shape (n, n): row s lists which probs column to take from view i

    for spec in dataset_specs:
        files = index_files(f"val_{spec.name}", [spec.val])
        channel_stats = stats.get_stats(
            spec,
            input_size=recipe_input_size,
            resize_size=recipe_resize_size,
        )
        transform = build_eval_transform(
            mean=channel_stats.mean,
            std=channel_stats.std,
            input_size=recipe_input_size,
            resize_size=recipe_resize_size,
        )
        ds = AmbiguityScanDataset(files, transform)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=VALIDATOR_WORKERS,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=VALIDATOR_PREFETCH,
        )
        logger.info("scanning %s (%d images)", spec.name, len(ds))
        n_batches = len(loader)

        for batch_idx, (views, idxs) in enumerate(loader):
            if batch_idx % SCAN_LOG_EVERY == 0:
                logger.info(
                    "scan %s batch %d/%d kept_so_far=%d",
                    spec.name, batch_idx, n_batches, len(scored),
                )
            b = views.size(0)
            views = views.to(device, non_blocking=True).view(b * n, *views.shape[2:])
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits = model(views)
            probs = logits.float().softmax(dim=-1).view(b, n, n)

            implied = (probs.argmax(dim=-1) - r_arange) % n  # (b, n)
            top_count = (
                nn.functional.one_hot(implied, n).sum(dim=1).max(dim=-1).values
            )  # (b,)
            shifted = probs.gather(
                -1,
                posterior_idx.expand(b, n, n),
            )
            posterior = shifted.mean(dim=1)
            confidence = posterior.max(dim=-1).values  # (b,)

            mask = top_count < n
            if not bool(mask.any()):
                continue
            keep_idxs = idxs[mask.cpu()].tolist()
            keep_conf = confidence[mask].cpu().tolist()
            keep_counts = top_count[mask].cpu().tolist()
            for ki, kc, tc in zip(keep_idxs, keep_conf, keep_counts):
                # AmbiguityScanDataset retries past unreadable images by
                # probing forward, so the same idx can arrive in multiple
                # batches. Dedupe here so a path lands in scored at most
                # once and n_weak/n_ambiguous don't double-count.
                path = str(files[ki])
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                scored.append((kc, path))
                if tc == n - 1:
                    n_weak += 1
                else:
                    n_ambiguous += 1

    scored.sort(key=lambda t: t[0])
    return [p for _, p in scored], n_weak, n_ambiguous


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
    n_batches = len(loader)

    for batch_idx, (imgs, labels) in enumerate(loader):
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
        if batch_idx % VALIDATION_LOG_EVERY == 0:
            logger.info(
                "validation batch %d/%d running_acc=%.4f",
                batch_idx, n_batches,
                (correct.sum() / seen.sum().clamp(min=1)).item(),
            )

    return (
        total_loss / max(1, total_n),
        (correct.sum() / seen.sum().clamp(min=1)).item(),
        (correct / seen.clamp(min=1)).cpu().tolist(),
    )


def validator_worker(
    task_q: mp.Queue,
    result_q: mp.Queue,
    log_q: mp.Queue,
    backbone: Backbone,
    recipe: TrainDefaults,
    dataset_specs: list[DatasetSpec],
    device_index: int,
) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(QueueHandler(log_q))
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
        if isinstance(task, ValidationTask):
            log.info(
                "validation starting (epoch %d phase=%s)",
                task.epoch, task.phase,
            )
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
        elif isinstance(task, AmbiguityScanTask):
            log.info("ambiguity scan starting (epoch %d)", task.epoch)
            paths, n_weak, n_ambiguous = scan_dataset(
                model,
                dataset_specs=dataset_specs,
                device=device,
                dtype=torch.float16,
                batch_size=SCAN_BATCH_SIZE,
                recipe_input_size=recipe.input_size,
                recipe_resize_size=recipe.resize_size,
            )
            AMBIGUITY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with AMBIGUITY_OUTPUT_PATH.open("w") as fh:
                json.dump(paths, fh)
            dt = time.monotonic() - t0
            log.info(
                "ambiguity scan epoch %d -> %s (weak=%d ambiguous=%d total=%d in %.0fs)",
                task.epoch, AMBIGUITY_OUTPUT_PATH, n_weak, n_ambiguous, len(paths), dt,
            )
            result_q.put(
                AmbiguityScanResult(
                    epoch=task.epoch,
                    output_path=AMBIGUITY_OUTPUT_PATH,
                    n_total=len(paths),
                    n_weak=n_weak,
                    n_ambiguous=n_ambiguous,
                    seconds=dt,
                )
            )
        else:
            log.error("unknown task type: %r", type(task))


def log_listener_loop(log_q: mp.Queue) -> None:
    """Pull LogRecord objects emitted by the validator child and dispatch
    them through the parent's root logger. The parent's TqdmLoggingHandler
    coordinates with the active progress bar, so child log lines appear
    cleanly above it instead of overwriting it.
    """
    while True:
        record = log_q.get()
        if record is None:
            return
        logging.getLogger(record.name).handle(record)


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
        self.log_q: mp.Queue = ctx.Queue()
        self.log_listener = threading.Thread(
            target=log_listener_loop,
            args=(self.log_q,),
            daemon=True,
        )
        self.log_listener.start()
        self.process = ctx.Process(
            target=validator_worker,
            args=(
                self.task_q,
                self.result_q,
                self.log_q,
                backbone,
                recipe,
                dataset_specs,
                device_index,
            ),
            # Not daemonic: this process spawns its own DataLoader workers,
            # and the multiprocessing rule is that daemonic parents can't
            # have children. shutdown() in the try/finally handles cleanup.
            daemon=False,
        )
        self.process.start()
        logger.info(
            "spawned async validator on cuda:%d (pid=%d)",
            device_index, self.process.pid,
        )

    def child_alive(self, what: str) -> bool:
        if self.process.is_alive():
            return True
        logger.error(
            "validator child is dead (exit code %s); dropping %s",
            self.process.exitcode, what,
        )
        return False

    def submit(
        self,
        ema_module: nn.Module,
        epoch: int,
        phase: str,
    ) -> None:
        if not self.child_alive(f"validation epoch {epoch} phase={phase}"):
            return
        cpu_sd = self.snapshot_state(ema_module)
        try:
            self.task_q.put_nowait(
                ValidationTask(epoch=epoch, phase=phase, state_dict=cpu_sd),
            )
        except queue.Full:
            logger.warning(
                "skipped validation epoch %d phase=%s: prior task still running",
                epoch, phase,
            )

    def submit_scan(
        self,
        ema_module: nn.Module,
        epoch: int,
    ) -> None:
        if not self.child_alive(f"ambiguity scan epoch {epoch}"):
            return
        cpu_sd = self.snapshot_state(ema_module)
        try:
            self.task_q.put_nowait(
                AmbiguityScanTask(epoch=epoch, state_dict=cpu_sd),
            )
        except queue.Full:
            logger.warning(
                "skipped ambiguity scan epoch %d: prior task still running",
                epoch,
            )

    @staticmethod
    def snapshot_state(ema_module: nn.Module) -> dict:
        return {k: v.detach().cpu() for k, v in ema_module.state_dict().items()}

    def drain(self) -> list[ValidationResult | AmbiguityScanResult]:
        out: list[ValidationResult | AmbiguityScanResult] = []
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
        self.log_q.put(None)
        self.log_listener.join(timeout=5.0)
