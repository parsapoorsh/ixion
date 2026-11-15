"""Rotation detector core: data pipeline, training, evaluation, and tools.

Everything that is not a model building block or a file-system helper lives
here: transform factories, datasets, norm stats, the training loop,
checkpoint loading, validation, four-view inference, GradCAM, the browse
browser, DFR head retraining, ONNX export, and the subcommand bodies that
``cli.py`` dispatches after parsing.

Keeping the tools in one module lets them share a single checkpoint loader,
stats decoder, and rotation-vote helper instead of each carrying a copy.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import queue
import random
import shutil
import sys
import sysconfig
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, UnidentifiedImageError
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

import constants
from config import DatasetSpec, build_datasets
from model import (
    SAM,
    ModelEMA,
    RotationModel,
    disable_running_stats,
    enable_running_stats,
)
from utils.image_io import open_rgb
from utils.index import cache_fingerprint, index_files
from utils.logsetup import setup_logging

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

N_ROTS = len(constants.ROTATIONS)

# Checkpoint keys holding the two weight sets: EMA shadow and raw weights.
WEIGHT_KEYS = {"ema": "ema", "raw": "model"}


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def stats_to_dict(stats: NormStats) -> dict[str, list[float]]:
    return {"mean": list(stats.mean), "std": list(stats.std)}


def stats_from_dict(raw: dict) -> NormStats:
    return NormStats(
        mean=tuple(raw["mean"]),  # type: ignore[arg-type]
        std=tuple(raw["std"]),  # type: ignore[arg-type]
    )


def load_stats_cache(path: Path, fingerprint: str | None = None) -> NormStats | None:
    if not path.exists():
        raise FileNotFoundError(f"stats cache not found: {path}")
    data = json.loads(path.read_text())
    if fingerprint is not None and data.get("fingerprint") != fingerprint:
        return None
    return stats_from_dict(data)


def save_stats_cache(
    path: Path,
    stats: NormStats,
    fingerprint: str | None = None,
) -> None:
    payload: dict = stats_to_dict(stats)
    if fingerprint is not None:
        payload["fingerprint"] = fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def stats_mp_context():
    if sys.platform != "win32" and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def preprocess_stats_image(path: Path) -> np.ndarray:
    try:
        image = open_rgb(
            path,
            preview_size=(constants.RESIZE_SIZE, constants.RESIZE_SIZE),
        )
        image = TF.resize(image, [constants.RESIZE_SIZE])
        image = TF.center_crop(
            image,
            [constants.INPUT_SIZE, constants.INPUT_SIZE],
        )
        array = np.array(image, dtype=np.uint8, copy=True)
        image.close()
        return array
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"unreadable image {path}: {exc}") from exc


def compute_stats_chunk(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, int, int]:
    warnings.filterwarnings(
        "ignore",
        message="Corrupt EXIF data.*",
        category=UserWarning,
    )

    sum_channels = np.zeros(3, dtype=np.uint64)
    sum_sq_channels = np.zeros(3, dtype=np.uint64)
    total_pixels = 0

    for path in paths:
        array = preprocess_stats_image(path)
        pixels = array.reshape(-1, 3).astype(np.uint64)
        sum_channels += pixels.sum(
            axis=0,
            dtype=np.uint64,
        )
        sum_sq_channels += (pixels * pixels).sum(
            axis=0,
            dtype=np.uint64,
        )
        total_pixels += pixels.shape[0]

    return sum_channels, sum_sq_channels, total_pixels, len(paths)


def iter_path_chunks(
    files: Sequence[Path],
    sample_indices: Sequence[int] | None,
    chunk_size: int,
) -> Iterable[list[Path]]:
    if sample_indices is None:
        total = len(files)
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)
            yield [files[idx] for idx in range(start, stop)]
        return

    total = len(sample_indices)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        yield [files[sample_indices[idx]] for idx in range(start, stop)]


def iter_parallel_results(
    chunks: Iterable[list[Path]],
    worker_count: int,
    prefetch_factor: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, int, int]]:
    max_pending = max(
        worker_count,
        worker_count * max(1, prefetch_factor),
    )
    context = stats_mp_context()

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        pending: set[Future[tuple[np.ndarray, np.ndarray, int, int]]] = set()
        chunk_iter = iter(chunks)
        exhausted = False

        while True:
            while len(pending) < max_pending and not exhausted:
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(compute_stats_chunk, chunk))

            if not pending:
                break

            done, pending = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                yield future.result()


def iter_stats_results(
    chunks: Iterable[list[Path]],
    worker_count: int,
    prefetch_factor: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, int, int]]:
    if worker_count <= 1:
        for chunk in chunks:
            yield compute_stats_chunk(chunk)
        return

    yield from iter_parallel_results(
        chunks=chunks,
        worker_count=worker_count,
        prefetch_factor=prefetch_factor,
    )


def compute_stats(
    files: Sequence[Path],
    device: torch.device,
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    sample_indices: Sequence[int] | None = None,
    cache_path: Path | None = None,
    invalidate_cache: bool = False,
    fingerprint: str | None = None,
) -> NormStats:
    if cache_path is not None and cache_path.exists() and not invalidate_cache:
        cached = load_stats_cache(cache_path, fingerprint)
        if cached is not None:
            return cached
        logger.info("stats cache %s does not match configuration; recomputing", cache_path)

    sample_total = len(files) if sample_indices is None else len(sample_indices)
    if sample_total == 0:
        raise RuntimeError("no images decoded; cannot compute stats")

    chunk_size = max(1, batch_size)
    worker_count = min(
        max(0, num_workers),
        sample_total,
    )
    logger.info(
        "stats worker_count=%d chunk_size=%d prefetch_factor=%d images=%d",
        worker_count,
        chunk_size,
        prefetch_factor,
        sample_total,
    )

    sum_channels = np.zeros(3, dtype=np.float64)
    sum_sq_channels = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    chunks = iter_path_chunks(
        files=files,
        sample_indices=sample_indices,
        chunk_size=chunk_size,
    )
    results = iter_stats_results(
        chunks=chunks,
        worker_count=worker_count,
        prefetch_factor=prefetch_factor,
    )

    pbar = tqdm(
        total=sample_total,
        desc="stats",
        unit="img",
        file=sys.stdout,
    )
    try:
        for partial_sum, partial_sum_sq, partial_pixels, image_count in results:
            sum_channels += partial_sum
            sum_sq_channels += partial_sum_sq
            total_pixels += partial_pixels
            pbar.update(image_count)
    finally:
        pbar.close()

    if total_pixels == 0:
        raise RuntimeError("no images decoded; cannot compute stats")

    scale = 255.0
    mean = sum_channels / (total_pixels * scale)
    variance = (sum_sq_channels / (total_pixels * scale * scale)) - mean * mean
    std = np.sqrt(np.clip(variance, 0.0, None))
    stats = NormStats(
        mean=tuple(mean.tolist()),  # type: ignore[arg-type]
        std=tuple(std.tolist()),  # type: ignore[arg-type]
    )
    if cache_path is not None:
        save_stats_cache(cache_path, stats, fingerprint)
    return stats


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    erasing_p: float = 0.25,
    crop_policy: str = "random",
) -> T.Compose:
    """Label-safe augmentation only. No spatial flip/rotate (would corrupt labels)."""
    if crop_policy == "none":
        spatial = [T.Resize(constants.INPUT_SIZE)]
    elif crop_policy == "random":
        spatial = [T.Resize(constants.RESIZE_SIZE), T.RandomCrop(constants.INPUT_SIZE)]
    elif crop_policy == "center":
        spatial = [T.Resize(constants.RESIZE_SIZE), T.CenterCrop(constants.INPUT_SIZE)]
    else:
        raise ValueError(f"unknown crop policy: {crop_policy}")

    transforms = [
        *spatial,
        T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.05,
        ),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
    if erasing_p > 0:
        # After normalize so erased patches are mean-gray in model space.
        transforms.append(
            T.RandomErasing(
                p=erasing_p,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0,
            )
        )
    return T.Compose(transforms)


eval_preprocess = T.Compose(
    [
        T.Resize(constants.RESIZE_SIZE),
        T.CenterCrop(constants.INPUT_SIZE),
        T.ToTensor(),
    ]
)


def build_eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    return T.Compose(
        [
            *eval_preprocess.transforms,
            T.Normalize(mean, std),
        ]
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ImageLoader(Dataset):
    """Loads images by index from a path source, optionally applying a transform.

    Without a transform, returns a ``PIL.Image`` (used by the rotation datasets).
    With a transform, returns the transformed tensor.

    ``files`` only needs to support ``__len__`` and ``__getitem__(idx)`` -
    anything sequence-like works (a list, an ``index.PathList`` view over a
    trie, etc.). We don't require a materialised list so the caller can
    keep the index in a compact representation.
    """

    def __init__(
        self,
        files: Sequence[Path],
        transform: Callable | None = None,
        preview_size: tuple[int, int] | None = None,
    ) -> None:
        self.files = files
        self.transform = transform
        self.preview_size = preview_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Image.Image | torch.Tensor:
        img = open_rgb(self.files[idx], preview_size=self.preview_size)
        if self.transform is not None:
            return self.transform(img)
        return img


def load_base_image(base: ImageLoader, img_idx: int) -> Image.Image:
    """Load ``base[img_idx]``, failing loudly on unreadable files.

    Substituting a neighbouring image (the old wrap-around behaviour)
    silently duplicated samples and skewed validation metrics, so corrupt
    files now surface as errors instead.
    """
    try:
        return base[img_idx]  # type: ignore[return-value]
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"unreadable image {base.files[img_idx]}: {exc}") from exc


class RotationPairDataset(Dataset):
    """Wraps a base image source so each item is (image, is_upright_label).

    For a base of N images, len = 4 * N. Sample i decomposes to
    img_idx = i // 4 and rot_idx = i % 4. The label is 1.0 when rot_idx == 0
    (upright), else 0.0.
    """

    def __init__(self, base: ImageLoader, transform: Callable) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base) * len(constants.ROTATIONS)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        img_idx, rot_idx = divmod(idx, len(constants.ROTATIONS))
        img = load_base_image(self.base, img_idx)
        tensor = self.transform(img)
        if rot_idx != 0:
            tensor = TF.rotate(
                tensor,
                constants.ROTATIONS[rot_idx],
                expand=True,
            )

        prob = 1.0 if rot_idx == 0 else 0.0
        return tensor, prob


class RotationGroupDataset(Dataset):
    """One item = all four rotations of a single base image.

    Transform runs once (shared crop / jitter / erase), then each view is
    produced with ``TF.rotate`` - same geometry as pair training and infer.
    Returns a ``(4, C, H, W)`` tensor ordered as ``constants.ROTATIONS``;
    upright is index 0 for group cross-entropy.
    """

    def __init__(self, base: ImageLoader, transform: Callable) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = load_base_image(self.base, idx)
        tensor = self.transform(img)
        views: list[torch.Tensor] = []
        for rot_idx, deg in enumerate(constants.ROTATIONS):
            if rot_idx == 0:
                views.append(tensor)
            else:
                views.append(TF.rotate(tensor, deg, expand=True))
        return torch.stack(views, dim=0)


def sample_stats_indices(files: Sequence[Path], sample_count: int) -> list[int] | None:
    """Return the seeded subset used to estimate per-channel mean/std."""
    if sample_count >= len(files):
        return None
    rng = random.Random(constants.STATS_SAMPLE_SEED)
    return sorted(rng.sample(range(len(files)), sample_count))


def build_dataset(
    split: Literal["train", "val"],
    datasets: Sequence[DatasetSpec],
    device: torch.device,
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    stats: NormStats | None = None,
    invalidate_cache: bool = False,
    erasing_p: float = 0.25,
    crop_policy: str = "random",
) -> tuple[RotationPairDataset, NormStats]:
    roots = [root for ds in datasets for root in (ds.train if split == "train" else ds.val)]
    files = index_files(
        name=split,
        roots=roots,
        cache_dir=constants.INDEX_CACHE_DIR,
        invalidate_cache=invalidate_cache,
    )

    if stats is None:
        stats_cache = constants.INDEX_CACHE_DIR / f"stats_{split}.json"
        stats_fingerprint = cache_fingerprint(
            "stats",
            sorted(str(root) for root in roots),
            constants.INPUT_SIZE,
            constants.RESIZE_SIZE,
            constants.STATS_SAMPLE_FRACTION,
            constants.STATS_SAMPLE_SEED,
        )
        sample_count = min(
            len(files),
            max(1, int(len(files) * constants.STATS_SAMPLE_FRACTION)),
        )
        logger.info(
            "computing dataset stats for %s split on %d/%d files",
            split,
            sample_count,
            len(files),
        )
        stats = compute_stats(
            files=files,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            sample_indices=sample_stats_indices(files, sample_count),
            cache_path=stats_cache,
            invalidate_cache=invalidate_cache,
            fingerprint=stats_fingerprint,
        )
        logger.info("stats - mean: %s | std: %s", stats.mean, stats.std)

    transform = (
        build_train_transform(
            mean=stats.mean,
            std=stats.std,
            erasing_p=erasing_p,
            crop_policy=crop_policy,
        )
        if split == "train"
        else build_eval_transform(mean=stats.mean, std=stats.std)
    )
    return RotationPairDataset(
        base=ImageLoader(files=files),
        transform=transform,
    ), stats


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def load_checkpoint_stats(payload: dict, strict: bool = False) -> NormStats:
    """Read mean/std from a checkpoint, falling back to the cached train stats."""
    raw_stats = payload.get("stats")
    if raw_stats:
        return stats_from_dict(raw_stats)
    if strict:
        raise KeyError("checkpoint has no 'stats'; DFR needs the normalization the backbone was trained with")

    cache_file = constants.INDEX_CACHE_DIR / "stats_train.json"
    cached = load_stats_cache(cache_file)
    if cached is None:
        raise RuntimeError(f"stats cache has no matching fingerprint: {cache_file}")
    logger.info("loaded stats from cache file %s", cache_file)
    return cached


def load_checkpoint_model(
    checkpoint: Path,
    device: torch.device,
    weights: str = "ema",
    strict_stats: bool = False,
) -> tuple[dict, RotationModel, NormStats]:
    """Load one weight set and the normalization stats from a checkpoint.

    ``weights`` picks the EMA shadow ("ema", production) or the raw training
    weights ("raw"). Returned model is in eval mode.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_key = WEIGHT_KEYS[weights]
    if state_key not in payload:
        raise KeyError(f"checkpoint missing '{state_key}' state dict; keys={list(payload)}")

    stats = load_checkpoint_stats(payload, strict=strict_stats)
    model = RotationModel(
        pretrained=False,
        pooling=payload.get("pooling", "avg"),
    ).to(device)
    model.load_state_dict(payload[state_key])
    model.eval()
    return payload, model, stats


def save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dataclass
class RotationVote:
    """Per-rotation upright probabilities and derived product fields.

    `probs` maps each rotation in degrees (0/90/180/270) to its four-view
    softmax probability (same geometry as training: transform first, then
    ``TF.rotate``).

    `normalized_probs` is retained as an alias of `probs` for callers.

    `best_deg` is argmax of `probs` - the counter-clockwise rotation
    that should be applied, matching ``TF.rotate``, to bring the image upright.

    `margin` is top1 - top2 softmax probability; small values mean ambiguous.

    `angle` is the weighted circular mean of the `probs` votes
    (secondary; prefer `best_deg` for discrete 90° correction).
    """

    probs: dict[int, float]
    normalized_probs: dict[int, float]
    best_deg: int
    margin: float
    angle: float

    def should_abstain(self, threshold: float) -> bool:
        """True when top-1 vs top-2 margin is below *threshold*."""
        return self.margin < threshold


def softmax_np(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()


def circular_mean_angle(probs: Sequence[float] | np.ndarray) -> float:
    """Weighted circular mean of the four rotation angles, in degrees."""
    angles = np.deg2rad(constants.ROTATIONS)
    values = np.asarray(probs, dtype=np.float64)
    x = float(np.sum(values * np.cos(angles)))
    y = float(np.sum(values * np.sin(angles)))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def vote_from_probs(probs: np.ndarray) -> RotationVote:
    order = np.argsort(probs)
    top1_i = int(order[-1])
    top2_i = int(order[-2]) if len(order) > 1 else top1_i
    best_deg = int(constants.ROTATIONS[top1_i])
    margin = float(probs[top1_i] - probs[top2_i])

    return RotationVote(
        probs={deg: float(p) for deg, p in zip(constants.ROTATIONS, probs, strict=True)},
        normalized_probs={deg: float(p) for deg, p in zip(constants.ROTATIONS, probs, strict=True)},
        best_deg=best_deg,
        margin=margin,
        angle=circular_mean_angle(probs),
    )


@torch.no_grad()
def predict_upright_score(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> float:
    """Return the raw sigmoid upright score for a single view.

    This score is not calibrated by the four-view group cross-entropy
    objective; prefer ``predict_rotations`` for product decisions.
    """
    transform = build_eval_transform(
        mean=mean,
        std=std,
    )
    tensor = transform(img).unsqueeze(0).to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logit = model(tensor)
    prob = float(logit.float().sigmoid().item())

    return prob


@torch.no_grad()
def predict_rotations(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> RotationVote:
    """Score four orientations with the same geometry as training.

    Train applies resize/crop/normalize first, then ``TF.rotate`` on the
    tensor (see ``RotationPairDataset``). Inference matches that order so
    checkpoints are evaluated under the distribution they saw.
    """
    transform = build_eval_transform(
        mean=mean,
        std=std,
    )
    base = transform(img)
    # expand=False: after center-crop the tensor is square; 90° keeps HxW.
    # fill=0 is mean-gray in normalized space (same family as train).
    views = [TF.rotate(base, deg, expand=False) for deg in constants.ROTATIONS]
    batch = torch.stack(views).to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch)
    probs = logits.float().softmax(dim=0).reshape(-1).cpu().numpy()

    return vote_from_probs(probs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class EpochStats:
    loss_sum: float = 0.0
    correct: int = 0
    seen: int = 0
    grad_norm_sum: float = 0.0
    grad_steps: int = 0

    def merge(
        self,
        loss_sum: float,
        correct: int,
        grad_norm_sum: float,
        n: int,
        steps: int,
    ) -> None:
        self.loss_sum += loss_sum
        self.correct += correct
        self.seen += n
        self.grad_norm_sum += grad_norm_sum
        self.grad_steps += steps

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(1, self.seen)

    @property
    def acc(self) -> float:
        return self.correct / max(1, self.seen)

    @property
    def avg_grad_norm(self) -> float:
        return self.grad_norm_sum / max(1, self.grad_steps)


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
    # Under SAM the real state lives in the wrapped base optimizer; save that
    # so the on-disk state is plain AdamW and resumes with or without --sam.
    base_optimizer = getattr(optimizer, "base_optimizer", optimizer)
    payload: dict = {
        "model": raw_model.state_dict(),
        "ema": ema.state_dict(),
        "ema_step": ema.step,
        "optimizer": base_optimizer.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }
    if extras:
        payload.update(extras)
    save_payload(path, payload)


def cosine_with_warmup(
    step: int,
    total_steps: int,
    warmup: int,
) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    t = min(1.0, t)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def can_compile_model() -> tuple[bool, str | None]:
    if shutil.which("gcc") is None:
        return False, "gcc is not available; install it with sudo apt install build-essential"

    def python_headers_available() -> bool:
        for key in ("include", "platinclude"):
            include_dir = sysconfig.get_path(key)
            if include_dir is not None and (Path(include_dir) / "Python.h").exists():
                return True
        return False

    if not python_headers_available():
        version = sysconfig.get_python_version()
        return False, (
            f"Python.h is not available under {sysconfig.get_path('include')}; "
            f"install Python headers with sudo apt install python{version}-dev"
        )
    return True, None


def autocast_dtype(device: torch.device) -> torch.dtype:
    # bfloat16 needs Ampere+ tensor cores. Older GPUs (e.g. Turing GTX 16xx)
    # emulate it, which also breaks torch.compile's layout propagation, so
    # fall back to fp32 there.
    capability = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if capability[0] >= 8 else torch.float32


def train(
    device: torch.device,
    train_loader: DataLoader,
    checkpoint_dir: Path,
    seed: int,
    stats: NormStats,
    epochs: int,
    lr: float,
    weight_decay: float,
    drop_rate: float,
    drop_path_rate: float,
    pooling: str,
    warmup_epochs: float,
    grad_clip: float,
    ema_horizon_epochs: float,
    use_sam: bool,
    sam_rho: float,
    backbone_lr_mult: float,
    pretrained: bool,
    compile_model: bool,
    resume_from: Path | None,
) -> None:
    steps_per_epoch = len(train_loader)
    # Per-step EMA decay so the shadow averages over ema_horizon_epochs
    # epochs regardless of --batch-size; the warmup covers one epoch.
    ema_decay = math.exp(-1.0 / (ema_horizon_epochs * steps_per_epoch))
    ema_warmup_steps = steps_per_epoch

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    autocast = autocast_dtype(device)

    # Peek at the checkpoint before building the model so resumes honour the
    # pooling variant it was trained with (attention vs global average).
    resume_payload = None
    if resume_from is not None:
        resume_payload = torch.load(resume_from, map_location=device, weights_only=False)

    # The checkpoint's own values win over this run's flags; say so when they
    # disagree instead of silently dropping what the user asked for.
    resume_pooling = None
    if resume_payload is not None:
        resume_pooling = str(resume_payload["pooling"])
        if resume_pooling != pooling:
            logger.info(
                "resuming with the checkpoint's pooling=%s (ignoring --pooling %s)",
                resume_pooling,
                pooling,
            )
    if pretrained and resume_payload is not None:
        logger.info("ignoring --pretrain: the resume overwrites every weight")

    # Resumes overwrite every weight anyway, so skip the download there.
    model = RotationModel(
        pretrained=pretrained and resume_from is None,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pooling=pooling if resume_pooling is None else resume_pooling,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    ema = ModelEMA(
        model,
        decay=ema_decay,
        warmup_steps=ema_warmup_steps,
    )

    # lr_scale is applied on top of the schedule each step; a lower backbone
    # scale keeps pretrained features from being wrecked while the head learns.
    param_groups = [
        {"params": model.backbone.parameters(), "lr_scale": backbone_lr_mult},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("backbone.")], "lr_scale": 1.0},
    ]
    if use_sam:
        optimizer: torch.optim.Optimizer = SAM(
            param_groups,
            base_optimizer=torch.optim.AdamW,
            rho=sam_rho,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        logger.info("using SAM (rho=%.3f) over AdamW", sam_rho)
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
    # Inference picks the rotation with the highest upright logit
    # (argmax over 4 views), so the absolute decision threshold
    # doesn't matter - only the relative ranking.
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    saved_schedule_total_steps = None
    saved_schedule_warmup_steps = None
    if resume_from is not None:
        assert resume_payload is not None
        payload = resume_payload
        model.load_state_dict(payload["model"])

        ema.module.load_state_dict(payload["ema"])
        ema.step = int(payload["ema_step"])

        # Saved state is plain base-optimizer (AdamW) state; load it into the
        # base so resume works whether or not SAM wraps it this run.
        base_optimizer = getattr(optimizer, "base_optimizer", optimizer)
        base_optimizer.load_state_dict(payload["optimizer"])

        # map_location=device pulls the saved ByteTensors to CUDA, but
        # set_rng_state_all requires CPU ByteTensors - move them back.
        torch.set_rng_state(payload["rng_cpu"].cpu())
        if payload["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in payload["rng_cuda"]])

        start_epoch = int(payload["epoch"]) + 1
        saved_schedule_total_steps = int(payload["schedule_total_steps"])
        saved_schedule_warmup_steps = int(payload["schedule_warmup_steps"])
        logger.info(
            "resumed from %s (next epoch=%d, ema_step=%d)",
            resume_from,
            start_epoch,
            ema.step,
        )

    # Compile after any state_dict load so checkpoints are saved/loaded in
    # uncompiled form (no "_orig_mod." prefix in the on-disk dict).
    if compile_model:
        can_compile, reason = can_compile_model()
        if can_compile:
            model = cast(RotationModel, torch.compile(model))
            logger.info("enabled torch.compile")
        else:
            logger.warning("disabled torch.compile: %s", reason)
    else:
        logger.info("disabled torch.compile by request")

    if pretrained and resume_from is None:
        logger.info("initialized backbone from ImageNet-pretrained weights")

    requested_total_steps = steps_per_epoch * epochs
    requested_warmup_steps = int(steps_per_epoch * warmup_epochs)
    total_steps = saved_schedule_total_steps if saved_schedule_total_steps is not None else requested_total_steps
    warmup_steps = saved_schedule_warmup_steps if saved_schedule_warmup_steps is not None else requested_warmup_steps
    if saved_schedule_total_steps is not None and (
        saved_schedule_total_steps != requested_total_steps or saved_schedule_warmup_steps != requested_warmup_steps
    ):
        logger.info(
            "checkpoint schedule wins for the LR curve: total %d, warmup %d "
            "(--epochs %d and --warmup-epochs %s would give total %d, warmup %d; "
            "the run still stops after --epochs)",
            saved_schedule_total_steps,
            saved_schedule_warmup_steps,
            epochs,
            warmup_epochs,
            requested_total_steps,
            requested_warmup_steps,
        )
    logger.info(
        "%d epochs x %d steps, schedule total %d (warmup %d steps)",
        epochs,
        steps_per_epoch,
        total_steps,
        warmup_steps,
    )

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.monotonic()
        epoch_stats = EpochStats()

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            unit="batch",
            leave=False,
            file=sys.stdout,
        )
        # GPU-side accumulators flushed to epoch_stats at LOG_STEP boundaries
        # and once at epoch end, so .item() doesn't sync every step.
        loss_buf = torch.zeros((), device=device)
        correct_buf = torch.zeros((), device=device, dtype=torch.long)
        gn_buf = torch.zeros((), device=device)
        pending_n = 0
        pending_steps = 0

        for imgs, labels in pbar:
            imgs = imgs.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            labels = labels.to(device, non_blocking=True).float()

            scale = cosine_with_warmup(
                global_step,
                total_steps,
                warmup_steps,
            )

            # set lr
            for group in optimizer.param_groups:
                group["lr"] = lr * scale * group.get("lr_scale", 1.0)

            with torch.autocast(device_type=device.type, dtype=autocast):  # pyright: ignore[reportPrivateImportUsage]
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if isinstance(optimizer, SAM):
                # First pass climbs to the worst-case point; the second pass
                # recomputes the gradient there and applies the real update.
                optimizer.first_step(zero_grad=True)
                disable_running_stats(model)
                with torch.autocast(device_type=device.type, dtype=autocast):
                    loss_adv = loss_fn(model(imgs), labels)
                loss_adv.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
                optimizer.second_step(zero_grad=True)
                enable_running_stats(model)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
                optimizer.step()
            ema.update(model)

            preds = (logits > 0).float()
            n = imgs.size(0)
            loss_buf += loss.detach() * n
            correct_buf += (preds == labels).sum()
            gn_buf += grad_norm.detach()
            pending_n += n
            pending_steps += 1

            if global_step % constants.LOG_STEP == 0:
                window_loss = loss_buf.item() / max(1, pending_n)
                window_acc = int(correct_buf.item()) / max(1, pending_n)
                window_gn = gn_buf.item() / max(1, pending_steps)
                epoch_stats.merge(
                    loss_sum=loss_buf.item(),
                    correct=int(correct_buf.item()),
                    grad_norm_sum=gn_buf.item(),
                    n=pending_n,
                    steps=pending_steps,
                )
                loss_buf.zero_()
                correct_buf.zero_()
                gn_buf.zero_()
                pending_n = 0
                pending_steps = 0

                lr_now = optimizer.param_groups[-1]["lr"]
                pbar.set_postfix(
                    acc=f"{window_acc:.5f}",
                    loss=f"{window_loss:.5f}",
                    lr=f"{lr_now:.2e}",
                    gn=f"{window_gn:.5f}",
                )
                logger.info(
                    "epoch %d step %d/%d loss=%.5f acc=%.5f lr=%.2e grad_norm=%.5f",
                    epoch,
                    global_step,
                    total_steps,
                    window_loss,
                    window_acc,
                    lr_now,
                    window_gn,
                )
            global_step += 1

        if pending_steps > 0:
            epoch_stats.merge(
                loss_sum=loss_buf.item(),
                correct=int(correct_buf.item()),
                grad_norm_sum=gn_buf.item(),
                n=pending_n,
                steps=pending_steps,
            )

        dt = time.monotonic() - epoch_start
        logger.info(
            "epoch %d done in %.0fs, throughput %.1f img/s",
            epoch,
            dt,
            epoch_stats.seen / dt,
        )

        ckpt = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            ckpt,
            model,
            ema,
            optimizer,
            extras={
                "epoch": epoch,
                "schedule_total_steps": total_steps,
                "schedule_warmup_steps": warmup_steps,
                "pooling": pooling,
                "pretrained": pretrained,
                "stats": stats_to_dict(stats),
            },
        )
        logger.info("saved %s", ckpt)

    logger.info("done.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def log_coverage_curve(margins: list[float], corrects: list[bool]) -> None:
    """Log accuracy and coverage at fixed abstention thresholds."""
    margin_t = torch.tensor(margins, dtype=torch.float64)
    ok_t = torch.tensor(corrects, dtype=torch.bool)
    total = len(margins)
    logger.info("margin threshold -> coverage / accuracy:")
    for threshold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        kept = margin_t >= threshold
        n = int(kept.sum())
        if n == 0:
            continue
        acc = float(ok_t[kept].float().mean())
        logger.info(
            "  margin>=%.2f  coverage=%.3f  acc=%.4f  (n=%d)",
            threshold,
            n / total,
            acc,
            n,
        )


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, float, float, int]:
    """Return group CE, group4 accuracy, margin means, and group count.

    ``RotationPairDataset`` is ordered as base0-rot0..3,
    base1-rot0..3, and so on when ``shuffle=False``.
    """
    model.train(False)
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    # Carry logits across batch boundaries so batch_size need not be % 4.
    logit_buf: list[torch.Tensor] = []
    group4_correct = 0
    group4_total = 0
    margin_ok_sum = 0.0
    margin_bad_sum = 0.0
    margin_ok_n = 0
    margin_bad_n = 0
    margins: list[float] = []
    corrects: list[bool] = []

    def flush_groups() -> None:
        nonlocal group4_correct, group4_total, total_loss
        nonlocal margin_ok_sum, margin_bad_sum, margin_ok_n, margin_bad_n
        while len(logit_buf) >= N_ROTS:
            chunk = torch.stack(logit_buf[:N_ROTS])
            del logit_buf[:N_ROTS]
            # Training target: rot 0 is upright. Product pick: argmax logit.
            probs = chunk.softmax(dim=0)
            pred = int(probs.argmax().item())
            target = torch.zeros(1, dtype=torch.long, device=device)
            total_loss += float(loss_fn(chunk.unsqueeze(0), target).item())
            # top1 - top2 margin on four-view softmax probabilities.
            sorted_p, _ = probs.sort(descending=True)
            margin = float((sorted_p[0] - sorted_p[1]).item())
            ok = pred == 0
            margins.append(margin)
            corrects.append(ok)
            group4_total += 1
            if ok:
                group4_correct += 1
                margin_ok_sum += margin
                margin_ok_n += 1
            else:
                margin_bad_sum += margin
                margin_bad_n += 1

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs = batch[0].to(device, non_blocking=True)
        logits = model(imgs)

        for logit in logits.detach().float().reshape(-1):
            logit_buf.append(logit)
        flush_groups()

        if batch_idx % constants.LOG_STEP == 0:
            g4 = group4_correct / max(1, group4_total)
            logger.info(
                "validation batch %d/%d group4_acc=%.4f",
                batch_idx,
                n_batches,
                g4,
            )

    if logit_buf:
        logger.warning(
            "dropped %d trailing val samples not forming a full rotation group",
            len(logit_buf),
        )

    if margins:
        log_coverage_curve(margins, corrects)

    loss = total_loss / max(1, group4_total)
    group4_acc = group4_correct / max(1, group4_total)
    mean_margin_ok = margin_ok_sum / max(1, margin_ok_n)
    mean_margin_bad = margin_bad_sum / max(1, margin_bad_n)

    return (
        loss,
        group4_acc,
        mean_margin_ok,
        mean_margin_bad,
        group4_total,
    )


def validate_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    device: torch.device,
    which_weights: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    max_batches: int | None,
) -> None:
    """Load a checkpoint's weights and run validation in-process.

    which_weights: "ema" or "raw" - pick which set of weights from the
    checkpoint to load. Helpful for separating "EMA shadow is bad" from
    "validator code is bad".
    """
    payload, model, stats = load_checkpoint_model(
        checkpoint_path,
        device,
        weights=which_weights,
    )
    logger.info(
        "loaded %s (epoch=%s, ema_step=%s, weights=%s)",
        checkpoint_path,
        payload.get("epoch"),
        payload.get("ema_step"),
        which_weights,
    )

    val_ds, _ = build_dataset(
        split="val",
        datasets=build_datasets(data_root),
        device=device,
        num_workers=num_workers,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        stats=stats,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    logger.info(
        "val pairs: %d, device=%s, batch_size=%d, workers=%d, prefetch=%d, max_batches=%s",
        len(val_ds),
        device,
        batch_size,
        num_workers,
        prefetch_factor,
        max_batches,
    )

    t0 = time.monotonic()
    (
        loss,
        group4_acc,
        mean_margin_ok,
        mean_margin_bad,
        group4_n,
    ) = run_validation(
        model,
        val_loader,
        device=device,
        max_batches=max_batches,
    )
    dt = time.monotonic() - t0
    logger.info(
        "checkpoint %s -> group_ce=%.4f group4_acc=%.4f mean_margin_ok=%.4f mean_margin_bad=%.4f group4_n=%d in %.0fs",
        checkpoint_path.name,
        loss,
        group4_acc,
        mean_margin_ok,
        mean_margin_bad,
        group4_n,
        dt,
    )


# ---------------------------------------------------------------------------
# Deep Feature Reweighting (DFR)
# ---------------------------------------------------------------------------


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
        "stats": stats_to_dict(stats),
        "epoch": "dfr",
        "pooling": model.pooling,
        "dfr_source": str(source_checkpoint),
    }
    save_payload(path, payload)
    logger.info("saved DFR checkpoint %s", path)


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------


def compute_gradcam_batch(
    model: RotationModel,
    tensor: torch.Tensor,
) -> tuple[list[np.ndarray], list[float]]:
    """Run one forward+backward for a batched tensor and return per-sample
    GradCAM heatmaps (each in [0, 1]) plus per-sample upright logits."""
    target_layer: nn.Module = model.backbone.conv_head  # type: ignore[assignment]

    saved_features: list[torch.Tensor] = []
    saved_grads: list[torch.Tensor] = []

    def fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
        saved_features.append(out)

    def bwd(_m: nn.Module, _gin: object, gout: tuple[torch.Tensor, ...]) -> None:
        saved_grads.append(gout[0])

    fh = target_layer.register_forward_hook(fwd)
    bh = target_layer.register_full_backward_hook(bwd)  # type: ignore[arg-type]

    output = model(tensor)  # (B,) or (B, 1)
    model.zero_grad()
    output.sum().backward()  # per-sample grads are independent

    fh.remove()
    bh.remove()

    feat = saved_features[0]  # (B, C, H, W)
    grad = saved_grads[0]  # (B, C, H, W)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = nn.functional.relu((weights * feat).sum(dim=1))  # (B, H, W)
    cam_max = cam.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    cam_np = (cam / cam_max).cpu().detach().float().numpy()

    logits = output.detach().cpu().float().reshape(-1).tolist()
    return [cam_np[i] for i in range(cam_np.shape[0])], logits


def make_overlay(
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


def compute_guided_gradcam_batch(
    model: RotationModel,
    tensor: torch.Tensor,
    cams: list[np.ndarray],
) -> list[np.ndarray]:
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

    silu_modules = [m for m in model.modules() if isinstance(m, nn.SiLU)]
    original_inplace = [sm.inplace for sm in silu_modules]
    for sm in silu_modules:
        sm.inplace = False  # backward hooks reject view+inplace combo
    for i, sm in enumerate(silu_modules):
        handles.append(sm.register_forward_hook(make_fwd(i)))
        handles.append(sm.register_full_backward_hook(make_bwd(i)))  # type: ignore[arg-type]

    try:
        x = tensor.detach().clone().requires_grad_(True)
        output = model(x)
        model.zero_grad()
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


def visualise_grid(
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

    for idx, (angle, heat, prob, disp) in enumerate(zip(constants.ROTATIONS, heatmaps, probs, displays, strict=True)):
        img_overlay = make_overlay(disp, heat, colormap=colormap)
        ax = axes[idx // 2, idx % 2]
        marker = " *" if idx == pred else ""
        ax.imshow(img_overlay)
        ax.set_title(f"{angle}°{marker}  softmax={prob:.0%}", fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def compute_rotations(
    model: RotationModel,
    pil_img: Image.Image,
    preprocess: T.Compose,
    device: torch.device,
) -> tuple[list[Image.Image], list[float], list[np.ndarray], torch.Tensor]:
    # Display images: resize + center-crop, then PIL-rotate each view.
    crop = T.Compose(
        [
            T.Resize(constants.RESIZE_SIZE),
            T.CenterCrop(constants.INPUT_SIZE),
        ]
    )(pil_img)
    displays = [crop.rotate(a) if a != 0 else crop for a in constants.ROTATIONS]

    # Model input: transform once (resize/crop/to_tensor/normalize), then
    # rotate the tensor - same geometry as train/infer (RotationPairDataset,
    # infer.predict_rotations), not PIL-rotate-then-transform.
    base = preprocess(pil_img)
    views = [TF.rotate(base, a, expand=False) for a in constants.ROTATIONS]
    tensors = torch.stack(views).to(device)

    cams, logits = compute_gradcam_batch(model, tensors)
    probs = softmax_np(logits).tolist()
    return displays, probs, cams, tensors


def visualise(
    displays: list[Image.Image],
    probs: list[float],
    cams: list[np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    visualise_grid(displays, probs, cams, title, out_path, colormap="jet")


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


@dataclass
class InferResult:
    path: Path
    original: Image.Image
    corrected: Image.Image
    vote: RotationVote
    best_deg: int
    abstained: bool = False
    error: str | None = field(default=None)


def scan_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def prefetch_worker(
    paths: list[Path],
    model: RotationModel,
    device: torch.device,
    stats: NormStats,
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
                mean=stats.mean,
                std=stats.std,
            )
            best_deg = vote.best_deg
            abstained = abstain_margin is not None and vote.should_abstain(abstain_margin)
            # Low margin: keep original (no auto-rotate).
            if abstained:
                corrected = original
            else:
                corrected = original.rotate(best_deg, expand=True)
            result: InferResult = InferResult(
                path=path,
                original=original,
                corrected=corrected,
                vote=vote,
                best_deg=best_deg,
                abstained=abstained,
            )
        except Exception as exc:
            blank = Image.new("RGB", (224, 224), color=(30, 30, 30))
            result = InferResult(
                path=path,
                original=blank,
                corrected=blank,
                vote=RotationVote(
                    probs={0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0},
                    normalized_probs={0: 0.25, 90: 0.25, 180: 0.25, 270: 0.25},
                    best_deg=0,
                    margin=0.0,
                    angle=0.0,
                ),
                best_deg=0,
                error=str(exc),
            )

        out.put(result)
        bar.update(1)

    out.put(DONE)


def show_result(result: InferResult, index: int, total: int) -> None:
    """Open a blocking tkinter window for *result*; returns when it is closed."""
    import tkinter as tk

    from PIL import ImageTk

    prob_str = "  ".join(f"{deg}°: {prob:.3f}" for deg, prob in sorted(result.vote.normalized_probs.items()))
    if result.abstained:
        decision = f"ABSTAIN (margin={result.vote.margin:.3f})"
    else:
        decision = f"correction: {result.best_deg}° (margin={result.vote.margin:.3f})"
    title = f"[{index}/{total}]  {result.path.name}  ->  {decision}  |  {prob_str}"

    root = tk.Tk()
    root.title(title)
    root.resizable(True, True)
    root.bind("<q>", lambda _e: root.destroy())

    img = result.corrected.copy()
    img.thumbnail((1280, 900), Image.Resampling.LANCZOS)

    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=photo, bg="black")
    img_label.pack(fill=tk.BOTH, expand=True)

    if result.error:
        err_label = tk.Label(
            root,
            text=f"⚠  {result.error}",
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


def build_onnx_meta(
    payload: dict,
    stats: NormStats,
    checkpoint: Path,
    weights: str,
) -> dict:
    return {
        "checkpoint": str(checkpoint),
        "epoch": payload.get("epoch"),
        "weights": weights,
        "pooling": payload.get("pooling", "avg"),
        "pretrained": payload.get("pretrained"),
        **stats_to_dict(stats),
        "input_size": constants.INPUT_SIZE,
        "resize_size": constants.RESIZE_SIZE,
        "rotations": list(constants.ROTATIONS),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
    }


def export_onnx(
    model: RotationModel,
    out_path: Path,
    device: torch.device,
    opset: int,
    dynamo: bool,
    verify: bool,
) -> None:
    """Trace ``model`` to ONNX with a dynamic batch axis."""
    size = constants.INPUT_SIZE
    dummy = torch.randn(1, 3, size, size, device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        INPUT_NAME: {0: "batch"},
        OUTPUT_NAME: {0: "batch"},
    }

    export_kwargs: dict = {
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": dynamic_axes,
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
        verify_export(model, out_path, dummy, device)


def verify_export(
    model: RotationModel,
    out_path: Path,
    dummy: torch.Tensor,
    device: torch.device,
) -> None:
    """Compare torch vs onnxruntime on the dummy batch (optional smoke check)."""
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


def write_onnx_meta(meta: dict, out_path: Path, opset: int, dynamo: bool) -> Path:
    meta_path = out_path.with_suffix(".json")
    payload = {
        **meta,
        "onnx": str(out_path),
        "opset": opset,
        "dynamo": dynamo,
    }
    meta_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote metadata %s", meta_path)
    return meta_path


# ---------------------------------------------------------------------------
# CLI subcommand bodies
# ---------------------------------------------------------------------------


def run_train(args: argparse.Namespace) -> None:
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
    datasets = build_datasets(args.data_root)
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

    train(
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
    validate_checkpoint(
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
    _, model, stats = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    print("Model ready.\n")

    root_dir = Path(args.dir)
    print(f"Scanning {root_dir}...")
    paths = scan_images(root_dir)
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
        if item is DONE:
            break

        assert isinstance(item, InferResult)
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

        show_result(item, shown, len(paths))

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
    _, model, stats = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    preprocess = build_eval_transform(mean=stats.mean, std=stats.std)

    args.out_dir.mkdir(exist_ok=True)

    for img_path in images:
        pil = open_rgb(img_path)
        displays, probs, cams, tensors = compute_rotations(model, pil, preprocess, device)
        angle = circular_mean_angle(probs)

        visualise(
            displays,
            probs,
            cams,
            f"{img_path.name} - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_gradcam.png",
        )
        print(f"{img_path.name} - predicted angle {angle:.1f}°")

        guided_cams = compute_guided_gradcam_batch(model, tensors, cams)
        visualise_grid(
            displays,
            probs,
            guided_cams,
            f"{img_path.name} - guided GradCAM - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_guidedcam.png",
            colormap="inferno",
        )


def run_dfr(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    payload, model, stats = load_checkpoint_model(
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


def run_export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    payload, model, stats = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    meta = build_onnx_meta(
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

    export_onnx(
        model=model,
        out_path=out_path,
        device=device,
        opset=args.opset,
        dynamo=args.dynamo,
        verify=args.verify,
    )
    write_onnx_meta(meta, out_path, opset=args.opset, dynamo=args.dynamo)
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
