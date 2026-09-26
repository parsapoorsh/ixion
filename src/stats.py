"""Per-channel mean/std over a file list.

Stats are decoded and reduced inside worker processes. Each worker returns only
per-channel sums instead of full image tensors, avoiding the large
multiprocessing transfer that made the old DataLoader path crawl.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import sys
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import UnidentifiedImageError
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

import constants
from utils.image_io import open_rgb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def load_stats_cache(path: Path, fingerprint: str | None = None) -> NormStats | None:
    if not path.exists():
        raise FileNotFoundError(f"stats cache not found: {path}")
    data = json.loads(path.read_text())
    if fingerprint is not None and data.get("fingerprint") != fingerprint:
        return None
    return NormStats(mean=tuple(data["mean"]), std=tuple(data["std"]))  # type: ignore[arg-type]


def save_stats_cache(
    path: Path,
    stats: NormStats,
    fingerprint: str | None = None,
) -> None:
    payload: dict[str, list[float] | str] = {
        "mean": list(stats.mean),
        "std": list(stats.std),
    }
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
