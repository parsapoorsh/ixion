"""Dataset normalization statistics with JSON caching."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import sys
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

import constants
from utils.image_io import open_rgb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

Stats = tuple[float, float, float]
MeanStd = tuple[Stats, Stats]


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
        try:
            image = open_rgb(
                path,
                preview_size=(constants.RESIZE_SIZE, constants.RESIZE_SIZE),
            )
            image = image.resize(
                (constants.RESIZE_SIZE, constants.RESIZE_SIZE),
                Image.Resampling.BILINEAR,
            )
            offset = (constants.RESIZE_SIZE - constants.INPUT_SIZE) // 2
            image = image.crop(
                (
                    offset,
                    offset,
                    offset + constants.INPUT_SIZE,
                    offset + constants.INPUT_SIZE,
                )
            )
            array = np.array(image, dtype=np.uint8, copy=True)
            image.close()
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"unreadable image {path}: {exc}") from exc

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


def iter_stats_results(
    chunks: Iterable[list[Path]],
    worker_count: int,
    prefetch_factor: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, int, int]]:
    if worker_count <= 1:
        yield from map(compute_stats_chunk, chunks)
        return

    max_pending = max(
        worker_count,
        worker_count * max(1, prefetch_factor),
    )
    context: BaseContext
    if sys.platform != "win32" and "fork" in mp.get_all_start_methods():
        context = mp.get_context("fork")
    else:
        context = mp.get_context()

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


def compute_stats(
    files: Sequence[Path],
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    sample_indices: Sequence[int] | None = None,
    cache_path: Path | None = None,
    invalidate_cache: bool = False,
    fingerprint: str | None = None,
) -> MeanStd:
    if cache_path is not None and cache_path.exists() and not invalidate_cache:
        cached = json.loads(cache_path.read_text())
        if fingerprint is None or cached.get("fingerprint") == fingerprint:
            return (
                cast(tuple[float, float, float], tuple(cached["mean"])),
                cast(tuple[float, float, float], tuple(cached["std"])),
            )
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

    if sample_indices is None:
        chunks: Iterable[list[Path]] = (
            list(files[start:start + chunk_size])
            for start in range(0, len(files), chunk_size)
        )
    else:
        chunks = (
            [files[idx] for idx in sample_indices[start:start + chunk_size]]
            for start in range(0, len(sample_indices), chunk_size)
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
    mean_arr = sum_channels / (total_pixels * scale)
    variance = (sum_sq_channels / (total_pixels * scale * scale)) - mean_arr * mean_arr
    std_arr = np.sqrt(np.clip(variance, 0.0, None))
    mean_std: MeanStd = (
        cast(tuple[float, float, float], tuple(mean_arr.tolist())),
        cast(tuple[float, float, float], tuple(std_arr.tolist())),
    )
    if cache_path is not None:
        mean, std = mean_std
        payload: dict = {"mean": list(mean), "std": list(std)}
        if fingerprint is not None:
            payload["fingerprint"] = fingerprint
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
    return mean_std
