"""Per-dataset channel mean/std — computed once on GPU, cached to disk.

When a DatasetSpec leaves mean/std as None, we sample
``config.STATS_SAMPLE_FRACTION`` of the train split (sorted by path so
HDD-backed datasets read in on-disk locality order), decode in parallel
through a DataLoader using the shared ``build_eval_preprocess`` pipeline,
reduce per-batch sum and sum-of-squares on ``config.DEVICE`` in float32
(consumer GPUs are ~1/64 speed for float64), and accumulate on the CPU in
float64. Result lands in ``cache/stats_<name>_<resize>x<input>.json``. A spec that *does*
carry explicit mean/std (e.g. pinned to ImageNet) bypasses all of this
and is returned as-is. Runnable as a CLI to pre-warm the cache before a
training run.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
import constants
from builders import build_eval_preprocess
from utils.image_io import open_rgb
from utils.index import index_files
from utils.logsetup import setup_logging

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from datasets import DatasetSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def clear_cache() -> None:
    """Delete every cached stats_*.json under the cache directory."""
    if not constants.INDEX_CACHE_DIR.exists():
        return
    for f in constants.INDEX_CACHE_DIR.glob("stats_*.json"):
        f.unlink()
        logger.info("removed cached stats %s", f.name)


def get_stats(
    spec: DatasetSpec,
    input_size: int,
    resize_size: int,
    num_workers: int = constants.NUM_WORKERS,
) -> ChannelStats:
    """Return (mean, std) for the spec, computing+caching on first miss.

    Stats depend on the actual pixels averaged, which depend on the
    Resize→CenterCrop pipeline. The cache filename includes both sizes so
    a different preprocess gets its own cache rather than silently
    reusing stale numbers.

    If the spec already carries explicit mean/std we return those directly —
    the cache and the train-set scan are skipped entirely.
    """
    if spec.mean is not None and spec.std is not None:
        return ChannelStats(mean=spec.mean, std=spec.std)

    constants.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = constants.INDEX_CACHE_DIR / f"stats_{spec.name}_{resize_size}x{input_size}.json"
    if cache.exists():
        with cache.open() as f:
            payload = json.load(f)
        return ChannelStats(
            mean=tuple(payload["mean"]),
            std=tuple(payload["std"]),
        )

    files = index_files(f"train_{spec.name}", [spec.train])
    target_n = max(1, int(len(files) * config.STATS_SAMPLE_FRACTION))
    sample = sample_files(files, target_n, config.STATS_SAMPLE_SEED)
    logger.info(
        "computing %s stats over %d sampled images (%.1f%% of %d) on %s at %dx%d...",
        spec.name, len(sample), 100 * config.STATS_SAMPLE_FRACTION,
        len(files), config.DEVICE, resize_size, input_size,
    )

    t0 = time.monotonic()
    stats = compute_stats(
        sample,
        input_size=input_size,
        resize_size=resize_size,
        num_workers=num_workers,
    )
    dt = time.monotonic() - t0
    logger.info(
        "%s stats: mean=%s std=%s (%.1fs, %.1f img/s)",
        spec.name, stats.mean, stats.std, dt, len(sample) / dt,
    )

    with cache.open("w") as f:
        json.dump(
            {"mean": list(stats.mean), "std": list(stats.std), "n_sampled": len(sample)},
            f,
        )
    return stats


def sample_files(
    files: Sequence[Path],
    n: int,
    seed: int,
) -> list[Path]:
    """Deterministic uniform sample, returned in *path-sorted* order.

    Sorting after sampling keeps the same set of files but reads them in
    on-disk locality order — without it, scattered seeks across a 1.2M
    file tree on a spinning disk stall workers once the page cache runs
    out (~700 img/s → ~150 img/s in the wild).
    """
    if len(files) <= n:
        sampled = list(files)
    else:
        rng = random.Random(seed)
        sampled = rng.sample(files, n)
    sampled.sort()
    return sampled


class StatsDataset(Dataset):
    """Decode → resize → centre-crop → tensorise for the stats DataLoader.

    A fixed (input_size, input_size) crop is used so every sample has the
    same shape and the loader can stack them into a real batch — variable
    sizes would force batch_size=1 and kill the parallelism. Decode
    failures return ``None`` and are filtered by ``stats_collate``.
    """

    def __init__(
        self,
        files: Sequence[Path],
        input_size: int,
        resize_size: int,
    ) -> None:
        self.files = files
        self.transform = build_eval_preprocess(
            input_size=input_size,
            resize_size=resize_size,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        img = open_rgb(self.files[idx])
        return self.transform(img)


def stats_collate(batch: list[torch.Tensor | None]) -> torch.Tensor | None:
    items = [x for x in batch if x is not None]
    if not items:
        return None
    return torch.stack(items)


def compute_stats(
    files: Sequence[Path],
    input_size: int,
    resize_size: int,
    num_workers: int = constants.NUM_WORKERS,
    batch_size: int = 64,
) -> ChannelStats:
    """Per-channel mean/std on a sample, decoded in workers and reduced on GPU.

    Per batch we compute sum and sum-of-squares on ``config.DEVICE`` in
    float32 (consumer GPUs are slow at float64), then accumulate those small
    tensors on the CPU in float64. Per-batch precision stays tight because
    each partial is bounded by the batch's pixel count.
    """
    loader = DataLoader(
        StatsDataset(files, input_size=input_size, resize_size=resize_size),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=stats_collate,
        pin_memory=config.DEVICE.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_sq_c = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    with tqdm(total=len(files), desc="stats", unit="img") as pbar:
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(config.DEVICE, non_blocking=True)

            b, _, h, w = batch.shape
            sum_c += batch.sum(dim=(0, 2, 3)).double().cpu()
            sum_sq_c += (batch * batch).sum(dim=(0, 2, 3)).double().cpu()
            total_pixels += b * h * w
            pbar.update(b)

    if total_pixels == 0:
        raise RuntimeError("no images decoded; cannot compute stats")
    mean = sum_c / total_pixels
    variance = (sum_sq_c / total_pixels) - mean * mean
    std = variance.clamp_min(0).sqrt()
    return ChannelStats(
        mean=tuple(mean.tolist()),
        std=tuple(std.tolist()),
    )


def main() -> None:
    setup_logging()
    defaults = config.DEFAULT_BACKBONE.train_defaults

    p = argparse.ArgumentParser(
        description="Compute per-dataset channel mean/std and cache the result. "
                    "By default runs every dataset in config.DATASETS at the "
                    "configured backbone's preprocess sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=constants.NUM_WORKERS,
        help="DataLoader worker processes used for image decoding",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=defaults.input_size,
        help="crop size used as the model input (drives the stats cache key)",
    )
    p.add_argument(
        "--resize-size",
        type=int,
        default=defaults.resize_size,
        help="pre-crop resize, applied before CenterCrop",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached stats JSONs before recomputing",
    )
    args = p.parse_args()

    if args.invalidate_cache:
        clear_cache()

    for spec in config.DATASETS:
        result = get_stats(
            spec,
            input_size=args.input_size,
            resize_size=args.resize_size,
            num_workers=args.num_workers,
        )
        logger.info(
            "%s: mean=%s std=%s",
            spec.name, result.mean, result.std,
        )


if __name__ == "__main__":
    main()
