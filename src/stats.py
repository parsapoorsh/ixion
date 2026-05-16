"""Per-dataset channel mean/std — computed once, cached to disk.

When a DatasetSpec leaves mean/std as None, we sample a fixed number of
training images, accumulate per-channel sums and sums-of-squares in float64,
and cache the result as JSON under cache/. A spec that *does* carry
explicit mean/std (e.g. pinned to ImageNet) bypasses all of this and is
returned as-is.
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
from torchvision import transforms as T
from tqdm import tqdm

import config
from utils.image_io import open_rgb
from utils.index import index_files
from utils.logsetup import setup_logging

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from spec import DatasetSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def clear_cache() -> None:
    """Delete every cached stats_*.json under the cache directory."""
    if not config.INDEX_CACHE_DIR.exists():
        return
    for f in config.INDEX_CACHE_DIR.glob("stats_*.json"):
        f.unlink()
        logger.info("removed cached stats %s", f.name)


def get_stats(
    spec: DatasetSpec,
    num_workers: int = 8,
) -> ChannelStats:
    """Return (mean, std) for the spec, computing+caching on first miss.

    If the spec already carries explicit mean/std we return those directly —
    the cache and the train-set scan are skipped entirely.
    """
    if spec.mean is not None and spec.std is not None:
        return ChannelStats(mean=spec.mean, std=spec.std)

    config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.INDEX_CACHE_DIR / f"stats_{spec.name}.json"
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
        "computing %s stats over %d sampled images (%.1f%% of %d) on %s...",
        spec.name, len(sample), 100 * config.STATS_SAMPLE_FRACTION,
        len(files), config.DEVICE,
    )

    t0 = time.monotonic()
    stats = compute_stats(sample, num_workers=num_workers)
    dt = time.monotonic() - t0
    logger.info(
        "%s stats: mean=%s std=%s (%.1fs)",
        spec.name, stats.mean, stats.std, dt,
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
    if len(files) <= n:
        return list(files)
    rng = random.Random(seed)
    return rng.sample(files, n)


class StatsDataset(Dataset):
    """Decode → resize → centre-crop → tensorise for the stats DataLoader.

    A fixed (INPUT_SIZE, INPUT_SIZE) crop is used so every sample has the
    same shape and the loader can stack them into a real batch — variable
    sizes would force batch_size=1 and kill the parallelism. Decode
    failures return ``None`` and are filtered by ``stats_collate``.
    """

    def __init__(self, files: Sequence[Path]) -> None:
        self.files = files
        self.transform = T.Compose([
            T.Resize(config.RESIZE_SIZE),
            T.CenterCrop(config.INPUT_SIZE),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        try:
            img = open_rgb(self.files[idx])
        except Exception as exc:
            logger.warning("stats: skipping %s (%s)", self.files[idx], exc)
            return None
        return self.transform(img)


def stats_collate(batch: list[torch.Tensor | None]) -> torch.Tensor | None:
    items = [x for x in batch if x is not None]
    if not items:
        return None
    return torch.stack(items)


def compute_stats(
    files: Sequence[Path],
    num_workers: int = 8,
    batch_size: int = 64,
) -> ChannelStats:
    """Per-channel mean/std on a sample, decoded in workers and reduced on GPU.

    Per batch we compute sum and sum-of-squares on ``config.DEVICE`` in
    float32 (consumer GPUs are slow at float64), then accumulate those small
    tensors on the CPU in float64. Per-batch precision stays tight because
    each partial is bounded by the batch's pixel count.
    """
    loader = DataLoader(
        StatsDataset(files),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=stats_collate,
        pin_memory=config.DEVICE.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_sq_c = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for batch in tqdm(loader, desc="stats", unit="batch"):
        if batch is None:
            continue
        batch = batch.to(config.DEVICE, non_blocking=True)
        b, _, h, w = batch.shape
        sum_c += batch.sum(dim=(0, 2, 3)).double().cpu()
        sum_sq_c += (batch * batch).sum(dim=(0, 2, 3)).double().cpu()
        total_pixels += b * h * w

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
    p = argparse.ArgumentParser(
        description="Compute per-dataset channel mean/std and cache the result. "
                    "By default runs every dataset in config.DATASETS.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="restrict to a single dataset by name (matches DatasetSpec.name)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker processes used for image decoding",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached stats JSONs before recomputing",
    )
    args = p.parse_args()

    if args.invalidate_cache:
        clear_cache()

    if args.dataset is None:
        specs = list(config.DATASETS)
    else:
        specs = [s for s in config.DATASETS if s.name == args.dataset]
        if not specs:
            available = ", ".join(s.name for s in config.DATASETS)
            raise SystemExit(f"no dataset named {args.dataset!r} (available: {available})")

    for spec in specs:
        stats = get_stats(spec, num_workers=args.num_workers)
        logger.info(
            "%s: mean=%s std=%s",
            spec.name, stats.mean, stats.std,
        )


if __name__ == "__main__":
    main()
