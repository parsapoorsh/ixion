"""Per-dataset channel mean/std — computed once, cached to disk.

When a DatasetSpec leaves mean/std as None, we sample a fixed number of
training images, accumulate per-channel sums and sums-of-squares in float64,
and cache the result as JSON under cache/. A spec that *does* carry
explicit mean/std (e.g. pinned to ImageNet) bypasses all of this and is
returned as-is.
"""

import json
import logging
import random
import time
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import functional as TF

import config
from index import index_files
from spec import DatasetSpec

logger = logging.getLogger(__name__)

SAMPLE_SIZE = 5000
SAMPLE_SEED = 0


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


def get_stats(spec: DatasetSpec) -> ChannelStats:
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
    sample = sample_files(files, SAMPLE_SIZE, SAMPLE_SEED)
    logger.info("computing %s stats over %d sampled images...", spec.name, len(sample))

    t0 = time.monotonic()
    stats = compute_stats(sample)
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


def sample_files(files, n, seed):
    if len(files) <= n:
        return list(files)
    rng = random.Random(seed)
    return rng.sample(list(files), n)


def compute_stats(files) -> ChannelStats:
    """Per-channel mean/std on a sample, vectorised per image in float64.

    Each image is resized to INPUT_SIZE so the cost is bounded and consistent
    with what training will see. Decode failures are skipped with a warning
    instead of aborting the run.
    """
    total_pixels = 0
    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_sq_c = torch.zeros(3, dtype=torch.float64)

    for path in files:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning("stats: skipping %s (%s)", path, exc)
            continue
        img = TF.resize(img, config.INPUT_SIZE)
        tensor = TF.to_tensor(img).to(torch.float64)  # [3, H, W] in [0, 1]
        flat = tensor.reshape(3, -1)
        sum_c += flat.sum(dim=1)
        sum_sq_c += (flat * flat).sum(dim=1)
        total_pixels += flat.shape[1]

    if total_pixels == 0:
        raise RuntimeError("no images decoded; cannot compute stats")
    mean = sum_c / total_pixels
    variance = (sum_sq_c / total_pixels) - mean * mean
    std = variance.clamp_min(0).sqrt()
    return ChannelStats(
        mean=tuple(mean.tolist()),
        std=tuple(std.tolist()),
    )
