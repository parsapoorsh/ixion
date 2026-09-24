"""Per-channel mean/std over a file list.

Decoded in parallel through a DataLoader with a fixed Resize/CenterCrop/
ToTensor pipeline so every batch has matching shapes and pixel values
are in [0, 1]. Sum and sum-of-squares reduce on device in float32; we
accumulate on the CPU in float64 to keep per-batch precision bounded.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils.index import cache_fingerprint

if TYPE_CHECKING:
    pass


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
    payload = {"mean": list(stats.mean), "std": list(stats.std)}
    if fingerprint is not None:
        payload["fingerprint"] = fingerprint
    path.write_text(json.dumps(payload))


def compute_stats(
    dataset: Dataset,
    device: torch.device,
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    cache_path: Path | None = None,
    invalidate_cache: bool = False,
    fingerprint: str | None = None,
) -> NormStats:
    if cache_path is not None and cache_path.exists() and not invalidate_cache:
        cached = load_stats_cache(cache_path, fingerprint)
        if cached is not None:
            return cached
        logging.info("stats cache %s does not match configuration; recomputing", cache_path)

    stats_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_sq_c = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    pbar = tqdm(
        total=len(dataset),
        desc="stats",
        unit="img",
        file=sys.stdout,
    )
    for batch in stats_loader:
        batch = batch.to(device, non_blocking=True)
        b, _, h, w = batch.shape
        sum_c += batch.sum(dim=(0, 2, 3)).double().cpu()
        sum_sq_c += (batch * batch).sum(dim=(0, 2, 3)).double().cpu()
        total_pixels += b * h * w
        pbar.update(b)
    pbar.close()

    if total_pixels == 0:
        raise RuntimeError("no images decoded; cannot compute stats")
    mean = sum_c / total_pixels
    variance = (sum_sq_c / total_pixels) - mean * mean
    std = variance.clamp_min(0).sqrt()
    stats = NormStats(
        mean=tuple(mean.tolist()),  # type: ignore[arg-type]
        std=tuple(std.tolist()),  # type: ignore[arg-type]
    )
    if cache_path is not None:
        save_stats_cache(cache_path, stats, fingerprint)
    return stats
