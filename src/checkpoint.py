"""Checkpoint loading and saving."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn

import constants
from model import SAM, ModelEMA, RotationModel
from stats import MeanStd

logger = logging.getLogger(__name__)

# Checkpoint keys holding the two weight sets: raw training weights and the
# EMA shadow.
WeightName = Literal["ema", "raw"]

def load_checkpoint_model(
    checkpoint: Path,
    device: torch.device,
    weights: WeightName = "ema",
    strict_stats: bool = False,
) -> tuple[dict, RotationModel, MeanStd]:
    """Load one weight set and the normalization stats from a checkpoint.

    ``weights`` picks the EMA shadow ("ema", production) or the raw training
    weights ("raw"). Returned model is in eval mode.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if weights not in payload:
        raise KeyError(f"checkpoint missing '{weights}' weights; keys={list(payload)}")

    raw_stats = payload.get("stats")
    if not raw_stats:
        if strict_stats:
            raise KeyError("checkpoint has no 'stats'; normalization cannot be inferred")
        cache_file = constants.INDEX_CACHE_DIR / "stats_train.json"
        if not cache_file.exists():
            raise RuntimeError(f"stats cache not found: {cache_file}")
        raw_stats = json.loads(cache_file.read_text())
        logger.info("loaded stats from cache file %s", cache_file)
    mean_std: MeanStd = (
        cast(tuple[float, float, float], tuple(raw_stats["mean"])),
        cast(tuple[float, float, float], tuple(raw_stats["std"])),
    )

    logger.info(
        "loaded %s (epoch=%s, ema_step=%s, weights=%s, pooling=%s)",
        checkpoint,
        payload.get("epoch"),
        payload.get("ema_step"),
        weights,
        payload.get("pooling", "avg"),
    )
    model = RotationModel(
        pretrained=False,
        pooling=payload.get("pooling", "avg"),
    ).to(device)
    model.load_state_dict(payload[weights])
    model.eval()
    return payload, model, mean_std


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: SAM,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    extras: dict | None = None,
) -> None:
    # torch.compile() wraps the module and prefixes state_dict keys with
    # "_orig_mod." - unwrap so checkpoints stay portable across compile.
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "raw": raw_model.state_dict(),
        "ema": ema.state_dict(),
        "stats": {"mean": list(mean), "std": list(std)},
        "ema_step": ema.step,
        # The real state lives in SAM's wrapped AdamW.
        "optimizer": optimizer.base_optimizer.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        **(extras or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
