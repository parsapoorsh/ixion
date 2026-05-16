"""Checkpoint save/load helpers.

Both train.py and calibrate.py read/write checkpoints. Centralizing the
{"model": ..., "ema": ..., ...} dict layout here means the format only needs
to change in one place.
"""

from pathlib import Path

import torch
from torch import nn

from ema import ModelEMA


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    extras: dict | None = None,
) -> None:
    payload: dict = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
    }
    if extras:
        payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_state_dict_from_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:
    """Load weights from a checkpoint, preferring EMA weights if present."""
    state = torch.load(path, map_location=device)
    return state.get("ema") or state.get("model") or state
