"""Per-class accuracy + loss over a validation loader.

Reusable across train.py (per-epoch eval) and any standalone evaluation
script.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

import config


@dataclass
class ValidationResult:
    loss: float
    acc: float
    per_class: list[float]


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> ValidationResult:
    model.train(False)
    correct = torch.zeros(config.NUM_CLASSES, device=device)
    seen = torch.zeros(config.NUM_CLASSES, device=device)
    total_loss = 0.0
    total_n = 0
    loss_fn = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        for c in range(config.NUM_CLASSES):
            mask = labels == c
            seen[c] += mask.sum()
            correct[c] += (preds[mask] == c).sum()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)
    return ValidationResult(
        loss=total_loss / max(1, total_n),
        acc=(correct.sum() / seen.sum().clamp(min=1)).item(),
        per_class=(correct / seen.clamp(min=1)).cpu().tolist(),
    )
