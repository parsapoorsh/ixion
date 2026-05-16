"""Calibration utilities: gather logits, fit temperature, measure ECE.

Pure functions, no I/O or CLI. The CLI entry point lives in calibrate.py.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def gather_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over the loader, returning float32 logits and labels on CPU."""
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(imgs)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
) -> float:
    """Minimize NLL(logits / T, labels) over T using LBFGS.

    T is parameterized as exp(log_t) to keep it positive without bounded
    optimization tricks.
    """
    log_t = torch.zeros(1, requires_grad=True)
    optim = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    loss_fn = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optim.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optim.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Standard ECE with equal-width bins on top-1 confidence."""
    confidences, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = probs.size(0)
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if in_bin.any():
            acc_bin = correct[in_bin].mean().item()
            conf_bin = confidences[in_bin].mean().item()
            ece += (in_bin.float().sum().item() / n) * abs(acc_bin - conf_bin)
    return ece
