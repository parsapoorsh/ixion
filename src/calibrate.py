"""Calibration utilities: gather logits, fit temperature, measure ECE.

CLI: fit temperature scaling on the validation set and save a calibrated checkpoint.
"""

import logging
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
import constants
import stats
from model import RotationModel
from builders import build_val_dataset
from utils.index import clear_cache
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)


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


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(
        description="Fit temperature scaling on the validation set "
                    "and write a calibrated checkpoint."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=constants.CHECKPOINT_DIR / "best.pt",
        help="path to a trained checkpoint (defaults to checkpoints/best.pt)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="batch size used while gathering logits",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=config.NUM_WORKERS,
        help="DataLoader worker processes",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached file-index and per-dataset stats JSONs before running",
    )
    args = p.parse_args()

    if args.invalidate_cache:
        clear_cache()
        stats.clear_cache()

    model = RotationModel(pretrained=False).to(config.DEVICE)
    state = torch.load(args.checkpoint, map_location=config.DEVICE)
    state_dict = state.get("ema") or state.get("model") or state
    model.load_state_dict(state_dict)
    model.train(False)

    val_ds = build_val_dataset(config.DATASETS)
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info("gathering logits...")
    logits, labels = gather_logits(model, loader, config.DEVICE)

    ece_before = expected_calibration_error(logits.softmax(dim=-1), labels)
    T = fit_temperature(logits, labels)
    ece_after = expected_calibration_error((logits / T).softmax(dim=-1), labels)
    logger.info(
        "T = %.4f  ECE before = %.4f  ECE after = %.4f",
        T, ece_before, ece_after,
    )

    model.temperature.fill_(T)
    out = args.checkpoint.with_name(args.checkpoint.stem + "_calibrated.pt")
    torch.save({"model": model.state_dict(), "temperature": T}, out)
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
