"""Post-hoc temperature scaling.

Fits a single scalar T on a held-out validation set by minimizing NLL of
logits/T against the true rotation label. Writes T back into model.temperature.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from datasets import build_val_dataset
from infer import load_model_for_inference

log = logging.getLogger(__name__)


@torch.no_grad()
def gather_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(imgs)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
) -> float:
    """Minimize NLL(logits / T, labels) over T using LBFGS."""
    log_t = torch.zeros(1, requires_grad=True)  # parameterize as log T for positivity
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINT_DIR / "best.pt",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda")
    model = load_model_for_inference(args.checkpoint, device)
    val_ds = build_val_dataset(config.DATASETS)
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    log.info("gathering logits...")
    logits, labels = gather_logits(model, loader, device)

    ece_before = expected_calibration_error(logits.softmax(dim=-1), labels)
    T = fit_temperature(logits, labels)
    ece_after = expected_calibration_error((logits / T).softmax(dim=-1), labels)
    log.info(
        "T = %.4f  ECE before = %.4f  ECE after = %.4f",
        T, ece_before, ece_after,
    )

    model.temperature.fill_(T)
    out = args.checkpoint.with_name(args.checkpoint.stem + "_calibrated.pt")
    torch.save({"model": model.state_dict(), "temperature": T}, out)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
