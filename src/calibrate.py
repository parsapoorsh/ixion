"""CLI: fit temperature scaling on the validation set and save a calibrated checkpoint."""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from builders import build_val_dataset
from calibration import expected_calibration_error, fit_temperature, gather_logits
from checkpoint import load_state_dict_from_checkpoint
from logsetup import setup_logging
from model import RotationModel

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
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
    model = RotationModel(pretrained=False).to(device)
    model.load_state_dict(load_state_dict_from_checkpoint(args.checkpoint, device))
    model.train(False)

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
