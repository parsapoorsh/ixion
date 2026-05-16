"""CLI: fit temperature scaling on the validation set and save a calibrated checkpoint."""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from builders import build_val_dataset
from calibration import expected_calibration_error, fit_temperature, gather_logits
from index import clear_cache
from logsetup import setup_logging
from model import RotationModel

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(
        description="Fit temperature scaling on the validation set "
                    "and write a calibrated checkpoint."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINT_DIR / "best.pt",
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
        default=8,
        help="DataLoader worker processes",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached file-index JSONs before indexing",
    )
    args = p.parse_args()

    if args.invalidate_cache:
        clear_cache()

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
