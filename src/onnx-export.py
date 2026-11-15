#!/usr/bin/env python3
"""Export a rotation-detector checkpoint to ONNX.

Loads EMA (or raw) weights from an epoch checkpoint, traces the binary
upright head to ONNX, and writes a sibling ``.json`` metadata file with
the mean/std and geometry the runner needs. Product inference still scores
four 90° views outside the graph - the ONNX model is a single forward
pass over ``(B, 3, H, W)`` -> logits ``(B,)``.

Usage (from project root):
    venv/bin/python src/onnx-export.py \\
        --checkpoint checkpoints/epoch_010.pt \\
        --out checkpoints/epoch_010.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

import constants
from model.rotation import RotationModel
from stats import NormStats, load_stats_cache
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)

INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def load_model(
    checkpoint: Path,
    device: torch.device,
    weights: str,
) -> tuple[RotationModel, NormStats, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    state_key = {"ema": "ema", "raw": "model"}[weights]
    if state_key not in payload:
        raise KeyError(f"checkpoint missing '{state_key}' key; available={list(payload)}")

    raw_stats = payload.get("stats")
    if raw_stats:
        stats = NormStats(
            mean=tuple(raw_stats["mean"]),  # type: ignore[arg-type]
            std=tuple(raw_stats["std"]),  # type: ignore[arg-type]
        )
    else:
        cached = load_stats_cache(constants.INDEX_CACHE_DIR / "stats_train.json")
        if cached is None:
            raise RuntimeError("stats cache has no matching fingerprint")
        stats = cached
        logger.info("loaded stats from cache (checkpoint had none)")

    pooling = payload.get("pooling", "avg")
    model = RotationModel(
        pretrained=False,
        pooling=pooling,
    )
    model.load_state_dict(payload[state_key])
    model.to(device)
    model.eval()

    meta = {
        "checkpoint": str(checkpoint),
        "epoch": payload.get("epoch"),
        "weights": weights,
        "pooling": pooling,
        "pretrained": payload.get("pretrained"),
        "mean": list(stats.mean),
        "std": list(stats.std),
        "input_size": constants.INPUT_SIZE,
        "resize_size": constants.RESIZE_SIZE,
        "rotations": list(constants.ROTATIONS),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
    }
    return model, stats, meta


def export_onnx(
    model: RotationModel,
    out_path: Path,
    device: torch.device,
    opset: int,
    dynamo: bool,
    verify: bool,
) -> None:
    """Trace ``model`` to ONNX with a dynamic batch axis."""
    size = constants.INPUT_SIZE
    dummy = torch.randn(1, 3, size, size, device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        INPUT_NAME: {0: "batch"},
        OUTPUT_NAME: {0: "batch"},
    }

    export_kwargs: dict = {
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": dynamic_axes,
        "dynamo": dynamo,
    }

    logger.info(
        "exporting ONNX -> %s (opset=%d, dynamo=%s, device=%s)",
        out_path,
        opset,
        dynamo,
        device,
    )

    # Legacy exporter writes ``f`` directly and returns None. The dynamo
    # exporter returns an ONNXProgram that must be saved when f is None.
    try:
        result = torch.onnx.export(
            model,
            (dummy,),
            str(out_path) if not dynamo else None,
            **export_kwargs,
        )
    except ModuleNotFoundError as exc:
        if dynamo and "onnxscript" in str(exc):
            raise RuntimeError(
                "dynamo export needs the onnxscript package "
                "(pip install onnxscript); or omit --dynamo for the legacy exporter"
            ) from exc
        raise

    if result is not None:
        result.save(str(out_path))

    if not out_path.is_file():
        raise RuntimeError(f"ONNX export produced no file at {out_path}")

    if verify:
        _verify_export(model, out_path, dummy, device)


def _verify_export(
    model: RotationModel,
    out_path: Path,
    dummy: torch.Tensor,
    device: torch.device,
) -> None:
    """Compare torch vs onnxruntime on the dummy batch (optional smoke check)."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("verify requires onnxruntime") from exc

    with torch.no_grad():
        torch_out = model(dummy).detach().float().cpu().numpy()

    session = ort.InferenceSession(
        str(out_path),
        providers=["CPUExecutionProvider"],
    )
    ort_out = session.run(
        [OUTPUT_NAME],
        {INPUT_NAME: dummy.detach().float().cpu().numpy()},
    )[0]

    max_abs = float(abs(torch_out.reshape(-1) - ort_out.reshape(-1)).max())
    logger.info("verify max|torch-ort|=%.6g on dummy batch", max_abs)
    if max_abs > 1e-4:
        logger.warning("torch/ort diverge more than 1e-4; check opset / dynamo flags")


def write_meta(meta: dict, out_path: Path, opset: int, dynamo: bool) -> Path:
    meta_path = out_path.with_suffix(".json")
    payload = {
        **meta,
        "onnx": str(out_path),
        "opset": opset,
        "dynamo": dynamo,
    }
    meta_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote metadata %s", meta_path)
    return meta_path


def main(args: argparse.Namespace) -> None:
    setup_logging(constants.LOGS_DIR)
    device = torch.device(args.device)

    model, _stats, meta = load_model(
        checkpoint=args.checkpoint,
        device=device,
        weights=args.weights,
    )
    logger.info(
        "loaded %s (epoch=%s, weights=%s, pooling=%s)",
        args.checkpoint,
        meta.get("epoch"),
        args.weights,
        meta.get("pooling"),
    )

    out_path: Path = args.out
    if out_path is None:
        out_path = args.checkpoint.with_suffix(".onnx")

    export_onnx(
        model=model,
        out_path=out_path,
        device=device,
        opset=args.opset,
        dynamo=args.dynamo,
        verify=args.verify,
    )
    write_meta(meta, out_path, opset=args.opset, dynamo=args.dynamo)
    logger.info("done: %s (%.1f MiB)", out_path, out_path.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export a rotation-detector checkpoint to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to an epoch_NNN.pt checkpoint",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .onnx path (default: checkpoint with .onnx suffix)",
    )
    p.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="which weights from the checkpoint to export",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="torch device used for the export trace",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    p.add_argument(
        "--dynamo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the torch.export/dynamo ONNX exporter (default: legacy)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="compare torch vs onnxruntime on a dummy batch after export",
    )
    main(p.parse_args())
