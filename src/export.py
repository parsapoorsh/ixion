"""ONNX export for a trained checkpoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import constants
from checkpoint import load_checkpoint_model
from model import RotationModel

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def export_onnx(
    model: RotationModel,
    out_path: Path,
    device: torch.device,
    opset: int,
    dynamo: bool,
    verify: bool,
    metadata: dict[str, str],
) -> None:
    """Trace ``model`` to ONNX with a dynamic batch axis."""
    dummy = torch.randn(1, 3, constants.INPUT_SIZE, constants.INPUT_SIZE, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs: dict = {
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": {
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
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

    # ONNX custom metadata is a string map; the standalone runner reads it
    # through onnxruntime, so the model is self-describing without a sidecar.
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("embedding ONNX metadata requires the onnx package (pip install onnx)") from exc

    model_proto = onnx.load(out_path)
    for key, value in metadata.items():
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model_proto, out_path)
    logger.info("embedded %d metadata entries", len(metadata))

    if verify:
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


def run_export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    payload, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    epoch = payload.get("epoch")
    epoch_label = epoch if epoch is not None else "unknown"
    out_path = args.out or (
        constants.CHECKPOINT_DIR / "onnx" / f"ixion-efficientnet-e{epoch_label}-{args.weights}.onnx"
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "epoch": str(epoch_label),
        "weights": args.weights,
        "pooling": str(payload.get("pooling", "avg")),
        "pretrained": str(payload.get("pretrained")),
        "mean": json.dumps(list(mean)),
        "std": json.dumps(list(std)),
        "input_size": str(constants.INPUT_SIZE),
        "resize_size": str(constants.RESIZE_SIZE),
        "rotations": json.dumps(list(constants.ROTATIONS)),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "opset": str(args.opset),
        "dynamo": str(args.dynamo).lower(),
    }

    export_onnx(
        model=model,
        out_path=out_path,
        device=device,
        opset=args.opset,
        dynamo=args.dynamo,
        verify=args.verify,
        metadata=metadata,
    )

    logger.info("done: %s (%.1f MiB)", out_path, out_path.stat().st_size / (1024 * 1024))
