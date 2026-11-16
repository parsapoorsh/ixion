#!/usr/bin/env python3
"""Standalone ONNX rotation runner.

Ship this file together with an exported ``.onnx`` model; it needs only
``onnxruntime``, ``numpy``, and ``Pillow``.

CLI:
    python run_onnx.py model.onnx photo.jpg
    python run_onnx.py model.onnx photos/ --output-dir corrected/
    python run_onnx.py model.onnx photo.jpg --abstain-margin 0.15 --json

Import:
    from run_onnx import load_model

    model = load_model("model.onnx")
    prediction = model.predict(image)
    corrected = model.correct(image, prediction)

Metadata comes from the ONNX file (``metadata_props``). A sidecar JSON with
the same stem is used as a fallback, and hardcoded constants are used when
neither exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = 224
DEFAULT_RESIZE_SIZE = 232
DEFAULT_ROTATIONS = (0, 90, 180, 270)
DEFAULT_INPUT_NAME = "input"
DEFAULT_OUTPUT_NAME = "logits"

IMAGE_EXTS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    }
)


class Prediction(TypedDict):
    probs: dict[int, float]
    best_deg: int
    margin: float
    logits: list[float]


class OnnxRotationModel:
    """Four-view rotation scorer backed by an ONNX session."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.metadata = self.read_metadata()
        self.mean = np.asarray(self.read_list("mean", DEFAULT_MEAN), dtype=np.float32)
        self.std = np.asarray(self.read_list("std", DEFAULT_STD), dtype=np.float32)
        self.input_size = self.read_int("input_size", DEFAULT_INPUT_SIZE)
        self.resize_size = self.read_int("resize_size", DEFAULT_RESIZE_SIZE)
        self.rotations = [int(value) for value in self.read_list("rotations", DEFAULT_ROTATIONS)]
        self.input_name = self.read_str("input_name", DEFAULT_INPUT_NAME)
        self.output_name = self.read_str("output_name", DEFAULT_OUTPUT_NAME)

    def read_metadata(self) -> dict[str, object]:
        """Sidecar JSON first, embedded ONNX metadata second (it wins)."""
        metadata: dict[str, object] = {}
        sidecar = self.model_path.with_suffix(".json")
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict):
                metadata.update(data)

        embedded = self.session.get_modelmeta().custom_metadata_map or {}
        metadata.update(embedded)
        return metadata

    def read_list(self, key: str, default: tuple) -> list:
        value = self.metadata.get(key, default)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return list(default)
        return list(value) if isinstance(value, (list, tuple)) else list(default)

    def read_int(self, key: str, default: int) -> int:
        value = self.metadata.get(key, default)
        if not isinstance(value, (int, str)):
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def read_str(self, key: str, default: str) -> str:
        value = self.metadata.get(key)
        return default if value is None else str(value)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize shorter side, center crop, normalize, return CHW float32."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = self.resize_size / min(width, height)
        if scale != 1.0:
            image = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.BILINEAR,
            )

        left = (image.width - self.input_size) // 2
        top = (image.height - self.input_size) // 2
        image = image.crop(
            (
                left,
                top,
                left + self.input_size,
                top + self.input_size,
            )
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        return array.transpose(2, 0, 1)

    def predict(self, image: Image.Image) -> Prediction:
        """Score the four rotations and return probabilities and the best view."""
        chw = self.preprocess(image)
        views = [
            np.rot90(chw, k=(deg // 90) % 4, axes=(1, 2))
            for deg in self.rotations
        ]
        batch = np.stack(views).astype(np.float32)
        logits = np.asarray(
            self.session.run([self.output_name], {self.input_name: batch})[0],
            dtype=np.float64,
        ).reshape(-1)

        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        order = np.argsort(probabilities)
        top1 = int(order[-1])
        top2 = int(order[-2])

        return {
            "probs": {
                deg: float(probability)
                for deg, probability in zip(self.rotations, probabilities, strict=True)
            },
            "best_deg": self.rotations[top1],
            "margin": float(probabilities[top1] - probabilities[top2]),
            "logits": [float(value) for value in logits],
        }

    def correct(
        self,
        image: Image.Image,
        prediction: Prediction,
        abstain_margin: float | None = None,
    ) -> Image.Image:
        """Return the image rotated upright, or unchanged when it abstains."""
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if abstain_margin is not None and prediction["margin"] < abstain_margin:
            return rgb
        return rgb.rotate(prediction["best_deg"], expand=True)


def load_model(model_path: str | Path) -> OnnxRotationModel:
    """Load an exported ONNX model and its preprocessing metadata."""
    return OnnxRotationModel(model_path)


def collect_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS
                )
            )
        elif path.is_file():
            images.append(path)
        else:
            print(f"not found: {path}", file=sys.stderr)
    return images


def save_corrected(image: Image.Image, path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_corrected{path.suffix.lower()}"
    if output_path.suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score four rotations of photos and correct them upright.",
    )
    parser.add_argument("model", type=Path, help="exported .onnx model")
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="image files or directories to scan recursively",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="write corrected images here instead of only printing predictions",
    )
    parser.add_argument(
        "--abstain-margin",
        type=float,
        default=None,
        help="leave the image unrotated when top1-top2 probability is below this",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print one JSON object per image",
    )
    args = parser.parse_args(argv)

    model = load_model(args.model)
    paths = collect_images(args.images)
    if not paths:
        print("no images found", file=sys.stderr)
        return 1

    epoch = model.metadata.get("epoch")
    weights = model.metadata.get("weights")
    print(
        f"model: {args.model} (epoch={epoch}, weights={weights})",
        file=sys.stderr,
    )

    for path in paths:
        try:
            with Image.open(path) as image:
                prediction = model.predict(image)
                corrected = model.correct(image, prediction, args.abstain_margin)
        except (OSError, ValueError) as exc:
            print(f"{path}: ERROR: {exc}", file=sys.stderr)
            continue

        abstained = (
            args.abstain_margin is not None
            and prediction["margin"] < args.abstain_margin
        )
        output_path = (
            save_corrected(corrected, path, args.output_dir)
            if args.output_dir is not None
            else None
        )

        if args.json:
            print(
                json.dumps(
                    {
                        "path": str(path),
                        "best_deg": prediction["best_deg"],
                        "margin": round(prediction["margin"], 6),
                        "probs": {
                            str(deg): round(probability, 6)
                            for deg, probability in prediction["probs"].items()
                        },
                        "abstained": abstained,
                        "output": str(output_path) if output_path is not None else None,
                    }
                )
            )
        else:
            status = "ABSTAIN" if abstained else f"correction {prediction['best_deg']}°"
            confidence = max(prediction["probs"].values())
            suffix = f" -> {output_path}" if output_path is not None else ""
            print(
                f"{path}: {status}  margin={prediction['margin']:.3f}"
                f"  conf={confidence:.3f}{suffix}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
