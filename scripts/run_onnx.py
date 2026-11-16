#!/usr/bin/env python3
"""Standalone ONNX rotation runner.

Ship this file together with an exported ``.onnx`` model; it needs only
``onnxruntime``, ``numpy``, and ``Pillow``.

CLI:
    python run_onnx.py model.onnx photo.jpg
    python run_onnx.py model.onnx photos/ --output-dir corrected/
    python run_onnx.py model.onnx photo.jpg --abstain-margin 0.15 --json

Import:
    from run_onnx import OnnxRotationModel

    model = OnnxRotationModel("model.onnx")
    prediction = model.predict(image)
    corrected = model.correct(image, prediction)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class Metadata:
    """Preprocessing and provenance values stored with the model."""

    mean: tuple[float, ...] = DEFAULT_MEAN
    std: tuple[float, ...] = DEFAULT_STD
    input_size: int = DEFAULT_INPUT_SIZE
    resize_size: int = DEFAULT_RESIZE_SIZE
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS
    input_name: str = DEFAULT_INPUT_NAME
    output_name: str = DEFAULT_OUTPUT_NAME
    epoch: str | None = None
    weights: str | None = None
    pooling: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Metadata:
        """Build metadata from ONNX strings, JSON values, or defaults."""

        def numbers(key: str, default: tuple[float | int, ...]) -> tuple:
            value = data.get(key, default)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return default
            return tuple(value) if isinstance(value, (list, tuple)) else default

        def number(key: str, default: int) -> int:
            value = data.get(key, default)
            return int(value) if isinstance(value, (int, str)) else default

        def text(key: str, default: str) -> str:
            value = data.get(key)
            return default if value is None else str(value)

        return cls(
            mean=tuple(float(value) for value in numbers("mean", DEFAULT_MEAN)),
            std=tuple(float(value) for value in numbers("std", DEFAULT_STD)),
            input_size=number("input_size", DEFAULT_INPUT_SIZE),
            resize_size=number("resize_size", DEFAULT_RESIZE_SIZE),
            rotations=tuple(int(value) for value in numbers("rotations", DEFAULT_ROTATIONS)),
            input_name=text("input_name", DEFAULT_INPUT_NAME),
            output_name=text("output_name", DEFAULT_OUTPUT_NAME),
            epoch=text("epoch", "") or None,
            weights=text("weights", "") or None,
            pooling=text("pooling", "") or None,
        )


@dataclass(frozen=True)
class Prediction:
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

        # Priority: embedded ONNX metadata > sidecar JSON > defaults.
        data: dict[str, object] = {}
        sidecar = self.model_path.with_suffix(".json")
        if sidecar.exists():
            try:
                loaded = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                loaded = None
            data = loaded if isinstance(loaded, dict) else {}
        data.update(self.session.get_modelmeta().custom_metadata_map or {})
        self.metadata = Metadata.from_dict(data)

        self.mean = np.asarray(self.metadata.mean, dtype=np.float32)
        self.std = np.asarray(self.metadata.std, dtype=np.float32)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize shorter side, center crop, normalize, return CHW float32."""
        meta = self.metadata
        image = ImageOps.exif_transpose(image).convert("RGB")

        width, height = image.size
        scale = meta.resize_size / min(width, height)
        if scale != 1.0:
            image = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.BILINEAR,
            )

        left = (image.width - meta.input_size) // 2
        top = (image.height - meta.input_size) // 2
        image = image.crop(
            (
                left,
                top,
                left + meta.input_size,
                top + meta.input_size,
            )
        )

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        return array.transpose(2, 0, 1)

    def predict(self, image: Image.Image) -> Prediction:
        """Score every rotation and return probabilities and the best view."""
        meta = self.metadata
        chw = self.preprocess(image)
        views = [
            np.rot90(chw, k=(degrees // 90) % 4, axes=(1, 2))
            for degrees in meta.rotations
        ]
        batch = np.stack(views).astype(np.float32)
        logits = np.asarray(
            self.session.run([meta.output_name], {meta.input_name: batch})[0],
            dtype=np.float64,
        ).reshape(-1)

        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        order = np.argsort(probabilities)
        top1 = int(order[-1])
        top2 = int(order[-2])

        return Prediction(
            probs={
                degrees: float(probability)
                for degrees, probability in zip(meta.rotations, probabilities, strict=True)
            },
            best_deg=meta.rotations[top1],
            margin=float(probabilities[top1] - probabilities[top2]),
            logits=[float(value) for value in logits],
        )

    def correct(
        self,
        image: Image.Image,
        prediction: Prediction,
        abstain_margin: float | None = None,
    ) -> Image.Image:
        """Return the image rotated upright, or unchanged when it abstains."""
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if abstain_margin is not None and prediction.margin < abstain_margin:
            return rgb
        return rgb.rotate(prediction.best_deg, expand=True)


def collect_images(paths: list[Path]) -> list[Path]:
    """Expand files and directories into a flat image list."""
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


def main(
    model_path: Path,
    images: list[Path],
    output_dir: Path | None = None,
    abstain_margin: float | None = None,
    as_json: bool = False,
) -> int:
    model = OnnxRotationModel(model_path)
    paths = collect_images(images)
    if not paths:
        print("no images found", file=sys.stderr)
        return 1

    print(
        f"model: {model_path}"
        f" (epoch={model.metadata.epoch}, weights={model.metadata.weights})",
        file=sys.stderr,
    )

    for path in paths:
        try:
            with Image.open(path) as image:
                prediction = model.predict(image)
                corrected = model.correct(image, prediction, abstain_margin)
        except (OSError, ValueError) as exc:
            print(f"{path}: ERROR: {exc}", file=sys.stderr)
            continue

        abstained = abstain_margin is not None and prediction.margin < abstain_margin
        output_path = (
            save_corrected(corrected, path, output_dir)
            if output_dir is not None
            else None
        )

        if as_json:
            print(
                json.dumps(
                    {
                        "path": str(path),
                        "best_deg": prediction.best_deg,
                        "margin": round(prediction.margin, 6),
                        "probs": {
                            str(degrees): round(probability, 6)
                            for degrees, probability in prediction.probs.items()
                        },
                        "abstained": abstained,
                        "output": str(output_path) if output_path is not None else None,
                    }
                )
            )
        else:
            status = "ABSTAIN" if abstained else f"correction {prediction.best_deg}°"
            confidence = max(prediction.probs.values())
            suffix = f" -> {output_path}" if output_path is not None else ""
            print(
                f"{path}: {status}  margin={prediction.margin:.3f}"
                f"  conf={confidence:.3f}{suffix}"
            )

    return 0


if __name__ == "__main__":
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
    args = parser.parse_args()
    raise SystemExit(
        main(
            args.model,
            args.images,
            args.output_dir,
            args.abstain_margin,
            args.json,
        )
    )
