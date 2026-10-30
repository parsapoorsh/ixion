#!/usr/bin/env python3
"""Run four-view rotation inference with an ONNX upright detector.

Standalone runner for a model produced by ``src/onnx-export.py``. Prefers the
sibling ``.json`` metadata (mean/std, sizes, IO names); falls back to ImageNet
defaults if the JSON is missing. Geometry matches training and ``infer.py``:
eval preprocess first, then 90° tensor rotates, then a single batched ONNX
forward. Softmax over the four logits yields the upright vote.

Dependencies: onnxruntime, numpy, Pillow. Optional pillow-jxl-plugin for .jxl.

Usage (from project root):
    python scripts/onnx-run.py \\
        --model checkpoints/epoch_010.onnx \\
        --image examples/source/tholo2.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps

try:
    import pillow_jxl  # noqa: F401  registers JXL with PIL when installed
except ImportError:
    pass

import onnxruntime as ort

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Defaults mirror src/constants.py so a bare .onnx still runs.
DEFAULT_ROTATIONS = (0, 90, 180, 270)
DEFAULT_INPUT_SIZE = 224
DEFAULT_RESIZE_SIZE = 232
# ImageNet - only used when the export JSON is absent.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelMeta:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    input_size: int
    resize_size: int
    rotations: tuple[int, ...]
    input_name: str
    output_name: str


@dataclass(frozen=True)
class RotationVote:
    probs: dict[int, float]
    best_deg: int
    margin: float
    angle: float

    def should_abstain(self, threshold: float) -> bool:
        return self.margin < threshold


def load_meta(model_path: Path, meta_path: Path | None) -> ModelMeta:
    path = meta_path if meta_path is not None else model_path.with_suffix(".json")
    if path.is_file():
        data = json.loads(path.read_text())
        return ModelMeta(
            mean=tuple(float(x) for x in data["mean"]),  # type: ignore[arg-type]
            std=tuple(float(x) for x in data["std"]),  # type: ignore[arg-type]
            input_size=int(data.get("input_size", DEFAULT_INPUT_SIZE)),
            resize_size=int(data.get("resize_size", DEFAULT_RESIZE_SIZE)),
            rotations=tuple(int(x) for x in data.get("rotations", DEFAULT_ROTATIONS)),
            input_name=str(data.get("input_name", "input")),
            output_name=str(data.get("output_name", "logits")),
        )

    print(f"warning: no metadata at {path}; using ImageNet mean/std", file=sys.stderr)
    return ModelMeta(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        input_size=DEFAULT_INPUT_SIZE,
        resize_size=DEFAULT_RESIZE_SIZE,
        rotations=DEFAULT_ROTATIONS,
        input_name="input",
        output_name="logits",
    )


def open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")


def resize_shorter_edge(img: Image.Image, size: int) -> Image.Image:
    """Match torchvision.transforms.Resize(size) on the shorter edge."""
    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {img.size}")
    if width < height:
        new_w = size
        new_h = int(round(height * size / width))
    else:
        new_h = size
        new_w = int(round(width * size / height))
    # BICUBIC is closer to modern torchvision bilinear+antialias than NEAREST.
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    left = (width - size) // 2
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess(img: Image.Image, meta: ModelMeta) -> np.ndarray:
    """Eval geometry: resize -> center-crop -> float CHW normalize.

    Returns a contiguous float32 array of shape ``(3, H, W)``. Uses
    torchvision when installed so pixel values match ``builders.build_eval_transform``;
    otherwise a pure PIL/numpy path.
    """
    try:
        from torchvision import transforms as T

        tensor = T.Compose(
            [
                T.Resize(meta.resize_size),
                T.CenterCrop(meta.input_size),
                T.ToTensor(),
                T.Normalize(meta.mean, meta.std),
            ]
        )(img)
        return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))
    except ImportError:
        pass

    img = resize_shorter_edge(img, meta.resize_size)
    img = center_crop(img, meta.input_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    mean = np.asarray(meta.mean, dtype=np.float32)[:, None, None]
    std = np.asarray(meta.std, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((arr - mean) / std)


def rotate_chw(chw: np.ndarray, deg: int) -> np.ndarray:
    """Counter-clockwise rotate a CHW tensor by a multiple of 90°.

    Matches ``torchvision.transforms.functional.rotate(..., expand=False)``
    on a square tensor.
    """
    if deg % 360 == 0:
        return chw
    if deg % 90 != 0:
        raise ValueError(f"only 90° steps supported, got {deg}")
    k = (deg // 90) % 4
    # axes (H, W) == (1, 2); np.rot90 is counter-clockwise like TF.rotate.
    return np.ascontiguousarray(np.rot90(chw, k=k, axes=(1, 2)))


def vote_from_logits(logits: np.ndarray, rotations: tuple[int, ...]) -> RotationVote:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits.shape[0] != len(rotations):
        raise ValueError(f"expected {len(rotations)} logits, got {logits.shape[0]}")

    # Numerically stable softmax over the four views.
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()

    order = np.argsort(probs)
    top1_i = int(order[-1])
    top2_i = int(order[-2]) if len(order) > 1 else top1_i
    best_deg = int(rotations[top1_i])
    margin = float(probs[top1_i] - probs[top2_i])

    angles = np.deg2rad(np.asarray(rotations, dtype=np.float64))
    x = float(np.sum(probs * np.cos(angles)))
    y = float(np.sum(probs * np.sin(angles)))
    angle = float(np.rad2deg(np.arctan2(y, x)) % 360.0)

    return RotationVote(
        probs={deg: float(p) for deg, p in zip(rotations, probs, strict=True)},
        best_deg=best_deg,
        margin=margin,
        angle=angle,
    )


def make_session(model_path: Path, providers: list[str] | None) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if providers is None:
        # Prefer CUDA when the wheel includes it; always keep CPU as fallback.
        chosen = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if not chosen:
            chosen = available
    else:
        chosen = providers
        missing = [p for p in chosen if p not in available]
        if missing:
            raise RuntimeError(f"requested providers unavailable: {missing}; have {available}")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=so, providers=chosen)


def predict_rotations(
    session: ort.InferenceSession,
    img: Image.Image,
    meta: ModelMeta,
) -> RotationVote:
    base = preprocess(img, meta)
    views = np.stack([rotate_chw(base, deg) for deg in meta.rotations], axis=0)
    outputs = session.run(
        [meta.output_name],
        {meta.input_name: views.astype(np.float32)},
    )
    logits = np.asarray(outputs[0]).reshape(-1)
    return vote_from_logits(logits, meta.rotations)


def format_vote(vote: RotationVote) -> str:
    parts = [f"{deg}°={vote.probs[deg]:.4f}" for deg in sorted(vote.probs)]
    return (
        f"best={vote.best_deg}°  margin={vote.margin:.4f}  angle={vote.angle:.1f}°  "
        f"probs={{{', '.join(parts)}}}"
    )


def main(args: argparse.Namespace) -> int:
    meta = load_meta(args.model, args.meta)
    session = make_session(args.model, args.providers)
    print(
        f"model={args.model}  providers={session.get_providers()}  "
        f"input={meta.input_name}  output={meta.output_name}",
        file=sys.stderr,
    )

    paths: list[Path] = list(args.images)
    if not paths:
        print("no images given", file=sys.stderr)
        return 2

    abstain_n = 0
    for path in paths:
        try:
            img = open_rgb(path)
            vote = predict_rotations(session, img, meta)
        except Exception as exc:  # noqa: BLE001 - CLI surface: report and continue
            print(f"{path}\tERROR\t{exc}")
            continue

        abstained = args.abstain_margin is not None and vote.should_abstain(args.abstain_margin)
        if abstained:
            abstain_n += 1
            status = "abstain"
        else:
            status = f"rotate={vote.best_deg}"
        print(f"{path}\t{status}\t{format_vote(vote)}")

        if args.save_corrected is not None and not abstained and vote.best_deg != 0:
            out_dir: Path = args.save_corrected
            out_dir.mkdir(parents=True, exist_ok=True)
            corrected = img.rotate(vote.best_deg, expand=True)
            out_path = out_dir / path.name
            corrected.save(out_path)
            print(f"  wrote {out_path}", file=sys.stderr)

    if args.abstain_margin is not None:
        print(f"abstained {abstain_n}/{len(paths)} (margin < {args.abstain_margin})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Four-view rotation inference with an ONNX upright detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="path to the .onnx model from onnx-export.py",
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="metadata JSON (default: model path with .json suffix)",
    )
    p.add_argument(
        "--image",
        dest="images",
        type=Path,
        action="append",
        default=[],
        help="image path (repeatable)",
    )
    p.add_argument(
        "positional_images",
        nargs="*",
        type=Path,
        help="image paths as positional args",
    )
    p.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="onnxruntime execution providers (default: CUDA then CPU if available)",
    )
    p.add_argument(
        "--abstain-margin",
        type=float,
        default=None,
        dest="abstain_margin",
        help="leave unrotated when top1-top2 margin is below this",
    )
    p.add_argument(
        "--save-corrected",
        type=Path,
        default=None,
        dest="save_corrected",
        help="if set, write corrected images (non-abstain, best_deg != 0) here",
    )
    ns = p.parse_args()
    ns.images = list(ns.images) + list(ns.positional_images)
    raise SystemExit(main(ns))
