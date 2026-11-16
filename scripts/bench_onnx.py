#!/usr/bin/env python3
"""Benchmark ONNX rotation inference throughput.

The timer covers preprocessing and inference; image decoding is done
before timing starts. Run with ``--device cuda`` for the GPU path or
``--device cpu`` to compare against the CPU execution provider.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from run_onnx import OnnxRotationModel, collect_images

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def main(
    model_path: Path,
    image_paths: list[Path],
    device: str = "cuda",
    iterations: int = 200,
    warmup: int = 20,
    max_images: int = 32,
) -> int:
    if iterations < 1:
        print("iterations must be at least 1", file=sys.stderr)
        return 2

    paths = collect_images(image_paths)[:max_images]
    if not paths:
        print("no images found", file=sys.stderr)
        return 2

    images = []
    for path in paths:
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        except OSError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
    if not images:
        print("no readable images", file=sys.stderr)
        return 2

    available = ort.get_available_providers()
    if device == "cuda":
        if CUDA_PROVIDER not in available:
            print(
                f"{CUDA_PROVIDER} is not available in this onnxruntime build.\n"
                f"available providers: {available}\n"
                "install the GPU build with:\n"
                "  pip uninstall -y onnxruntime && pip install onnxruntime-gpu",
                file=sys.stderr,
            )
            return 2
        providers = [CUDA_PROVIDER, CPU_PROVIDER]
    else:
        providers = [CPU_PROVIDER]

    try:
        model = OnnxRotationModel(model_path, providers=providers)
    except (ValueError, RuntimeError) as exc:
        print(f"failed to create ONNX session: {exc}", file=sys.stderr)
        return 2

    active = model.session.get_providers()[0]
    if device == "cuda" and active != CUDA_PROVIDER:
        print(
            f"requested {CUDA_PROVIDER} but the session uses {active};"
            " check the CUDA/cuDNN libraries and rerun",
            file=sys.stderr,
        )
        return 2

    for index in range(warmup):
        model.predict(images[index % len(images)])

    latencies: list[float] = []
    started = time.perf_counter()
    for index in range(iterations):
        image = images[index % len(images)]
        tick = time.perf_counter()
        model.predict(image)
        latencies.append(time.perf_counter() - tick)
    elapsed = time.perf_counter() - started

    images_per_second = iterations / elapsed
    views_per_second = images_per_second * len(model.metadata.rotations)
    milliseconds = np.asarray(latencies) * 1000

    print(f"model: {model_path}")
    print(f"provider: {active}")
    print(
        f"images: {len(images)} file(s), cycling"
        f"; warmup={warmup}, iterations={iterations}"
    )
    print(
        f"throughput: {images_per_second:.1f} images/s"
        f" ({views_per_second:.1f} views/s)"
    )
    print(
        f"latency: mean={milliseconds.mean():.2f} ms"
        f" p50={np.percentile(milliseconds, 50):.2f} ms"
        f" p95={np.percentile(milliseconds, 95):.2f} ms"
    )
    print("timing excludes image decode")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX rotation inference throughput.",
    )
    parser.add_argument("model", type=Path, help="exported .onnx model")
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="image files or directories to sample from",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="execution provider to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="timed images after warmup",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="untimed images before measuring",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=32,
        help="maximum images to preload into memory",
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            args.model,
            args.images,
            args.device,
            args.iterations,
            args.warmup,
            args.max_images,
        )
    )
