"""GradCAM and guided-GradCAM visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

import constants
from checkpoint import load_checkpoint_model
from data import build_eval_transform, display_preprocess
from inference import circular_mean_angle, stack_rotation_views
from model import RotationModel
from utils.image_io import open_rgb

if TYPE_CHECKING:
    import argparse

# ---------------------------------------------------------------------------


class GradCAM:
    """GradCAM and guided-GradCAM heatmap computation for one loaded model."""

    def __init__(
        self,
        model: RotationModel,
        device: torch.device,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.model = model
        self.device = device
        self.preprocess = build_eval_transform(mean=mean, std=std)

    def compute(self, tensor: torch.Tensor) -> tuple[list[np.ndarray], list[float]]:
        """Run one forward+backward and return per-sample GradCAM heatmaps
        (each in [0, 1]) plus per-sample upright logits."""
        target_layer: nn.Module = self.model.backbone.conv_head  # type: ignore[assignment]

        saved_features: list[torch.Tensor] = []
        saved_grads: list[torch.Tensor] = []

        def fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
            saved_features.append(out)

        def bwd(_m: nn.Module, _gin: object, gout: tuple[torch.Tensor, ...]) -> None:
            saved_grads.append(gout[0])

        fh = target_layer.register_forward_hook(fwd)
        bh = target_layer.register_full_backward_hook(bwd)  # type: ignore[arg-type]

        try:
            output = self.model(tensor)  # (B,) or (B, 1)
            self.model.zero_grad()
            output.sum().backward()  # per-sample grads are independent
        finally:
            fh.remove()
            bh.remove()

        feat = saved_features[0]  # (B, C, H, W)
        grad = saved_grads[0]  # (B, C, H, W)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = nn.functional.relu((weights * feat).sum(dim=1))  # (B, H, W)
        cam_max = cam.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        cam_np = (cam / cam_max).cpu().detach().float().numpy()

        logits = output.detach().cpu().float().reshape(-1).tolist()
        return list(cam_np), logits

    def guided(self, tensor: torch.Tensor, cams: list[np.ndarray]) -> list[np.ndarray]:
        """Guided backprop gated by the supplied GradCAM heatmaps.

        Produces sharp, class-discriminative, edge-like maps at full input
        resolution. SiLU stand-in for ReLU in the guided-BP gate: only pass
        gradient where the SiLU output was positive AND the incoming gradient
        is positive. Runs a single batched forward+backward over the batch.
        """
        silu_outputs: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_fwd(idx: int):
            def fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
                silu_outputs[idx] = out

            return fwd

        def make_bwd(idx: int):
            def bwd(
                _m: nn.Module,
                _gin: object,
                gout: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, ...]:
                out = silu_outputs[idx]
                new_grad = gout[0].clone()
                new_grad[out <= 0] = 0
                new_grad[new_grad < 0] = 0
                return (new_grad,)

            return bwd

        silu_modules = [m for m in self.model.modules() if isinstance(m, nn.SiLU)]
        original_inplace = [sm.inplace for sm in silu_modules]
        for sm in silu_modules:
            sm.inplace = False  # backward hooks reject view+inplace combo
        for i, sm in enumerate(silu_modules):
            handles.append(sm.register_forward_hook(make_fwd(i)))
            handles.append(sm.register_full_backward_hook(make_bwd(i)))  # type: ignore[arg-type]

        try:
            x = tensor.detach().clone().requires_grad_(True)
            output = self.model(x)
            self.model.zero_grad()
            output.sum().backward()
        finally:
            for h in handles:
                h.remove()
            for sm, was_inplace in zip(silu_modules, original_inplace, strict=True):
                sm.inplace = was_inplace

        grad = x.grad.abs().sum(dim=1)  # type: ignore[union-attr]  # (B, H, W)
        grad_np = grad.cpu().detach().float().numpy()

        guided_out: list[np.ndarray] = []
        for i, cam in enumerate(cams):
            h_in, w_in = grad_np[i].shape
            cam_up = (
                np.array(
                    Image.fromarray((cam * 255).astype(np.uint8)).resize((w_in, h_in), Image.Resampling.BICUBIC),
                ).astype(np.float32)
                / 255.0
            )

            guided = grad_np[i] * cam_up
            p99 = float(np.percentile(guided, 99))
            if p99 > 0:
                guided = np.clip(guided / p99, 0.0, 1.0)
            guided_out.append(guided)
        return guided_out

    def rotations(
        self,
        pil_img: Image.Image,
    ) -> tuple[list[Image.Image], list[float], list[np.ndarray], torch.Tensor]:
        # Display images: resize + center-crop, then PIL-rotate each view.
        crop = display_preprocess(pil_img)
        displays = [crop.rotate(a) if a != 0 else crop for a in constants.ROTATIONS]

        # Model input: transform once (resize/crop/to_tensor/normalize), then
        # rotate the tensor - same geometry as train/infer (RotationPairDataset,
        # predict_rotations), not PIL-rotate-then-transform.
        base = self.preprocess(pil_img)
        tensors = stack_rotation_views(base, self.device)

        cams, logits = self.compute(tensors)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0).tolist()
        return displays, probs, cams, tensors


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def overlay_heatmap(
    pil_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.65,
    colormap: str = "jet",
) -> Image.Image:
    cam = np.sqrt(cam)  # spread activation - avoids tiny concentrated hot spots
    heatmap = mpl.colormaps[colormap](cam)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_img.size, Image.Resampling.BICUBIC)
    blended = np.array(pil_img, dtype=np.float32) * (1 - alpha) + np.array(heatmap_pil, dtype=np.float32) * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def render_rotation_grid(
    displays: list[Image.Image],
    probs: list[float],
    heatmaps: list[np.ndarray],
    title: str,
    out_path: Path,
    colormap: str = "jet",
) -> None:
    """Render a 2x2 rotation grid with `heatmaps[i]` overlaid on `displays[i]`."""
    pred = int(np.argmax(probs))

    fig = Figure(figsize=(10, 10))
    axes = fig.subplots(2, 2)

    for idx, (angle, heat, prob, disp) in enumerate(
        zip(constants.ROTATIONS, heatmaps, probs, displays, strict=True)
    ):
        img_overlay = overlay_heatmap(disp, heat, colormap=colormap)
        ax = axes[idx // 2, idx % 2]
        marker = " *" if idx == pred else ""
        ax.imshow(img_overlay)
        ax.set_title(f"{angle}°{marker}  softmax={prob:.0%}", fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def run_gradcam(args: argparse.Namespace) -> None:
    images = [p for p in args.images if p.is_file()]

    device = torch.device(args.device)
    _, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    gradcam = GradCAM(model, device, mean, std)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        pil = open_rgb(img_path)
        displays, probs, cams, tensors = gradcam.rotations(pil)
        angle = circular_mean_angle(probs)

        render_rotation_grid(
            displays,
            probs,
            cams,
            f"{img_path.name} - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_gradcam.png",
            colormap="jet",
        )
        print(f"{img_path.name} - predicted angle {angle:.1f}°")

        guided_cams = gradcam.guided(tensors, cams)
        render_rotation_grid(
            displays,
            probs,
            guided_cams,
            f"{img_path.name} - guided GradCAM - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_guidedcam.png",
            colormap="inferno",
        )
