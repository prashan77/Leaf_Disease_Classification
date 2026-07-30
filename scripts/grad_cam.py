#!/usr/bin/env python
"""
grad_cam.py -- visualize what a trained checkpoint attends to.

Reuses src.data.build_transforms(train=False) for the model's input tensor,
so normalization always matches whatever the checkpoint was actually trained
with. Grad-CAM computed against a mismatched mean/std still produces a
plausible-looking heatmap -- just for the wrong input distribution -- so this
must not hardcode its own stats.

Usage:
    python -m scripts.grad_cam --checkpoint results/mobilenet_v2_color_pretrained_0.8_42/best_model.pt \
        data/raw/color/Tomato___healthy/some_image.JPG data/raw/color/Corn_*/other.JPG
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.data import IMAGE_SIZE, SOURCE_SIZE, build_transforms
from src.models import build_model

EVAL_CROP = transforms.Compose([
    transforms.Resize(SOURCE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
])


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def target_layer(model: torch.nn.Module, architecture: str) -> torch.nn.Module:
    """Last conv block before pooling -- the standard Grad-CAM hook point."""
    if architecture == "mobilenet_v2":
        return model.network.features[-1]
    if architecture == "vanilla_cnn":
        return model.features[-1]
    raise ValueError(f"no known target layer for architecture: {architecture}")


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = build_model(config["architecture"], config["initialization"], config["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, config, checkpoint["class_names"]


def load_image(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Return (normalized input tensor, [0,1] RGB array for the overlay)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        rgb_float = np.asarray(EVAL_CROP(img), dtype=np.float32) / 255.0
        input_tensor = build_transforms(train=False)(img).unsqueeze(0)
    return input_tensor, rgb_float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="defaults to <checkpoint's run folder>/gradcam, alongside that run's plots/",
    )
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    if args.out is None:
        args.out = args.checkpoint.parent / "gradcam"

    device = select_device()
    model, config, class_names = load_checkpoint(args.checkpoint, device)
    layer = target_layer(model, config["architecture"])
    cam = GradCAM(model=model, target_layers=[layer])

    args.out.mkdir(parents=True, exist_ok=True)
    for image_path in args.images:
        input_tensor, rgb_float = load_image(image_path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            predicted = model(input_tensor).argmax(dim=1).item()

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted)])[0]
        visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        out_path = args.out / f"{image_path.stem}_{class_names[predicted]}.png"
        Image.fromarray(visualization).save(out_path)
        print(f"{image_path.name}: predicted {class_names[predicted]} -> {out_path}")


if __name__ == "__main__":
    main()
