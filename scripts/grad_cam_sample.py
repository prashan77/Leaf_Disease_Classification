#!/usr/bin/env python
"""
grad_cam_sample.py -- Grad-CAM over a random sample of correctly and
incorrectly classified test images for one checkpoint.

Scoring the full test set uses the batched get_dataloaders() path (fast);
only the sampled images go through Grad-CAM's single-image, gradient-tracked
path afterward. Reuses scripts.grad_cam's checkpoint loading, target-layer
resolution, and image loading so normalization and target layer stay
identical to a plain grad_cam.py run.

Usage:
    python -m scripts.grad_cam_sample --checkpoint results/.../best_model.pt \
        --n-correct 5 --n-incorrect 5 --seed 0
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from scripts.grad_cam import load_checkpoint, load_image, select_device, target_layer
from src.data import DATA_ROOT, get_dataloaders


def score_test_set(model, config, device) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """Return (correct, incorrect) as (relpath, true_label, predicted_label) triples."""

    _, test_loader = get_dataloaders(
        config["variant"], config["ratio"], config["seed"], batch_size=64, num_workers=0
    )
    relpaths = test_loader.dataset.relpaths

    correct, incorrect = [], []
    seen = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for offset in range(labels.size(0)):
                relpath = relpaths[seen + offset]
                true_label = labels[offset].item()
                predicted = predictions[offset].item()
                (correct if predicted == true_label else incorrect).append((relpath, true_label, predicted))
            seen += labels.size(0)
    return correct, incorrect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="defaults to <checkpoint's run folder>/gradcam, alongside that run's plots/",
    )
    parser.add_argument("--n-correct", type=int, default=5)
    parser.add_argument("--n-incorrect", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out is None:
        args.out = args.checkpoint.parent / "gradcam"

    device = select_device()
    model, config, class_names = load_checkpoint(args.checkpoint, device)
    layer = target_layer(model, config["architecture"])
    cam = GradCAM(model=model, target_layers=[layer])
    root = DATA_ROOT / "raw" / config["variant"]

    print(f"scoring test set ({config['variant']}, ratio={config['ratio']}, seed={config['seed']})...")
    correct, incorrect = score_test_set(model, config, device)
    print(f"  correct: {len(correct)}  incorrect: {len(incorrect)}")

    if len(incorrect) < args.n_incorrect:
        print(f"  !! only {len(incorrect)} incorrect predictions available, using all of them")

    rng = random.Random(args.seed)
    chosen = (
        [(*item, "correct") for item in rng.sample(correct, min(args.n_correct, len(correct)))]
        + [(*item, "incorrect") for item in rng.sample(incorrect, min(args.n_incorrect, len(incorrect)))]
    )

    for bucket in ("correct", "incorrect"):
        (args.out / bucket).mkdir(parents=True, exist_ok=True)

    for relpath, true_label, predicted, bucket in chosen:
        source_path = root / relpath
        input_tensor, rgb_float = load_image(source_path)
        input_tensor = input_tensor.to(device)

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted)])[0]
        visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        stem = f"true-{class_names[true_label]}_pred-{class_names[predicted]}_{Path(relpath).stem}"
        bucket_dir = args.out / bucket
        shutil.copy2(source_path, bucket_dir / f"{stem}_original{source_path.suffix}")
        Image.fromarray(visualization).save(bucket_dir / f"{stem}_gradcam.png")
        print(f"  [{bucket}] {relpath}  true={class_names[true_label]}  pred={class_names[predicted]}")

    print(f"done -- {len(chosen)} images written under {args.out}/correct/ and {args.out}/incorrect/")


if __name__ == "__main__":
    main()
