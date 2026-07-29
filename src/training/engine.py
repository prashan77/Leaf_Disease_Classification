"""Device, metric, optimization, and evaluation helpers."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import f1_from_confusion


def seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    """Choose the requested accelerator or the best available device."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before measuring elapsed time."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> tuple[float, float]:
    """Train for one pass over the loader and return loss and accuracy."""

    model.train()
    loss_sum = 0.0
    correct = 0
    num_samples = 0

    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.numel()
        loss_sum += loss.detach().item() * batch_size
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        num_samples += batch_size
        progress.set_postfix(loss=f"{loss_sum / num_samples:.3f}")

    if num_samples == 0:
        raise ValueError("training loader is empty")
    return loss_sum / num_samples, correct / num_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    description: str,
) -> dict[str, float | torch.Tensor]:
    """Evaluate a model and return loss, accuracy, F1, confusion, and timing."""

    model.eval()
    loss_sum = 0.0
    correct = 0
    num_samples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    synchronize(device)
    started = time.perf_counter()
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")

        predictions = logits.argmax(dim=1)
        loss_sum += loss.item()
        correct += (predictions == labels).sum().item()
        num_samples += labels.numel()
        indices = (labels * num_classes + predictions).detach().cpu()
        confusion += torch.bincount(
            indices, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)
    synchronize(device)
    inference_seconds = time.perf_counter() - started

    if num_samples == 0:
        raise ValueError("evaluation loader is empty")
    f1_macro, f1_weighted = f1_from_confusion(confusion)
    return {
        "loss": loss_sum / num_samples,
        "acc": correct / num_samples,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": confusion,
        "inference_seconds": inference_seconds,
    }


def build_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int):
    """Create a warmup followed by cosine learning-rate decay."""

    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)

    warmup = LinearLR(
        optimizer,
        start_factor=1 / max(warmup_epochs, 2),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
