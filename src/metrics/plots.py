"""Report-ready plots shared by all model architectures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch


def plot_metrics(
    train_accuracy: Sequence[float],
    val_accuracy: Sequence[float],
    val_f1: Sequence[float],
    confusion_matrix: torch.Tensor,
    name: str,
    output_dir: Path = Path("images"),
) -> list[Path]:
    """Write the accuracy, validation-F1, and confusion-matrix plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_accuracy) + 1)
    file_name = name.lower().replace(" ", "_")
    written_paths = []

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(epochs, train_accuracy, marker="o", label="Train")
    axis.plot(epochs, val_accuracy, marker="o", label="Validation")
    axis.set_title(f"{name} Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.legend()
    figure.tight_layout()
    path = output_dir / f"{file_name}_accuracy.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written_paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(epochs, val_f1, marker="o")
    axis.set_title(f"{name} Validation F1")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("F1")
    figure.tight_layout()
    path = output_dir / f"{file_name}_validation_f1.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written_paths.append(path)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(confusion_matrix.cpu().numpy(), cmap="Blues")
    axis.set_title(f"{name} Confusion Matrix")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path = output_dir / f"{file_name}_confusion_matrix.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written_paths.append(path)

    return written_paths
