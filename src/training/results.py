"""Persistence for configuration, epoch history, and checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

RESULT_COLUMNS = (
    "architecture",
    "variant",
    "initialization",
    "ratio",
    "seed",
    "epoch",
    "learning_rate",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_f1_macro",
    "val_f1_weighted",
    "training_seconds",
    "val_inference_seconds",
    "epoch_seconds",
)


class RunArtifacts:
    """Own all files written by one experiment."""

    def __init__(self, root: Path, overwrite: bool):
        """Set the output paths and prepare the run directory."""

        self.root = Path(root)
        self.config_path = self.root / "config.json"
        self.history_path = self.root / "history.csv"
        self.checkpoint_path = self.root / "best_model.pt"
        self.summary_path = self.root / "summary.json"
        self._prepare(overwrite)

    def _prepare(self, overwrite: bool) -> None:
        """Create the output directory and handle existing result files."""

        paths = (
            self.config_path,
            self.history_path,
            self.checkpoint_path,
            self.summary_path,
        )
        existing = [path for path in paths if path.exists()]
        if existing and not overwrite:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"{self.root} already contains run artifacts ({names})"
            )
        if overwrite:
            for path in existing:
                path.unlink()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: dict[str, object]) -> None:
        """Save the settings and dataset details used for the run."""

        self.config_path.write_text(json.dumps(config, indent=2) + "\n")

    def append_epoch(self, result: dict[str, object]) -> None:
        """Append one epoch of training and validation metrics to the CSV."""

        write_header = not self.history_path.exists()
        with self.history_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(result)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
        epoch: int,
        val_f1_macro: float,
        config: dict[str, object],
        class_names: list[str],
    ) -> None:
        """Save model and optimizer state for the best validation checkpoint."""

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "val_f1_macro": val_f1_macro,
                "config": config,
                "class_names": class_names,
            },
            self.checkpoint_path,
        )

    def load_model(self, model: nn.Module, device: torch.device) -> None:
        """Load the saved best weights into a model."""

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    def write_summary(self, summary: dict[str, object]) -> None:
        """Save final test metrics and artifact locations as JSON."""

        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n")
