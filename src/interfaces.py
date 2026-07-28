"""
src/interfaces.py -- FROZEN. All four teammates must agree to change this file.


"""

from __future__ import annotations

import numpy as np
from torch import nn
from torch.utils.data import DataLoader


def get_dataloaders(variant: str, ratio: float, seed: int, batch_size: int
                     ) -> tuple[DataLoader, DataLoader]:
    """Yields (img: Tensor[B,3,224,224], label: Tensor[B]), labels 0..29."""
    raise NotImplementedError


def build_model(pretrained: bool, num_classes: int = 30) -> nn.Module:
    """Pretrained and random-init paths must produce byte-identical architectures."""
    raise NotImplementedError


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Keys: acc, f1_macro, f1_weighted."""
    raise NotImplementedError
