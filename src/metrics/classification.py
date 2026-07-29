"""Classification metrics derived from a confusion matrix."""

from __future__ import annotations

import torch


def f1_from_confusion(confusion: torch.Tensor) -> tuple[float, float]:
    """Return macro and support-weighted F1."""

    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    predicted = confusion.sum(dim=0)
    support = confusion.sum(dim=1)
    precision = true_positive / predicted.clamp(min=1)
    recall = true_positive / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    total = support.sum().clamp(min=1)
    return f1.mean().item(), (f1 * support).sum().div(total).item()
