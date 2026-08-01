"""ResNet50 definition for pretrained and scratch experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50(nn.Module):
    """ResNet50 classifier with optional ImageNet initialization."""

    def __init__(self, pretrained: bool, num_classes: int = 30):
        """Create ResNet50 and replace its classifier for this dataset."""

        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = resnet50(weights=weights)
        input_features = self.network.fc.in_features
        self.network.fc = nn.Linear(input_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""

        return self.network(inputs)

    def num_params(self) -> int:
        """Return the number of trainable and non-trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters())
