"""A small convolutional baseline trained without pretrained weights."""

from __future__ import annotations

import torch
from torch import nn


class VanillaCNN(nn.Module):
    """Five convolution blocks followed by global average pooling."""

    def __init__(self, num_classes: int = 30, dropout: float = 0.3):
        """Create the CNN baseline for the requested number of classes."""

        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        def block(input_channels: int, output_channels: int) -> nn.Sequential:
            """Build one convolution, normalization, activation, and pooling block."""

            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
            block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""

        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.fc(self.dropout(pooled))

    def num_params(self) -> int:
        """Return the number of trainable and non-trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters())
