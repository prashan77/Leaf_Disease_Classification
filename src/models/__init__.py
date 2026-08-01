"""Model definitions used by the experiment runners."""

from __future__ import annotations

from torch import nn

from src.models.mobilenet import MobileNetV2
from src.models.resnet import ResNet50
from src.models.vanilla_cnn import VanillaCNN

ARCHITECTURES = ("mobilenet_v2", "resnet50", "vanilla_cnn")

__all__ = ["ARCHITECTURES", "MobileNetV2", "ResNet50", "VanillaCNN", "build_model"]


def build_model(
    architecture: str,
    initialization: str,
    num_classes: int,
) -> nn.Module:
    """Build one of the supported classification models."""

    if architecture == "mobilenet_v2":
        if initialization not in {"pretrained", "scratch"}:
            raise ValueError(f"unsupported initialization: {initialization}")
        return MobileNetV2(
            pretrained=initialization == "pretrained",
            num_classes=num_classes,
        )

    if architecture == "resnet50":
        if initialization not in {"pretrained", "scratch"}:
            raise ValueError(f"unsupported initialization: {initialization}")
        return ResNet50(
            pretrained=initialization == "pretrained",
            num_classes=num_classes,
        )

    if architecture == "vanilla_cnn":
        if initialization != "scratch":
            raise ValueError("vanilla_cnn only supports scratch initialization")
        return VanillaCNN(num_classes=num_classes)

    raise ValueError(f"unsupported architecture: {architecture}")
