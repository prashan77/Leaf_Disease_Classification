"""MobileNetV2 construction for the transfer-learning experiment.

Both experiment arms use the same architecture. The only difference is
whether the initial feature weights come from ImageNet or are randomized.
All layers remain trainable so the pretrained arm performs full fine-tuning.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model(pretrained: bool, num_classes: int = 30) -> nn.Module:
    """Build MobileNetV2 for PlantVillage classification.

    Args:
        pretrained: Load ImageNet weights when true; otherwise use random
            initialization.
        num_classes: Number of PlantVillage output classes.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)
    return model
