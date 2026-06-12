"""Model factory — must match official task_template.py / server expectations."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

from src.paths import NUM_CLASSES

ALLOWED_ARCHITECTURES = ("resnet18", "resnet34", "resnet50")


def make_model(architecture: str = "resnet18", num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build ResNet with fc replaced for 9 classes (official template style)."""
    arch = architecture.lower()
    factories = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }
    if arch not in factories:
        raise ValueError(f"architecture must be one of {ALLOWED_ARCHITECTURES}, got {architecture!r}")
    model = factories[arch](weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def assert_model_io(model: nn.Module, device: str = "cpu") -> None:
    import torch

    model = model.to(device)
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    if out.shape != (1, NUM_CLASSES):
        raise ValueError(f"expected output (1, {NUM_CLASSES}), got {tuple(out.shape)}")
