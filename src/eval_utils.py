"""Clean and robust accuracy evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.attacks import pgd


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def robust_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 20,
    max_batches: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        x_adv = pgd(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)
