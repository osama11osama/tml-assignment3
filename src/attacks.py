"""FGSM / PGD attacks for local robustness evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn


def _clamp_perturbation(x: torch.Tensor, x_adv: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.clamp(x_adv - x, -eps, eps)
    return torch.clamp(x + delta, 0.0, 1.0)


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    model.eval()
    x_adv = x.detach().clone()
    x_adv.requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad = x_adv.grad.detach()
    return _clamp_perturbation(x, x_adv + eps * grad.sign(), eps)


def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    model.eval()
    x_adv = x.detach().clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = _clamp_perturbation(x, x_adv, eps)
    return x_adv.detach()
