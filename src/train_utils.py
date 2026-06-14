"""Shared training utilities: progress, checkpoints, resume."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler


from src.attacks import fgsm, pgd


def log(msg: str, *, step: str | None = None) -> None:
    """Print with optional step prefix and immediate flush."""
    prefix = f"[{step}] " if step else ""
    print(f"{prefix}{msg}", flush=True)


def log_banner(title: str) -> None:
    log("=" * 60)
    log(title)
    log("=" * 60)


def pct(done: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * done / total:.1f}%"


def default_run_dir(tag: str, architecture: str) -> Path:
    from src.paths import RESULTS_DIR

    return RESULTS_DIR / "runs" / f"{tag}_{architecture}"


def progress_json_path(run_dir: Path) -> Path:
    return run_dir / "progress.json"


def last_ckpt_path(run_dir: Path) -> Path:
    return run_dir / "last.pt"


def best_ckpt_path(run_dir: Path) -> Path:
    return run_dir / "best.pt"


def save_progress_json(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_json_path(run_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_resume_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    best_clean: float,
    history: list[dict[str, Any]],
    args: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_clean": best_clean,
        "history": history,
        "args": args,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def save_best_submission(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    n_batches = len(loader)
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed = time.time() - t0
            log(
                f"Epoch {epoch}/{total_epochs} | batch {batch_idx}/{n_batches} "
                f"({pct(batch_idx, n_batches)}) | loss {loss.item():.4f} | "
                f"{elapsed:.0f}s elapsed",
                step="TRAIN",
            )

    return total_loss / max(n, 1)


def load_init_weights(path: Path, model: nn.Module, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Init checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} must be a state_dict or resume checkpoint with model_state")
    model.load_state_dict(state, strict=True)


def train_one_epoch_fgsm(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    eps: float,
    epoch: int,
    total_epochs: int,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    n_batches = len(loader)
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = fgsm(model, x, y, eps=eps)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed = time.time() - t0
            log(
                f"Epoch {epoch}/{total_epochs} | batch {batch_idx}/{n_batches} "
                f"({pct(batch_idx, n_batches)}) | adv loss {loss.item():.4f} | "
                f"{elapsed:.0f}s elapsed",
                step="TRAIN",
            )

    return total_loss / max(n, 1)


def train_one_epoch_pgd(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    eps: float,
    alpha: float,
    steps: int,
    epoch: int,
    total_epochs: int,
    log_every: int = 50,
) -> float:
    """Madry-style PGD adversarial training for one epoch."""
    model.train()
    total_loss = 0.0
    n = 0
    n_batches = len(loader)
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = pgd(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed = time.time() - t0
            log(
                f"Epoch {epoch}/{total_epochs} | batch {batch_idx}/{n_batches} "
                f"({pct(batch_idx, n_batches)}) | adv loss {loss.item():.4f} | "
                f"PGD-{steps} | {elapsed:.0f}s elapsed",
                step="TRAIN",
            )

    return total_loss / max(n, 1)


def train_one_epoch_trades(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    eps: float,
    alpha: float,
    steps: int,
    beta: float,
    epoch: int,
    total_epochs: int,
    log_every: int = 50,
) -> float:
    """TRADES: CE on clean + beta * KL(model(x_adv) || model(x))."""
    model.train()
    total_loss = 0.0
    n = 0
    n_batches = len(loader)
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = pgd(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_nat = model(x)
        loss_nat = criterion(logits_nat, y)

        with torch.no_grad():
            logits_clean = model(x)
        logits_adv = model(x_adv)
        loss_rob = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean, dim=1),
            reduction="batchmean",
        )
        loss = loss_nat + beta * loss_rob
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed = time.time() - t0
            log(
                f"Epoch {epoch}/{total_epochs} | batch {batch_idx}/{n_batches} "
                f"({pct(batch_idx, n_batches)}) | loss {loss.item():.4f} "
                f"(nat {loss_nat.item():.4f} rob {loss_rob.item():.4f}) | "
                f"TRADES PGD-{steps} beta={beta:g} | {elapsed:.0f}s elapsed",
                step="TRAIN",
            )

    return total_loss / max(n, 1)
