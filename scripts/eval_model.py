"""Evaluate a checkpoint: clean + PGD robust accuracy on val split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import make_augmented_loaders
from src.eval_utils import accuracy, robust_accuracy
from src.model import make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--architecture", default="resnet18")
    parser.add_argument("--pgd-eps", type=float, default=8 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    _, val_loader, _, n_val = make_augmented_loaders(batch_size=256)

    model = make_model(args.architecture)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)

    clean = accuracy(model, val_loader, device)
    robust = robust_accuracy(model, val_loader, device, eps=args.pgd_eps, steps=args.pgd_steps)
    unified = 0.5 * clean + 0.5 * robust

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val samples: {n_val}")
    print(f"Clean accuracy:  {clean*100:.2f}%")
    print(f"Robust accuracy: {robust*100:.2f}%  (PGD-{args.pgd_steps}, eps={args.pgd_eps:.4f})")
    print(f"Unified score:   {unified:.4f}")


if __name__ == "__main__":
    main()
