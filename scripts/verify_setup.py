"""Verify Assignment 3 environment, data, and model I/O."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch

    from src.data import load_train_arrays, make_loaders
    from src.model import assert_model_io, make_model
    from src.paths import NUM_CLASSES, TRAIN_NPZ

    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    assert TRAIN_NPZ.exists(), f"Missing {TRAIN_NPZ} — run scripts/download_data.py"
    images, labels = load_train_arrays()
    assert images.shape[1:] == (3, 32, 32), images.shape
    assert labels.min() >= 0 and labels.max() < NUM_CLASSES, (labels.min(), labels.max())
    print(f"Data OK: {len(labels)} samples, shape {tuple(images.shape[1:])}")

    train_loader, val_loader = make_loaders(batch_size=128)
    x, y = next(iter(train_loader))
    assert x.shape[1:] == (3, 32, 32) and y.ndim == 1
    print(f"Loaders OK: train batches ~{len(train_loader)}, val ~{len(val_loader)}")

    for arch in ("resnet18", "resnet34", "resnet50"):
        model = make_model(arch)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert_model_io(model, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model OK: {arch} — {n_params:,} params, output (1, {NUM_CLASSES})")

    print("\nAll checks passed. Ready for baseline training (Stage 1).")


if __name__ == "__main__":
    main()
