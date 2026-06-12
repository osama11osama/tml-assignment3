"""Load train.npz — matches official task_template.py preprocessing."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.paths import TRAIN_NPZ


def load_train_arrays(npz_path=None):
    """Return float images in [0,1] and long labels."""
    path = TRAIN_NPZ if npz_path is None else npz_path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/download_data.py"
        )
    data = np.load(path)
    images = torch.from_numpy(data["images"]).float() / 255.0
    labels = torch.from_numpy(data["labels"]).long()
    return images, labels


def make_dataset(npz_path=None) -> TensorDataset:
    images, labels = load_train_arrays(npz_path)
    return TensorDataset(images, labels)


def make_loaders(
    batch_size: int = 256,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
):
    dataset = make_dataset()
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
