"""Dataset with optional CIFAR-style augmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from src.data import load_train_arrays


class ImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, train: bool = False):
        self.images = images
        self.labels = labels
        self.train = train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.images[idx]
        y = self.labels[idx]
        if self.train:
            x = F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            i = torch.randint(0, 9, (1,)).item()
            j = torch.randint(0, 9, (1,)).item()
            x = x[:, i : i + 32, j : j + 32]
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=(2,))
        return x, y


def make_augmented_loaders(
    batch_size: int = 256,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
):
    images, labels = load_train_arrays()
    n_val = int(len(labels) * val_fraction)
    n_train = len(labels) - n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(labels), generator=generator)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_ds = ImageDataset(images[train_idx], labels[train_idx], train=True)
    val_ds = ImageDataset(images[val_idx], labels[val_idx], train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, n_train, n_val
