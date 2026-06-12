"""Quick dataset exploration for train.npz."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_train_arrays
from src.paths import TRAIN_NPZ


def main() -> None:
    images, labels = load_train_arrays()
    arr = images.numpy()
    lbl = labels.numpy()

    print("=" * 60)
    print("Assignment 3 — train.npz summary")
    print("=" * 60)
    print(f"Path:       {TRAIN_NPZ}")
    print(f"Size (MB):  {TRAIN_NPZ.stat().st_size / 1e6:.1f}")
    print(f"N samples:  {len(labels)}")
    print(f"Image shape:{tuple(arr.shape[1:])}  (N, C, H, W)")
    print(f"Dtype:      float [0,1] after /255")
    print(f"Pixel min/max: {arr.min():.4f} / {arr.max():.4f}")
    print(f"Labels:     {lbl.min()} .. {lbl.max()}  ({len(np.unique(lbl))} classes)")

    counts = Counter(lbl.tolist())
    print("\nClass distribution:")
    for c in sorted(counts):
        bar = "#" * (counts[c] // 200)
        print(f"  class {c}: {counts[c]:5d}  {bar}")

    per_class = np.array([counts[i] for i in range(9)])
    print(f"\nBalance: min={per_class.min()}, max={per_class.max()}, ratio={per_class.min()/per_class.max():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
