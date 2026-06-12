"""Official-style template — load data, build model, save state_dict."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.paths import TRAIN_NPZ

# images: uint8 (N, 3, 32, 32), labels: int in [0, 8] — divide by 255 -> [0, 1]

npz_path = TRAIN_NPZ if TRAIN_NPZ.exists() else ROOT / "train.npz"
if not npz_path.exists():
    raise FileNotFoundError(
        f"train.npz not found at {TRAIN_NPZ}. Run: python scripts/download_data.py"
    )

data = np.load(npz_path)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

print("Dataset size:", len(dataset))
print("Image shape:", images.shape)
print("Label range:", labels.min().item(), "to", labels.max().item())

NUM_CLASSES = 9

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 32, 32))
print("Output shape:", out.shape)
assert out.shape == (1, NUM_CLASSES), out.shape

out_path = ROOT / "results" / "checkpoints" / "template_sanity.pt"
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out_path)
print("Saved sanity checkpoint:", out_path)
