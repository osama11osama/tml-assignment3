"""Repository paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_NPZ = DATA_DIR / "train.npz"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

HF_REPO = "SprintML/tml26_task3"
NUM_CLASSES = 9
INPUT_SIZE = 32
