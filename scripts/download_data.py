"""Download train.npz and official templates from HuggingFace."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from src.paths import DATA_DIR, HF_REPO, ROOT as PROJECT_ROOT, TRAIN_NPZ

def download(dest_dir: Path | None = None) -> Path:
    dest = dest_dir or (PROJECT_ROOT / "hf_download")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_REPO} ...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(dest),
    )
    return dest


def install_files(download_dir: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src_npz = download_dir / "train.npz"
    if not src_npz.exists():
        raise FileNotFoundError(f"train.npz not found in {download_dir}")
    if TRAIN_NPZ.exists() and TRAIN_NPZ.stat().st_size == src_npz.stat().st_size:
        print(f"Already present: {TRAIN_NPZ}")
    else:
        shutil.copy2(src_npz, TRAIN_NPZ)
        print(f"Copied -> {TRAIN_NPZ}")

    # Official HF templates are kept in hf_download/ for reference only.
    # Repo versions of task_template.py / submission.py are maintained locally.


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Assignment 3 data from HuggingFace")
    parser.add_argument(
        "--dest",
        type=Path,
        default=PROJECT_ROOT / "hf_download",
        help="HF download cache directory",
    )
    parser.add_argument("--skip-download", action="store_true", help="Only copy from existing dest")
    args = parser.parse_args()

    if not args.skip_download:
        download(args.dest)
    elif not (args.dest / "train.npz").exists():
        raise SystemExit(f"No train.npz in {args.dest}. Run without --skip-download.")

    install_files(args.dest)
    print("Done.")


if __name__ == "__main__":
    main()
