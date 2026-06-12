"""
Upload a robust classifier state dict to the course server.

Environment:
  TML_API_KEY — 32-char hash from CMS (also read from .env)

Examples:
  python submission.py --validate-only results/checkpoints/model.pt
  python submission.py results/checkpoints/model.pt --model-name resnet18
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import ALLOWED_ARCHITECTURES, assert_model_io, make_model

BASE_URL = "http://34.63.153.158"
TASK_ID = "03-robustness"


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def load_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TML_API_KEY", "").strip()
    if not key:
        die(
            "TML_API_KEY is not set.\n"
            "  Add to .env:  TML_API_KEY=your_32_char_key\n"
            '  Or: $env:TML_API_KEY="your_32_char_key"',
            2,
        )
    return key


def validate_checkpoint(model_path: Path, model_name: str) -> None:
    if not model_path.is_file():
        die(f"File not found: {model_path}")
    if model_name not in ALLOWED_ARCHITECTURES:
        die(f"model-name must be one of {ALLOWED_ARCHITECTURES}, got {model_name!r}")

    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        die("Checkpoint must be a state_dict (torch.save(model.state_dict(), path))")

    model = make_model(model_name)
    model.load_state_dict(state, strict=True)
    assert_model_io(model, device="cpu")
    print(f"Local validation OK: {model_path} ({model_name})")


def submit(model_path: Path, model_name: str, api_key: str) -> None:
    with open(model_path, "rb") as f:
        files = {"file": (model_path.name, f, "application/x-pytorch")}
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": api_key},
            files=files,
            data={"model_name": model_name},
            timeout=120,
        )

    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    if not resp.ok:
        print(f"Submission error (HTTP {resp.status_code}):", body, file=sys.stderr)
        raise SystemExit(1)

    print("Successfully submitted.")
    print("Server response:", body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit robust model to TML leaderboard")
    parser.add_argument(
        "model_path",
        nargs="?",
        type=Path,
        default=ROOT / "results" / "checkpoints" / "best.pt",
        help="Path to .pt state_dict",
    )
    parser.add_argument(
        "--model-name",
        default="resnet18",
        choices=ALLOWED_ARCHITECTURES,
        help="Architecture matching the checkpoint",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check file locally without uploading",
    )
    args = parser.parse_args()

    validate_checkpoint(args.model_path.resolve(), args.model_name)
    if args.validate_only:
        return

    api_key = load_api_key()
    submit(args.model_path.resolve(), args.model_name, api_key)


if __name__ == "__main__":
    main()
