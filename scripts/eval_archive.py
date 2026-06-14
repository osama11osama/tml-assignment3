"""Evaluate every checkpoint in results/archive/ and update eval_summary.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import make_augmented_loaders
from src.eval_utils import accuracy, robust_accuracy
from src.model import make_model

ARCHIVE = ROOT / "results" / "archive"
SUMMARY = ARCHIVE / "eval_summary.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_meta(pt_path: Path) -> dict:
    meta_path = pt_path.with_suffix(".meta.json")
    if meta_path.exists():
        return _read_json(meta_path)
    stem = pt_path.stem
    arch = "resnet18"
    for candidate in ("resnet50", "resnet34", "resnet18"):
        if candidate in stem:
            arch = candidate
            break
    if stem.startswith("baseline_erm_r34"):
        tag = "erm_r34"
    elif stem.startswith("baseline_erm"):
        tag = "erm_r18"
    elif "_resnet" in stem:
        tag = stem.rsplit("_resnet", 1)[0]
    else:
        parts = stem.split("_")
        tag = parts[1] if len(parts) > 1 else "unknown"
    return {"architecture": arch, "tag": tag}


def eval_one(pt_path: Path, device: torch.device, pgd_steps: int, pgd_eps: float) -> dict:
    meta = load_meta(pt_path)
    arch = meta.get("architecture", "resnet18")
    _, val_loader, _, n_val = make_augmented_loaders(batch_size=256)

    model = make_model(arch)
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)

    clean = accuracy(model, val_loader, device)
    robust = robust_accuracy(model, val_loader, device, eps=pgd_eps, steps=pgd_steps)
    unified = 0.5 * clean + 0.5 * robust

    return {
        "checkpoint": str(pt_path.relative_to(ROOT)).replace("\\", "/"),
        "tag": meta.get("tag"),
        "architecture": arch,
        "last_epoch": meta.get("last_epoch"),
        "cluster_unified": meta.get("cluster_unified"),
        "local_clean": round(clean, 6),
        "local_robust": round(robust, 6),
        "local_unified": round(unified, 6),
        "val_samples": n_val,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-eval archived cluster checkpoints")
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-eps", type=float, default=8 / 255)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default=None, help="Only eval checkpoints whose meta tag matches")
    parser.add_argument(
        "--also-checkpoints",
        action="store_true",
        help="Also eval results/checkpoints/*.pt (GUI folder)",
    )
    args = parser.parse_args()

    archive = args.archive_dir
    if not archive.is_dir():
        print(f"No archive dir: {archive}")
        print("Run: powershell -File scripts/cluster/sync_from_cluster.ps1")
        sys.exit(1)

    device = torch.device(args.device)
    pts = sorted(archive.glob("*.pt"))
    if args.also_checkpoints:
        ckpt_dir = ROOT / "results" / "checkpoints"
        if ckpt_dir.is_dir():
            seen = {p.resolve() for p in pts}
            for p in sorted(ckpt_dir.glob("*.pt")):
                if p.resolve() not in seen:
                    pts.append(p)
            pts = sorted(pts, key=lambda p: str(p).lower())
    if args.tag:
        pts = [p for p in pts if load_meta(p).get("tag") == args.tag]

    if not pts:
        print(f"No .pt files in {archive}")
        sys.exit(0)

    results = []
    print(f"Evaluating {len(pts)} checkpoint(s) on {device}...")
    print("-" * 72)
    for pt in pts:
        row = eval_one(pt, device, args.pgd_steps, args.pgd_eps)
        results.append(row)
        cu = row["cluster_unified"]
        cu_s = f"{cu:.4f}" if cu is not None else "n/a"
        print(
            f"{row['tag']:12} ep{row['last_epoch']!s:>4} | "
            f"cluster_u={cu_s} | local clean={row['local_clean']*100:5.2f}% "
            f"robust={row['local_robust']*100:5.2f}% unified={row['local_unified']:.4f}"
        )
        print(f"  -> {row['checkpoint']}")

    summary_path = archive / "eval_summary.json"
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "pgd_steps": args.pgd_steps,
        "results": sorted(results, key=lambda r: r["local_unified"], reverse=True),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 72)
    best = summary["results"][0]
    print(f"Best local unified: {best['local_unified']:.4f} ({best['tag']}, {best['checkpoint']})")
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
