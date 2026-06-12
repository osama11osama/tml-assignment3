"""
Step 2 — FGSM adversarial training (warm-up before PGD-AT).

Trains on FGSM adversarial examples (Goodfellow et al.) with optional init
from Step 1 ERM weights. Saves best checkpoint by validation unified score.

Features:
  - --init loads a .pt state_dict (e.g. baseline_erm_resnet18.pt)
  - --resume continues from results/runs/fgsm_at_<arch>/last.pt
  - Progress + resume checkpoints every --save-every epochs

Local:
  python scripts/train_fgsm_at.py --device cuda --init results/checkpoints/baseline_erm_resnet18.pt

Cluster (after setup — see docs/CLUSTER_STEP2.md):
  bash scripts/cluster/run_fgsm_at.sh --resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import make_augmented_loaders
from src.eval_utils import accuracy, robust_accuracy
from src.model import make_model
from src.train_utils import (
    best_ckpt_path,
    default_run_dir,
    last_ckpt_path,
    load_init_weights,
    load_resume_checkpoint,
    log,
    log_banner,
    progress_json_path,
    save_best_submission,
    save_progress_json,
    save_resume_checkpoint,
    train_one_epoch_fgsm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FGSM adversarial training (Step 2)")
    parser.add_argument("--architecture", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[30, 40])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--fgsm-eps", type=float, default=8 / 255, help="FGSM perturbation (train + should match eval eps)")
    parser.add_argument("--pgd-eps", type=float, default=8 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--robust-every", type=int, default=10, help="PGD val every N epochs (0 = final only)")
    parser.add_argument("--robust-eval-batches", type=int, default=0, help="0 = full val set for PGD eval")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="fgsm_at")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--init", type=Path, default=None, help="ERM or other state_dict to warm-start")
    parser.add_argument("--resume", action="store_true", help="Resume from <run-dir>/last.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Submit copy path (default: results/checkpoints/fgsm_at_<arch>.pt)",
    )
    return parser.parse_args()


def args_to_dict(args: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = args.run_dir or default_run_dir(args.tag, args.architecture)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = best_ckpt_path(run_dir)
    last_path = last_ckpt_path(run_dir)
    progress_path = progress_json_path(run_dir)
    submit_copy = args.output or (ROOT / "results" / "checkpoints" / f"{args.tag}_{args.architecture}.pt")

    log_banner("STEP 2 — FGSM adversarial training")
    log(f"Run directory: {run_dir}", step="SETUP")
    log(f"Device: {device}", step="SETUP")
    log(f"FGSM eps: {args.fgsm_eps:.6f} | PGD eval: {args.pgd_steps} steps @ {args.pgd_eps:.6f}", step="SETUP")
    log(f"Resume: {args.resume} | Init: {args.init or 'none'}", step="SETUP")
    log(f"Save every: {args.save_every} epoch(s)", step="SETUP")

    log("Loading data...", step="SETUP")
    train_loader, val_loader, n_train, n_val = make_augmented_loaders(
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )
    log(f"Train samples: {n_train} | Val samples: {n_val}", step="SETUP")

    log("Building model...", step="SETUP")
    model = make_model(args.architecture).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_clean = 0.0
    best_unified = 0.0
    history: list[dict] = []
    session_t0 = time.time()
    init_loaded = False

    if args.resume:
        if not last_path.is_file():
            log(f"No checkpoint at {last_path} — starting from scratch or --init.", step="RESUME")
        else:
            log(f"Loading {last_path} ...", step="RESUME")
            ckpt = load_resume_checkpoint(last_path, model, optimizer, scheduler, device)
            start_epoch = int(ckpt["epoch"]) + 1
            best_clean = float(ckpt.get("best_clean", 0.0))
            best_unified = float(ckpt.get("best_unified", 0.0))
            history = list(ckpt.get("history", []))
            init_loaded = True
            log(
                f"Resumed at epoch {start_epoch}/{args.epochs} | "
                f"best unified {best_unified:.4f} | best clean {best_clean*100:.2f}%",
                step="RESUME",
            )

    if not init_loaded and args.init is not None:
        log(f"Loading init weights from {args.init} ...", step="INIT")
        load_init_weights(args.init.resolve(), model, device)
        log("Init weights loaded.", step="INIT")

    if start_epoch > args.epochs:
        log("Training already complete for requested --epochs.", step="DONE")
        return

    log(f"Training epochs {start_epoch} -> {args.epochs}", step="SETUP")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        log(f"-- Epoch {epoch}/{args.epochs} --", step="EPOCH")

        train_loss = train_one_epoch_fgsm(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            eps=args.fgsm_eps,
            epoch=epoch,
            total_epochs=args.epochs,
            log_every=args.log_every,
        )
        scheduler.step()

        log("Clean validation...", step="EVAL")
        clean_acc = accuracy(model, val_loader, device)

        run_robust = epoch == args.epochs or (args.robust_every > 0 and epoch % args.robust_every == 0)
        robust_acc = None
        if run_robust:
            log(f"Robust validation (PGD-{args.pgd_steps})...", step="EVAL")
            robust_batches = args.robust_eval_batches if args.robust_eval_batches > 0 else None
            robust_acc = robust_accuracy(
                model,
                val_loader,
                device,
                eps=args.pgd_eps,
                steps=args.pgd_steps,
                max_batches=robust_batches,
            )

        unified = None if robust_acc is None else 0.5 * clean_acc + 0.5 * robust_acc
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "clean_acc": clean_acc,
            "robust_acc": robust_acc,
            "unified_score": unified,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_sec": time.time() - epoch_t0,
        }
        history.append(row)

        rob_s = f"{robust_acc*100:.2f}%" if robust_acc is not None else "n/a"
        uni_s = f"{unified:.4f}" if unified is not None else "n/a"
        log(
            f"Epoch {epoch}/{args.epochs} summary | adv loss {train_loss:.4f} | "
            f"clean {clean_acc*100:.2f}% | robust {rob_s} | unified {uni_s} | "
            f"lr {row['lr']:.5f} | {row['epoch_sec']:.0f}s",
            step="EPOCH",
        )

        saved_best = False
        if unified is not None and unified > best_unified:
            best_unified = unified
            best_clean = max(best_clean, clean_acc)
            save_best_submission(model, best_path)
            save_best_submission(model, submit_copy)
            saved_best = True
            log(
                f"New best unified {best_unified:.4f} (clean {clean_acc*100:.2f}%, robust {robust_acc*100:.2f}%) "
                f"-> {best_path.name}",
                step="SAVE",
            )
        elif unified is None and clean_acc > best_clean:
            best_clean = clean_acc
            save_best_submission(model, best_path)
            save_best_submission(model, submit_copy)
            saved_best = True
            log(f"New best clean {best_clean*100:.2f}% (robust not eval yet) -> {best_path.name}", step="SAVE")

        should_save = epoch % args.save_every == 0 or epoch == args.epochs or saved_best
        if should_save:
            save_resume_checkpoint(
                last_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_clean=best_clean,
                history=history,
                args=args_to_dict(args),
                extra={"best_unified": best_unified},
            )
            payload = {
                "method": "fgsm_adversarial_training",
                "status": "running" if epoch < args.epochs else "finished",
                "architecture": args.architecture,
                "last_epoch": epoch,
                "total_epochs": args.epochs,
                "init_checkpoint": str(args.init) if args.init else None,
                "best_clean_acc": best_clean,
                "best_unified_score": best_unified,
                "last_clean_acc": clean_acc,
                "last_robust_acc": robust_acc,
                "run_dir": str(run_dir),
                "best_checkpoint": str(best_path),
                "resume_checkpoint": str(last_path),
                "submit_copy": str(submit_copy),
                "history": history,
            }
            save_progress_json(run_dir, payload)
            log(f"Resume checkpoint -> {last_path.name} | progress -> {progress_path.name}", step="SAVE")

    log("Final eval on best checkpoint...", step="EVAL")
    best_state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state, strict=True)
    final_clean = accuracy(model, val_loader, device)
    final_robust = robust_accuracy(model, val_loader, device, eps=args.pgd_eps, steps=args.pgd_steps)
    final_unified = 0.5 * final_clean + 0.5 * final_robust

    summary = {
        "method": "fgsm_adversarial_training",
        "status": "finished",
        "architecture": args.architecture,
        "epochs_completed": args.epochs,
        "init_checkpoint": str(args.init) if args.init else None,
        "best_clean_acc": best_clean,
        "best_unified_score": best_unified,
        "final_clean_acc": final_clean,
        "final_robust_acc": final_robust,
        "final_unified_score": final_unified,
        "best_checkpoint": str(best_path),
        "resume_checkpoint": str(last_path),
        "submit_copy": str(submit_copy),
        "elapsed_sec": time.time() - session_t0,
        "history": history,
    }
    save_progress_json(run_dir, summary)

    log_banner("TRAINING COMPLETE")
    log(f"Best unified:    {best_unified:.4f}", step="DONE")
    log(f"Final clean:     {final_clean*100:.2f}%", step="DONE")
    log(f"Final robust:    {final_robust*100:.2f}%", step="DONE")
    log(f"Final unified:   {final_unified:.4f}", step="DONE")
    log(f"Submit file:     {submit_copy}", step="DONE")
    log(f"Resume with:     python scripts/train_fgsm_at.py --resume --device cuda", step="DONE")
    log(f"Total time:      {(time.time() - session_t0)/60:.1f} min", step="DONE")


if __name__ == "__main__":
    main()
