# Step 1 — ERM Baseline: Results & Status

> Last updated: 2026-06-11  
> **Step 1 status: COMPLETE** — 100-epoch ERM finished; first leaderboard submit scored.

---

## Is Step 1 complete?

| Criterion | Required | Current | OK? |
|-----------|----------|---------|-----|
| ERM training script | `train_standard.py` with resume | Done | Yes |
| Full training schedule | 100 epochs, LR schedule 50/75 | **100 epochs** | **Yes** |
| Val clean accuracy | High enough for baseline (~60%+) | **97.28%** (val) | **Yes** |
| Robust acc documented | PGD-20 on val | **0.24%** (expected for ERM) | Yes |
| Checkpoint for submit | `baseline_erm_resnet18.pt` | Done | Yes |
| GPU training verified | `--device cuda` | Yes (RTX 5060) | Yes |
| Leaderboard submit | Optional for Step 1 | **Scored** — LB **0.486371** (rank ~31/34) | Yes |

**Verdict:** Step 1 is **complete**. ERM baseline is strong on clean data but **not robust** — that is the expected trade-off and motivates Step 2/3.

---

## GPU check (2026-06-11)

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| PyTorch | `2.11.0+cu128` |
| CUDA available | `True` |
| Training device | `cuda` (when `--device cuda` or default) |

If you see GPU activity but **no** `train_standard.py` running in a terminal, it may be Cursor, WhatsApp, Windows widgets, or another app — not necessarily this assignment.

**How to confirm Assignment 3 training is running:**
- Terminal shows `[TRAIN] Epoch X/100 | batch ...`
- `results/runs/erm_resnet18/progress.json` updates every `--save-every` epochs
- `last.pt` modification time changes

---

## Experiments log

### exp001 — Smoke test (2 epochs, no resume)

| Field | Value |
|-------|-------|
| Date | 2026-06-11 |
| Command | `python scripts/train_standard.py --device cuda --epochs 2 --batch-size 128` |
| Script version | Before resume/checkpoint refactor |
| Log | `results/logs/baseline_erm_resnet18.json` |

| Epoch | Train loss | Clean val | Robust val (PGD-20) | Unified |
|-------|------------|-----------|---------------------|---------|
| 1 | 1.872 | 52.40% | — | — |
| 2 | 1.182 | **55.34%** | 3.28% | 0.293 |

| Final (best checkpoint) | Value |
|-------------------------|-------|
| Best clean | **55.34%** |
| Final robust | 3.16% |
| Unified (est.) | **0.2925** |
| Time | ~0.5 min |

**Notes:** Confirms ERM pattern — decent clean, near-zero robust. Above 50% server gate but **not** a real baseline (only 2 epochs).

---

### exp002 — Resume test (epochs 1→2 fresh, then resume to 3)

| Field | Value |
|-------|-------|
| Date | 2026-06-11 |
| Commands | `train_standard.py --epochs 2 --save-every 1` then `--epochs 3 --resume` |
| Run dir | `results/runs/erm_resnet18/` |
| Log | `results/runs/erm_resnet18/progress.json` |

| Epoch | Train loss | Clean val | Robust val | Unified | LR |
|-------|------------|-----------|------------|---------|-----|
| 1 | 1.600 | 13.98% | — | — | 0.1 |
| 2 | 1.021 | 34.94% | 10.86% | 0.229 | 0.1 |
| 3 | 0.843 | **40.86%** | 6.18% | 0.235 | 0.1 |

| Final | Value |
|-------|-------|
| Best clean | 40.86% |
| Final robust | 6.22% |
| Unified | 0.2354 |
| Status in progress.json | `finished` (for 3 epochs only) |
| Time | ~0.9 min (resume segment) |

**Notes:**
- Resume from `last.pt` worked correctly.
- This run **overwrote** `results/checkpoints/baseline_erm_resnet18.pt` with a **worse** model than exp001 (40.9% vs 55.3% clean).
- **Do not submit** this checkpoint for leaderboard.

---

### exp003 — Full 100-epoch baseline (completed)

| Field | Value |
|-------|-------|
| Status | **Done** (2026-06-11) |
| Command | `python scripts/train_standard.py --device cuda --epochs 100 --resume --save-every 5 --batch-size 256` |
| Run dir | `results/runs/erm_resnet18/` |
| Training time | ~15 min (912 s session) |

| Metric | Local val (PGD-20) | Server (public LB) |
|--------|-------------------|---------------------|
| Clean | **97.28%** | (included in unified) |
| Robust | **0.24%** | (included in unified) |
| Unified / LB score | 0.4876 (local est.) | **0.486371** |
| Leaderboard rank | — | **31 / 34** (team_XLVII) |
| Submission ID | — | 2721 (success) |

**Notes:**
- ERM pattern confirmed: very high clean, near-zero robust — matches lecture trade-off curve.
- Server score ≈ local unified estimate → local PGD-20 eval is a reasonable sanity check.
- **Do not expect to climb the leaderboard with ERM alone** — top teams (~0.63) use adversarial training.

---

## Submit history (GUI)

| Attempt | File | Result |
|---------|------|--------|
| 1 | `template_sanity.pt` | Rejected (clean < 50%) |
| 2 | `baseline_erm_resnet18.pt` (3 epochs) | Rejected (clean < 50%) |
| 3 | `baseline_erm_resnet18.pt` (100 epochs) | **Scored** — LB 0.486371 |

---

## Artifacts on disk

| Path | Description |
|------|-------------|
| `results/runs/erm_resnet18/best.pt` | Best weights (97.28% clean val) |
| `results/runs/erm_resnet18/last.pt` | Resume checkpoint (epoch 100) |
| `results/runs/erm_resnet18/progress.json` | Full 100-epoch history |
| `results/checkpoints/baseline_erm_resnet18.pt` | Submit copy — **scored on LB** |
| `results/checkpoints/template_sanity.pt` | Random weights — do not submit |

---

## Hyperparameters (Step 1 config)

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 (`torchvision`, fc → 9 classes) |
| Optimizer | SGD, lr=0.1, momentum=0.9, weight decay=5e-4 |
| LR schedule | MultiStepLR milestones [50, 75], gamma=0.1 |
| Batch size | 256 (128 in smoke tests) |
| Augmentation | Random crop + pad 4, horizontal flip |
| Val split | 10% (5000 images) |
| PGD eval | eps=8/255, 20 steps, every 25 epochs + final |
| Input | float [0, 1], shape 3×32×32 |

---

## Interpretation (for report)

1. **ERM alone is insufficient** — 97% clean but ~0.2% robust; unified score ~0.49 (matches server).
2. **Trade-off is real** — improving robustness requires adversarial training (Steps 2–3).
3. **Server gate passed** — clean >> 50% after full training.
4. **Baseline role** — ERM checkpoint is a warm-start for PGD-AT, not the final model.

---

## Next action → Step 2 (FGSM-AT)

See `docs/STEP2_ROBUSTNESS_STRATEGY.md` for paper-backed plan and commands.
