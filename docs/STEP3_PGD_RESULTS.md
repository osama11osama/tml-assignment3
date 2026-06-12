# Step 3 — PGD-AT: Results & Status

> **Last updated:** 2026-06-12  
> **Step 3 status: COMPLETE (80 epochs)** — leaderboard submit scored **0.575136**  
> **Team:** `team_XLVII` | **Cluster user:** `atml_team044`

---

## Verdict

| Criterion | Required | Current | OK? |
|-----------|----------|---------|-----|
| PGD adversarial training | Madry-style PGD-AT | **80 epochs** on cluster | **Yes** |
| Init | FGSM-AT warm-start | `fgsm_at_resnet18.pt` | Yes |
| Val unified (local, epoch 80) | Beat ERM 0.486 | **0.5693** | **Yes** |
| Leaderboard submit | Best model | **0.575136** (rank ~24/35) | **Yes** |
| Checkpoint for submit | `pgd_at_resnet18.pt` | On cluster + local copy | Yes |

Step 3 baseline PGD-AT is **complete**. Phase 4 (extend training) is optional — see `docs/CLUSTER_STEP4.md`.

---

## Training pipeline (completed)

| Step | Method | Epochs | Unified (local val) | LB submit? |
|------|--------|--------|---------------------|------------|
| 1 ERM | Standard training | 100 | 0.487 (clean only) | Yes — **0.486371** |
| 2 FGSM-AT | 1-step FGSM | 50 | 0.410 | No (worse than ERM) |
| 3 PGD-AT | PGD-7 train, PGD-20 eval | 80 | **0.5693** | Yes — **0.575136** |

---

## PGD-AT hyperparameters (cluster)

| Parameter | Value |
|-----------|-------|
| Architecture | **ResNet18** |
| Init | `fgsm_at_resnet18.pt` |
| ε | 8/255 |
| α | 2/255 |
| PGD train steps | **7** |
| PGD eval steps | 20 |
| Batch size (cluster) | 256 |
| LR | 0.01, milestones [50, 70], γ=0.1 |
| Chunks | 20 → 40 → 60 → 80 (4 × ~1 h Condor jobs) |

---

## Metrics by epoch (validation, PGD-20)

| Epoch | Clean | Robust | Unified (local) |
|-------|-------|--------|-----------------|
| 20 | 64.04% | 41.96% | 0.5300 |
| 40 | 66.60% | 40.26% | 0.5343 |
| 60 | 67.60% | 45.18% | 0.5639 |
| **80** | **67.88%** | **45.98%** | **0.5693** |

---

## Leaderboard submissions

| # | Checkpoint | Epochs | Server unified | Notes |
|---|------------|--------|----------------|-------|
| 1 | `baseline_erm_resnet18.pt` | 100 ERM | **0.486371** | Rank ~31/34 |
| 2 | `pgd_at_resnet18.pt` | 20 PGD-AT | 0.535790 | Intermediate |
| 3 | `pgd_at_resnet18.pt` | **80 PGD-AT** | **0.575136** | **Current best** |

---

## Parallel trial (epoch 20 only — not continued)

| Tag | PGD steps | Unified @20 | Winner? |
|-----|-----------|-------------|---------|
| `pgd_at` | 7 | 0.5300 | **Yes** (faster, same score) |
| `pgd_at_s10` | 10 | 0.5298 | No — abandoned |

---

## Artifacts

| Path | Description |
|------|-------------|
| `results/checkpoints/pgd_at_resnet18.pt` | **Submit file** (best unified @ epoch 80) |
| `results/runs/pgd_at_resnet18/last.pt` | Resume checkpoint (epoch 80) |
| `results/runs/pgd_at_resnet18/progress.json` | Full metrics + history |
| `results/runs/pgd_at_resnet18/best.pt` | Best val unified weights |

---

## Reproduce best result (summary)

```bash
# Cluster: 4 chunks from FGSM init (see docs/CLUSTER_STEP3.md)
bash scripts/cluster/run_pgd_at_chunk.sh 20   # after fgsm_at done
bash scripts/cluster/run_pgd_at_chunk.sh 40
bash scripts/cluster/run_pgd_at_chunk.sh 60
bash scripts/cluster/run_pgd_at_chunk.sh 80
```

Local eval:

```bash
python scripts/eval_model.py results/checkpoints/pgd_at_resnet18.pt --architecture resnet18
```

Submit: `_private/tools/launch_submit_gui.ps1` → `pgd_at_resnet18.pt`, model name `resnet18`.

---

## Next: Phase 4 (target higher LB)

See `docs/CLUSTER_STEP4.md` — resume `pgd_at` from epoch 80 → 120 with PGD-10 steps.

**Realistic target:** 0.58–0.62. **0.70** requires ResNet-34 + stronger methods (TRADES); top public LB ~0.63.
