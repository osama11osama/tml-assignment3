# Submission Snapshot — Best Leaderboard Result

> **Last updated:** 2026-06-14  
> **Assignment:** TML SS2026 — Task `03-robustness`  
> **Team:** `team_XLVII` | **Deadline:** 16 June 2026, 23:59  
> **Git tag:** `v1.2-submittable`

This document describes the **current best leaderboard result** for the CMS report and reproduction.

**Full story (β=8 sweep):** `docs/TRADES_B8_RESULTS.md`

---

## Best model (leaderboard) — **current**

| Field | Value |
|-------|-------|
| **Checkpoint** | `results/checkpoints/trades_b8_resnet18.pt` |
| **Architecture** | `resnet18` |
| **Method** | **TRADES β=8** fine-tune on TRADES-R18 init |
| **Training** | 20 epochs; init `trades_r18_resnet18.pt`; PGD-10 train; β=**8**; lr=0.003 |
| **Server unified score** | **0.582405** (~**0.5824**) |
| **Rank (approx.)** | ~20 / 48 |

---

## How this model was created (full pipeline)

```
Step 1  ERM R18 (100 ep)          → baseline_erm_resnet18.pt     LB 0.486
Step 2  FGSM-AT R18 (50 ep)       → fgsm_at_resnet18.pt          warm-up
Step 3  PGD-AT R18 (80 ep)        → pgd_at_resnet18.pt           LB 0.575136
Step 4  TRADES R18 β=6 (40 ep)    → trades_r18_resnet18.pt       LB 0.575571
Step 5  TRADES β=8 (20 ep)        → trades_b8_resnet18.pt        LB 0.582405  ← BEST
```

### Step 5 — TRADES β=8 (current best)

| Hyperparameter | Value |
|----------------|-------|
| Init checkpoint | `trades_r18_resnet18.pt` (TRADES β=6, 40 epochs) |
| Loss | TRADES: natural CE + β·KL(f(x), f(x_adv)) |
| β (trades_beta) | **8.0** |
| PGD train steps | **10** (ε=8/255, α=2/255) |
| Epochs | **20** |
| Batch size | 128 |
| Learning rate | **0.003** |
| Cluster | 1× 1h Condor chunk |

**Cluster:**

```bash
cd ~/tml26_task3
condor_submit scripts/cluster/condor/train_trades_b8_r18_chunk20.sub
# or: bash scripts/cluster/submit_parallel_tracks.sh max4
```

**Scripts / configs:**

- `scripts/train_trades.py`
- `scripts/cluster/run_trades_chunk.sh`
- `configs/trades_b8.yaml`
- Condor: `train_trades_b8_r18_chunk20.sub`

**Pull + submit:**

```powershell
scp atml_team044@conduit2.hpc.uni-saarland.de:~/tml26_task3/results/checkpoints/trades_b8_resnet18.pt results/checkpoints/
powershell -File _private/tools/launch_submit_gui.ps1
# architecture: resnet18 (auto-detected)
```

---

## Scores (local vs server)

| Model | Local clean | Local robust | Local unified (quick) | **Server LB** |
|-------|-------------|--------------|------------------------|---------------|
| ERM R18 | 97.28% | 0.24% | ~0.487 | 0.486371 |
| PGD-AT R18 | 69.44% | 46.94% | ~0.581 | 0.575136 |
| TRADES R18 β=6 | 74.24% | 41.76% | ~0.580 | 0.575571 |
| **TRADES β=8** | **72.34%** | **44.82%** | **0.5858** (est.) | **0.582405** |

Local quick eval **overestimates** server by ~0.003 — use only for ranking models, not exact LB prediction.

---

## Score progression (submissions)

| # | Model | Server unified |
|---|-------|----------------|
| 1 | ERM baseline | 0.486371 |
| 2 | PGD-AT (epoch 20) | 0.535790 |
| 3 | PGD-AT (epoch 80) | 0.575136 |
| 4 | TRADES R18 β=6 | 0.575571 |
| 5 | **TRADES β=8** | **0.582405** ← **cite this in CMS** |

---

## Reproduce Step 5 locally

Prerequisites: `data/train.npz`, `results/checkpoints/trades_r18_resnet18.pt`

```powershell
python scripts/train_trades.py --device cuda `
  --init results/checkpoints/trades_r18_resnet18.pt `
  --tag trades_b8 --architecture resnet18 `
  --epochs 20 --batch-size 128 --lr 0.003 `
  --pgd-train-steps 10 --trades-beta 8.0 `
  --robust-every 0 --save-every 5
```

---

## Previous bests (fallback)

| Tag | Checkpoint | Method | Server LB |
|-----|------------|--------|-----------|
| `v1.2-submittable` | `trades_b8_resnet18.pt` | TRADES β=8 | **0.582405** |
| `v1.0-submittable` | `pgd_at_resnet18.pt` | PGD-AT 80 ep | 0.575136 |
| — | `trades_r18_resnet18.pt` | TRADES β=6 | 0.575571 |

---

## CMS ZIP checklist

**Include:**

- [ ] Source code (`src/`, `scripts/`, `configs/`)
- [ ] `README.md` + **this file** + `docs/TRADES_B8_RESULTS.md`
- [ ] **2-page PDF report** (ICLR template) — ⚠️ **not in repo yet**
- [ ] Matriculation number + CMS team ID in report

**Do not include:** `data/train.npz`, `.pt` weights, `.env`, `.venv/`

---

## Version history

| Version | Date | Best LB | Tag | Notes |
|---------|------|---------|-----|-------|
| v1.0 | 2026-06-12 | 0.575136 | `v1.0-submittable` | PGD-AT R18 |
| v1.1 | 2026-06-14 | 0.575571 | — | TRADES R18 β=6 |
| **v1.2** | **2026-06-14** | **0.582405** | **`v1.2-submittable`** | **TRADES β=8 — current best** |

---

*Cluster runbook: `docs/CLUSTER_WEEK1.md` | β=8 details: `docs/TRADES_B8_RESULTS.md`*
