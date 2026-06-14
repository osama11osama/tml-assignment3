# TRADES β=8 — Best Leaderboard Result (0.5824)

> **Date:** 2026-06-14  
> **Assignment:** TML SS2026 — Task `03-robustness`  
> **Team:** `team_XLVII` (`atml_team044` on cluster)  
> **Git tag:** `v1.2-submittable`  
> **Deadline:** 16 June 2026, 23:59

This document records **how we built the current best model**, what it scored, and how to reproduce it. Use it for the CMS report and for future agents.

---

## Result summary

| Field | Value |
|-------|-------|
| **Checkpoint** | `results/checkpoints/trades_b8_resnet18.pt` |
| **Architecture** | `resnet18` |
| **Method** | TRADES fine-tune with **β=8** on TRADES-R18 init |
| **Training** | 20 epochs (1× 1h Condor chunk) |
| **Server unified (LB)** | **0.582405** (~**0.5824**) |
| **Rank (after submit)** | ~20 / 48 |
| **Improvement vs previous best** | +0.006834 (0.575571 → 0.582405) |

### Local vs server (important)

| Metric | Local quick eval (GUI) | Cluster val (training) | **Server LB** |
|--------|------------------------|-------------------------|---------------|
| Clean | 72.34% | ~72% | (hidden test) |
| Robust | 44.82% (12-batch quick) | ~45% | PGD on hidden test |
| Unified | **0.5858** (estimate) | **0.5774** (cluster) | **0.5824** |

**Lesson:** Local GUI eval ranks models well but **overestimates** server score by ~0.003–0.005. Only submit when local unified is clearly above current LB (e.g. +0.004 margin).

---

## Full pipeline (how this beast was created)

```
Step 1  ERM R18 (100 ep)              → baseline_erm_resnet18.pt       LB 0.486371
Step 2  FGSM-AT R18 (50 ep)           → fgsm_at_resnet18.pt            warm-up only
Step 3  PGD-AT R18 (80 ep)            → pgd_at_resnet18.pt             LB 0.575136
Step 4  TRADES R18 β=6 (40 ep)        → trades_r18_resnet18.pt         LB 0.575571
Step 5  TRADES β=8 sweep (20 ep)      → trades_b8_resnet18.pt          LB 0.582405  ← BEST
```

Steps 1–4 are documented in `docs/SUBMISSION_SNAPSHOT.md` (previous best).  
**Step 5 is the breakthrough** — same backbone, higher TRADES β, shorter fine-tune from Step 4 weights.

---

## Step 5 — TRADES β=8 (detailed)

### Why β=8?

| Experiment | β | Init | Epochs | Local unified | Server LB |
|------------|---|------|--------|---------------|-----------|
| TRADES R18 (Step 4) | 6 | `pgd_at_resnet18.pt` | 40 | ~0.580 | 0.575571 |
| TRADES β=4 pilot | 4 | `trades_r18_resnet18.pt` | 20 | ~0.536 | not submitted (worse) |
| **TRADES β=8 pilot** | **8** | `trades_r18_resnet18.pt` | **20** | **0.5858** | **0.582405** |

β=6 on PGD init gave a small LB gain. β=4 from TRADES init **hurt** robustness too much.  
β=8 pushes the TRADES KL term harder → **better clean–robust balance on hidden test** without another 40-epoch run.

### Hyperparameters

| Parameter | TRADES R18 (Step 4) | **TRADES β=8 (Step 5)** |
|-----------|---------------------|-------------------------|
| Init | `pgd_at_resnet18.pt` | **`trades_r18_resnet18.pt`** |
| `trades_beta` | 6.0 | **8.0** |
| Epochs | 40 | **20** |
| Learning rate | 0.005 | **0.003** |
| PGD train steps | 10 | 10 |
| ε / α | 8/255, 2/255 | 8/255, 2/255 |
| Batch size | 128 | 128 |
| `robust_every` | 10 | 0 (eval at end only) |
| Optimizer | SGD momentum 0.9, wd 5e-4 | same |
| Tag / run dir | `trades_r18` | **`trades_b8`** |

Config file: `configs/trades_b8.yaml`

### Cluster job

Submitted via parallel tracks (Week 1 extension):

```bash
cd ~/tml26_task3
bash scripts/cluster/submit_parallel_tracks.sh max4
# includes: condor_submit scripts/cluster/condor/train_trades_b8_r18_chunk20.sub
```

Condor spec: `scripts/cluster/condor/train_trades_b8_r18_chunk20.sub`

Environment (key vars):

```
TML3_TAG=trades_b8
TML3_ARCH=resnet18
TML3_INIT=results/checkpoints/trades_r18_resnet18.pt
TML3_TRADES_BETA=8
TML3_LR=0.003
TML3_PGD_TRAIN_STEPS=10
TML3_FINAL_EPOCHS=20
```

Wrapper: `scripts/cluster/run_trades_chunk.sh 20`

Logs: `runlogs/trades_b8_chunk20_*.log`

### Pull checkpoint to Windows

```powershell
cd Assignment3
$R = "atml_team044@conduit2.hpc.uni-saarland.de:~/tml26_task3/results/checkpoints"
scp "${R}/trades_b8_resnet18.pt" results/checkpoints/
powershell -ExecutionPolicy Bypass -File scripts/cluster/sync_from_cluster.ps1 -ClusterUser atml_team044
```

Submit via GUI: `_private/tools/launch_submit_gui.ps1` → select `trades_b8_resnet18.pt` → architecture auto-detects **resnet18**.

---

## Score progression (all LB submissions)

| # | Model | Server unified | Δ vs prev best |
|---|-------|----------------|----------------|
| 1 | ERM R18 | 0.486371 | — |
| 2 | PGD-AT ep20 | 0.535790 | +0.049 |
| 3 | PGD-AT ep80 | 0.575136 | +0.039 |
| 4 | TRADES R18 β=6 | 0.575571 | +0.0004 |
| 5 | **TRADES β=8** | **0.582405** | **+0.0068** |

---

## Reproduce locally (from TRADES-R18 init)

Prerequisites: `data/train.npz`, `results/checkpoints/trades_r18_resnet18.pt`

```powershell
cd Assignment3
.\.venv\Scripts\Activate.ps1
python scripts/train_trades.py --device cuda `
  --init results/checkpoints/trades_r18_resnet18.pt `
  --tag trades_b8 --architecture resnet18 `
  --epochs 20 --batch-size 128 --lr 0.003 `
  --pgd-train-steps 10 --trades-beta 8.0 `
  --robust-every 0 --save-every 5
```

Eval:

```powershell
python scripts/eval_model.py results/checkpoints/trades_b8_resnet18.pt --architecture resnet18
python submission.py --validate-only results/checkpoints/trades_b8_resnet18.pt --model-name resnet18
```

---

## What did *not* help (same period)

| Track | Outcome |
|-------|---------|
| TRADES β=4 (20 ep) | Local unified ~0.536 — **do not submit** |
| PGD-AT extend 120 ep | Plateau ~0.57 — no LB gain |
| ERM R34 | High clean, robust ~0.2% — init only, not for LB |

---

## CMS report talking points

1. **Baseline ERM** fails on robustness (unified ~0.49).
2. **PGD-AT** is the main robust baseline (0.575 on server after 80 ep).
3. **TRADES β=6** on PGD init gives a small gain (0.575571) via better clean–robust trade-off.
4. **TRADES β=8** fine-tune from TRADES weights (20 ep) is the **best result (0.5824)** — higher β regularizes adversarial logits more strongly.
5. **β sweep** (4 vs 8) shows sensitivity; β=4 over-regularizes in the wrong direction for this init.
6. Cite: Madry et al. (PGD-AT), Zhang et al. (TRADES).

---

## Related files

| Path | Purpose |
|------|---------|
| `configs/trades_b8.yaml` | Reproducible hyperparameters |
| `scripts/train_trades.py` | TRADES training entry point |
| `scripts/cluster/run_trades_chunk.sh` | Condor chunk wrapper |
| `scripts/cluster/condor/train_trades_b8_r18_chunk20.sub` | 20-epoch β=8 job |
| `docs/SUBMISSION_SNAPSHOT.md` | CMS snapshot (points here) |
| `experiments/exp_notes.md` | Experiment log |
| `_private/tools/tml_submit_gui.py` | Submit GUI with arch auto-detect |

---

## Version tags

| Tag | Best LB | Checkpoint |
|-----|---------|------------|
| `v1.0-submittable` | 0.575136 | `pgd_at_resnet18.pt` |
| `v1.2-submittable` | **0.582405** | **`trades_b8_resnet18.pt`** |

---

*Previous best (TRADES R18 β=6): still valid fallback — see git tag history and `docs/SUBMISSION_SNAPSHOT.md`.*
