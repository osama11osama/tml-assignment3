# Master Plan — 20 Days to Maximum LB Score

> **Assumption:** 20 full days from 2026-06-12  
> **Current best LB:** **0.575136** (`pgd_at_resnet18.pt`, PGD-AT 80 ep)  
> **Public top:** ~**0.629**  
> **Realistic target:** **0.60–0.63** | **Stretch:** **0.65+**

---

## Executive summary

Your R18 + PGD-AT pipeline is **solved** — extending epochs (Phase 4) hit a **plateau** at ~0.57 local / 0.575 server.

To reach the top you need **three upgrades in parallel**, not more of the same:

| Upgrade | Why | Expected gain |
|---------|-----|---------------|
| **TRADES loss** | Better clean–robust Pareto frontier | +0.02–0.05 |
| **ResNet-34** | More capacity for both metrics | +0.02–0.04 |
| **Stronger adversary + schedule** | PGD-10/20 train, cosine LR, ERM init | +0.01–0.03 |

Run **4 parallel cluster tracks** (different tags), submit only winners. Use local eval + 60 min cooldown strategically.

---

## Score math (what “best ever” looks like)

```
Unified = 0.5 × clean + 0.5 × robust
```

| Target unified | Example (clean / robust) |
|----------------|--------------------------|
| **0.575** (you now) | 68% / 47% |
| **0.60** | 72% / 48% or 70% / 50% |
| **0.63** (#1 now) | 75% / 51% or 80% / 46% |
| **0.65** (stretch) | 78% / 52% |

**Key insight:** Top teams keep **clean ≥ 70–80%** while pushing **robust ≥ 45–50%**. TRADES is designed for exactly this trade-off.

---

## Architecture decision tree

```
                    Start
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    Track A       Track B       Track C
    R34 PGD-AT    R18 TRADES    R34 TRADES  ← highest ceiling
         │            │            │
         └────────────┼────────────┘
                      ▼
              Pick best val unified
                      │
                      ▼
         Track D: fine-tune winner (AWP / longer / PGD-20 train)
                      │
                      ▼
              Final LB submit + CMS
```

**Do not** continue R18 PGD-AT beyond epoch 120 — proven plateau.

---

## 20-day calendar

### Week 1 — Build weapons + parallel pilots (Days 1–7)

| Day | Local (Windows / code) | Cluster (GPU) | Deliverable |
|-----|------------------------|---------------|-------------|
| **1** | Freeze v1.0 baseline in git tag. Implement `train_trades.py`. Add configs. | — | TRADES script runs 1 epoch locally |
| **2** | Unit-test TRADES vs PGD on 3 epochs locally. Cluster condor subs for R34 ERM chunks. | Submit **R34 ERM** chunk 1 (epochs 20) tag `erm_r34` | R34 pipeline started |
| **3** | Implement `run_erm_chunk.sh` for R34 if missing. TRADES cluster script. | R34 ERM → 40. **R18 TRADES** pilot 30 ep tag `trades_r18` init from `pgd_at_resnet18.pt` | 2 jobs running |
| **4** | Compare pilots: local eval all checkpoints. Document in `experiments/exp_notes.md`. | R34 ERM → 60. TRADES pilot finishes. | First TRADES unified number |
| **5** | Implement AWP hook (optional module). Hyperparam grid YAML. | R34 ERM → 100. **R34 FGSM-AT** 30 ep tag `fgsm_r34` | R34 ERM complete |
| **6** | Code review + sync cluster. | **R34 PGD-AT** chunk 20 tag `pgd_r34` init fgsm_r34. **R34 TRADES** 40 ep tag `trades_r34` init erm_r34 | 2 R34 AT jobs |
| **7** | **Checkpoint review.** Pick leading track by val unified. Kill losers. | Continue winners only. First LB submit if pilot > 0.58. | Week 1 winner identified |

**Week 1 exit criteria:** At least one model with local unified **> 0.58**, or clear winner track for Week 2.

---

### Week 2 — Scale the winner (Days 8–14)

| Day | Focus | Cluster |
|-----|-------|---------|
| **8–9** | Full **R34 PGD-AT** 80 ep (4 chunks) OR **R34 TRADES** 80 ep | 4 × 1h jobs, tag `pgd_r34` or `trades_r34` |
| **10** | Hyperparam sweep (parallel, **different tags**): β∈{4,6,8}, PGD steps∈{7,10,15} | 2–3 short 20-ep trials on best track |
| **11** | Merge best hyperparams. Restart full 80–100 ep run if sweep finds +0.01 | Main production job |
| **12** | **LB submit #2** if unified > 0.575. Analyze server vs local gap. | — |
| **13–14** | **Track D:** fine-tune best checkpoint: TRADES 20 ep OR AWP 10 ep OR PGD-20 train steps | 1–2 refinement jobs |

**Week 2 exit criteria:** LB **≥ 0.60** or val unified **≥ 0.59** consistently.

---

### Week 3 — Final push + CMS (Days 15–20)

| Day | Focus |
|-----|-------|
| **15–16** | Final 100–120 epoch run on **single best config** (winner from Week 2). |
| **17** | **LB submit #3** (best checkpoint). Update `SUBMISSION_SNAPSHOT.md` → v2.0. |
| **18** | Optional last resort: **ResNet-50** if R34 TRADES < 0.60 (slow, only if cluster idle). |
| **19** | **CMS report** (2 pages): method, ablations table, LB score, references. |
| **20** | CMS ZIP (code + report, no weights). Final leaderboard check. Buffer day. |

---

## Four parallel tracks (technical spec)

### Track A — ResNet-34 PGD-AT (proven method, bigger model)

| Setting | Value |
|---------|-------|
| Tag | `pgd_r34` |
| Init chain | `erm_r34` → `fgsm_r34` → `pgd_r34` |
| Epochs | 80–100 |
| PGD train | 10 steps, ε=8/255, α=2/255 |
| Batch (cluster) | 128–256 (reduce if OOM) |
| LR | 0.01, milestones [50, 75] or cosine |
| **Target** | unified 0.58–0.61 |

### Track B — ResNet-18 TRADES (fast iteration)

| Setting | Value |
|---------|-------|
| Tag | `trades_r18` |
| Init | `pgd_at_resnet18.pt` (your best weights) |
| Loss | TRADES: natural CE + β × KL( clean ‖ adv ) |
| β | sweep {4, 6, 8} — start **6** |
| PGD train steps | 10 |
| Epochs | 40–80 fine-tune |
| **Target** | unified 0.58–0.62 without full retrain |

Reference: Zhang et al., *Theoretically Principled Trade-off between Robustness and Accuracy*.

### Track C — ResNet-34 TRADES (highest ceiling)

| Setting | Value |
|---------|-------|
| Tag | `trades_r34` |
| Init | `erm_r34` checkpoint (epoch 100) |
| Same TRADES as B | β=6, 80–100 epochs |
| **Target** | unified **0.60–0.63** |

### Track D — Refinement (after winner picked)

| Method | When |
|--------|------|
| **AWP** | After TRADES converges — perturbs weights during AT |
| **PGD-20 train** | If val robust lags clean by >20% |
| **ERM-direct init** | If clean too low (<65%) — skip FGSM, PGD from ERM |

---

## TRADES implementation checklist (Day 1–2)

New files to create:

```
scripts/train_trades.py       # main TRADES trainer
configs/trades_r18.yaml       # fine-tune from pgd_at
configs/trades_r34.yaml       # from erm_r34
src/train_utils.py            # add train_one_epoch_trades()
scripts/cluster/run_trades_chunk.sh
scripts/cluster/condor/train_trades_*.sub
```

Core loss (pseudocode):

```python
x_adv = pgd(model, x, y, steps=10, eps=8/255, alpha=2/255)
loss_nat = CE(model(x), y)
loss_rob = KL( log_softmax(model(x)/T), softmax(model(x_adv)/T) )  # batchmean
loss = loss_nat + beta * loss_rob
```

---

## ResNet-34 ERM cluster (prerequisite for Track A/C)

Reuse `train_standard.py --architecture resnet34 --tag erm_r34 --epochs 100`.

Chunk schedule (1h jobs):

```
20 → 40 → 60 → 80 → 100   tag=erm_r34
```

Then FGSM-AT 30–50 ep → PGD-AT 80 ep (same pattern as R18).

---

## Submission strategy (60 min cooldown)

| Submit # | When | Min local unified |
|----------|------|-------------------|
| Keep v1.0 | Done | 0.575 LB |
| #2 | Day 7 or 12 | > 0.58 |
| #3 | Day 17 | > best so far + 0.01 |

**Rules:**

1. Always **local eval full val PGD-20** before submit.
2. Log server vs local gap in `experiments/exp_notes.md`.
3. Never submit FGSM-only or epoch-20 partial models.
4. One production track at a time after Day 7.

---

## What NOT to do (20 days)

| Avoid | Why |
|-------|-----|
| More R18 PGD epochs (120+) | Plateau proven |
| Parallel jobs same tag/run dir | Corrupts `last.pt` |
| Submit every experiment | Cooldown waste |
| ResNet-50 first | 3× slower; R34 first |
| Input preprocessing tricks | Gradient masking — fails server |
| Ignore CMS until day 19 | Report takes 2 full days |

---

## Success metrics by day

| Day | Minimum | Good | Excellent |
|-----|---------|------|-----------|
| 7 | val > 0.57 | val > 0.58 | LB > 0.58 |
| 14 | LB > 0.58 | LB > 0.60 | LB > 0.61 |
| 17 | LB > 0.60 | LB > 0.62 | LB > 0.63 |
| 20 | CMS done | LB top 10 | LB top 3 |

---

## Immediate next actions (Day 1 — today)

### On Windows (code)

1. [ ] Ask agent to implement `train_trades.py` + configs
2. [ ] Local smoke test: 2 epochs TRADES on GPU
3. [ ] Git commit: `feat: TRADES trainer`

### On cluster

1. [ ] `cd ~/tml26_task3` — **always**
2. [ ] Start R34 ERM: need condor sub or manual `train_standard.py --architecture resnet34 --tag erm_r34 --epochs 20` in docker job
3. [ ] Do **not** delete `tml26_task3` — keep R18 checkpoints as TRADES init

### Parallel

- [ ] Start CMS outline (intro + method section) — 30 min today

---

## File / tag registry (avoid collisions)

| Tag | Architecture | Method | Status |
|-----|--------------|--------|--------|
| `pgd_at` | R18 | PGD-AT 120ep | **DONE** — LB 0.575 |
| `erm_r34` | R34 | ERM | TODO |
| `fgsm_r34` | R34 | FGSM-AT | TODO |
| `pgd_r34` | R34 | PGD-AT | TODO |
| `trades_r18` | R18 | TRADES fine-tune | TODO |
| `trades_r34` | R34 | TRADES | TODO |

**Rule:** One tag = one `results/runs/{tag}_{arch}/` directory. Never share.

---

## References for report

| Paper | Use |
|-------|-----|
| Goodfellow FGSM | Step 2 warm-up |
| Madry PGD-AT | Baseline AT |
| Zhang TRADES | Main upgrade |
| Athalye obfuscated gradients | Why no preprocessing |
| Wu AWP (optional) | Refinement |

---

*This plan supersedes Phase 4 extend-R18. Next implementation task: **TRADES + R34 ERM**.*
