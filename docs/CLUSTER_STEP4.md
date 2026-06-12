# Phase 4 — Extend PGD-AT (push toward higher LB)

> **Goal:** Improve unified score beyond **0.575** (current best LB).  
> **Stretch goal:** 0.70 — **unlikely** in remaining time; public top ~**0.63**. Realistic: **0.58–0.62**.  
> **Base model:** Resume `pgd_at` from epoch **80** (`results/runs/pgd_at_resnet18/last.pt`).

---

## Strategy

| Track | What | Expected gain | Time |
|-------|------|---------------|------|
| **A (start here)** | Resume R18 epoch 80 → **120**, PGD-**10** steps | +0.01–0.03 | ~2 h cluster |
| B | ResNet-34 full pipeline (ERM → FGSM → PGD) | +0.02–0.05 | 1–2 days |
| C | TRADES fine-tune (not implemented yet) | +0.02–0.04 | dev + train |

**Track A** uses your **best checkpoint** — no new tag needed; same `pgd_at` run dir.

---

## Track A — Quick start (cluster)

### Prerequisites

- Step 3 complete: `results/runs/pgd_at_resnet18/last.pt` at epoch 80
- Sync latest scripts: `scripts/cluster/sync_to_cluster.ps1` (Windows)

On cluster after sync:

```bash
cd ~/tml26_task3
find scripts/cluster -type f \( -name '*.sh' -o -name '*.sub' \) -exec sed -i 's/\r$//' {} +
chmod +x scripts/cluster/*.sh
sed -i 's/YOUR_CLUSTER_USER/atml_team044/g' scripts/cluster/condor/train_pgd_at_chunk*.sub
```

### Submit chunks (manual — recommended)

```bash
# Chunk 5: epochs 81 → 100, PGD-10
condor_submit scripts/cluster/condor/train_pgd_at_chunk100.sub
condor_q
tail -f runlogs/pgd_at_chunk100_*.log

# After job finishes:
condor_submit scripts/cluster/condor/train_pgd_at_chunk120.sub
```

Or use helper:

```bash
export TML3_FINAL_EPOCHS=120
bash scripts/cluster/submit_pgd_extend_next.sh
```

### Check progress

```bash
TML3_FINAL_EPOCHS=120 bash scripts/cluster/check_pgd_progress.sh
python3 -c "
import json
p=json.load(open('results/runs/pgd_at_resnet18/progress.json'))
ep=p.get('last_epoch') or p.get('epochs_completed',0)
u=p.get('final_unified_score', p.get('best_unified_score',0))
print(f'epoch {ep}/120 | unified {u:.4f}')
"
```

### After epoch 120

```powershell
# Windows
scp atml_team044@conduit2.hpc.uni-saarland.de:~/tml26_task3/results/checkpoints/pgd_at_resnet18.pt results/checkpoints/pgd_at_resnet18.pt
python scripts/eval_model.py results/checkpoints/pgd_at_resnet18.pt --architecture resnet18
# Submit via GUI only if unified > 0.575
```

---

## Settings (chunk 100 / 120)

| Env var | Value | Why |
|---------|-------|-----|
| `TML3_TAG` | `pgd_at` | Same run dir + resume |
| `TML3_PGD_TRAIN_STEPS` | `10` | Stronger adversary during training |
| `TML3_BATCH` | `256` | Cluster GPU tuning |
| `TML3_ROBUST_EVERY` | `0` | Robust eval only at chunk target |
| `TML3_FINAL_EPOCHS` | `120` | Extended schedule |

LR schedule continues from resumed optimizer state (already at 0.0001 after epoch 70).

---

## Submit rules

1. **Local eval first** — only submit if unified **> 0.575136**
2. **60 min cooldown** between scored leaderboard uploads
3. Update `docs/SUBMISSION_SNAPSHOT.md` when you beat 0.575 (bump to v1.1)

---

## If Track A plateaus (< 0.58)

1. Start **ResNet-34** ERM on cluster (`TML3_ARCH=resnet34`, new tags)
2. Request **TRADES** implementation in repo
3. Do not re-run `pgd_at_s10` or FGSM-only models

---

## Chunk map (full schedule)

```
Phase 3 (done):  20 → 40 → 60 → 80   (PGD-7)
Phase 4:         100 → 120            (PGD-10, resume from 80)
```
