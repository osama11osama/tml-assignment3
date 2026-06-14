# Week 1 Cluster Runbook — TRADES + ResNet-34 ERM

> **Goal:** Start parallel tracks toward LB > 0.58 by day 7.  
> **Your action required:** sync code to cluster, run `submit_week1.sh`.

---

## What was added (code)

| File | Purpose |
|------|---------|
| `scripts/train_trades.py` | TRADES trainer |
| `scripts/cluster/run_erm_chunk.sh` | ERM in 1h chunks |
| `scripts/cluster/run_trades_chunk.sh` | TRADES in 1h chunks |
| `scripts/cluster/check_train_progress.sh` | Generic progress |
| `scripts/cluster/submit_week1.sh` | Launch both tracks |
| `configs/trades_r18.yaml` | TRADES fine-tune config |
| `configs/erm_r34.yaml` | R34 ERM config |

---

## YOUR actions (do today)

### Step 1 — Windows: sync to cluster

```powershell
cd "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment3"
powershell -ExecutionPolicy Bypass -File scripts/cluster/sync_to_cluster.ps1 -ClusterUser atml_team044
```

If `scp` fails (timeout): use VPN, or upload via OOD terminal + git pull on cluster.

### Step 2 — Cluster: fix line endings + submit Week 1

```bash
cd ~/tml26_task3
find scripts/cluster -type f \( -name '*.sh' -o -name '*.sub' \) -exec sed -i 's/\r$//' {} +
chmod +x scripts/cluster/*.sh

bash scripts/cluster/submit_week1.sh
```

This submits **2 jobs in parallel** (if 2 GPUs free):

| Job | Track | Target |
|-----|-------|--------|
| `train_erm_r34_chunk20.sub` | A — ERM R34 | epoch 20 |
| `train_trades_r18_chunk20.sub` | B — TRADES R18 | epoch 20 |

### Step 3 — Monitor

```bash
cd ~/tml26_task3
condor_q
tail -f runlogs/erm_r34_chunk20_*.log
# second terminal:
tail -f runlogs/trades_r18_chunk20_*.log
```

---

## After each chunk finishes

### Track A — ERM R34 (epochs 20 → 100)

```bash
# when condor_q empty and epoch < target:
condor_submit scripts/cluster/condor/train_erm_r34_chunk40.sub
# then 60, 80, 100
TML3_TAG=erm_r34 TML3_ARCH=resnet34 TML3_FINAL_EPOCHS=100 \
  TML3_CHUNKS=20,40,60,80,100 bash scripts/cluster/check_train_progress.sh
```

### Track B — TRADES R18 (epochs 20 → 40 pilot)

```bash
condor_submit scripts/cluster/condor/train_trades_r18_chunk40.sub
TML3_TAG=trades_r18 TML3_FINAL_EPOCHS=40 bash scripts/cluster/check_train_progress.sh
```

At epoch 40, check unified:

```bash
python3 -c "
import json
p=json.load(open('results/runs/trades_r18_resnet18/progress.json'))
u=p.get('final_unified_score', p.get('best_unified_score',0))
print('TRADES unified', u)
"
```

**Submit to LB only if unified > 0.575.**

---

## Optional — local smoke test (Windows GPU)

```powershell
cd Assignment3
.\.venv\Scripts\Activate.ps1
python scripts/train_trades.py --device cuda --epochs 2 --init results/checkpoints/pgd_at_resnet18.pt --tag trades_smoke --robust-every 0
```

---

## Week 1 schedule (days 1–7)

| Day | Cluster | Decision |
|-----|---------|----------|
| 1 | Submit week1 (ERM + TRADES) | — |
| 2–3 | ERM → 60, TRADES → 40 | Compare val unified |
| 4–5 | ERM → 100 | TRADES winner? extend to 80 |
| 6 | — | Pick leading track |
| 7 | LB submit if > 0.58 | Kill slower track |

---

## Tag registry (do not collide)

| Tag | Arch | Method |
|-----|------|--------|
| `pgd_at` | R18 | DONE — LB 0.575 |
| `erm_r34` | R34 | ERM 100ep |
| `trades_r18` | R18 | TRADES 40ep pilot |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| TRADES init missing | `pgd_at_resnet18.pt` must exist on cluster |
| OOM on R34 | `TML3_BATCH=128` in .sub environment |
| Same tag twice | Never run two jobs on same tag |
| progress shows epoch 0 | Use `check_train_progress.sh` not old pgd script |

See also: `docs/MASTER_PLAN_20D.md`
