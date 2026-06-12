# Step 3 — HPC Cluster Runbook (PGD-AT / Madry)

> Init from **FGSM-AT** (`fgsm_at_resnet18.pt`). Goal: unified **> 0.55** (beat ERM LB 0.486).

---

## 1-hour job limit (use this)

Saarland Condor allows **~1 hour per GPU job**. Full PGD-AT (80 epochs) needs **4 chained jobs** — same quality, split with `--resume`:

| Job | Target epoch | Condor file |
|-----|--------------|-------------|
| 1 | 20 | `train_pgd_at_chunk20.sub` |
| 2 | 40 | `train_pgd_at_chunk40.sub` |
| 3 | 60 | `train_pgd_at_chunk60.sub` |
| 4 | 80 | `train_pgd_at_chunk80.sub` |

Each job: `+MaxRuntime = 3600`, soft stop at 55 min, checkpoint every 3 epochs.

### Upload chunk scripts (once)

```powershell
scp -r scripts/cluster atml_team044@conduit2.hpc.uni-saarland.de:~/tml26_task3/scripts/
```

### On cluster — first chunk OR after a killed long job

```bash
cd ~/tml26_task3
find scripts/cluster -type f \( -name '*.sh' -o -name '*.sub' \) -exec sed -i 's/\r$//' {} +
chmod +x scripts/cluster/*.sh
sed -i 's/YOUR_CLUSTER_USER/atml_team044/g' scripts/cluster/condor/train_pgd_at*.sub

bash scripts/cluster/check_pgd_progress.sh
bash scripts/cluster/submit_pgd_next.sh
```

After each job finishes (or is killed at 1h), run **`submit_pgd_next.sh`** again until status `finished`.

### If job 49473 (long 80-epoch) is still running

```bash
condor_rm 49473.0
bash scripts/cluster/check_pgd_progress.sh
bash scripts/cluster/submit_pgd_next.sh
```

Progress is kept in `results/runs/pgd_at_resnet18/last.pt` (saved every 3 epochs).

---

## GPU optimization (already in chunk jobs)

Each chunk job now uses the GPU more efficiently:

| Setting | Value | Why |
|---------|-------|-----|
| `TML3_BATCH` | **256** | ~2× throughput vs 128 on same GPU |
| `TML3_NUM_WORKERS` | **4** | parallel CPU data loading |
| `TML3_ROBUST_EVERY` | **0** | skip slow PGD-20 eval mid-chunk; eval at chunk target only |
| `cudnn.benchmark` | on | faster convolutions on fixed input size |

Expect **~25–35 epochs per 1h chunk** instead of ~20.

---

## Option A — Auto chain (1 GPU, hands-free)

Runs all 4 chunks back-to-back without manual resubmit:

```bash
bash scripts/cluster/submit_pgd_chain.sh
condor_q -dagman
```

---

## Option B — Parallel GPU trials (2 GPUs at once)

Compare **PGD-7** vs **PGD-10** for the first 20 epochs on **two cluster GPUs simultaneously**:

```bash
bash scripts/cluster/submit_pgd_parallel_trial.sh
# after ~1h:
bash scripts/cluster/compare_pgd_trials.sh
bash scripts/cluster/submit_pgd_next.sh
# if s10 won: TML3_TAG=pgd_at_s10 bash scripts/cluster/submit_pgd_next.sh
```

**Note:** You cannot parallelize chunks 20→40→60→80 of the *same* model — each chunk needs the previous checkpoint. Parallel = different configs only.

---

## Option C — Cluster + local GPU in parallel

While cluster runs PGD chunks, use your **RTX 5060** locally for eval or hyperparameter search:

```powershell
cd Assignment3
.venv\Scripts\activate
python scripts/eval_model.py --checkpoint results/checkpoints/fgsm_at_resnet18.pt
```

---

## Quick start (legacy single job — not for 1h limit)

### A) From Windows — upload new code + FGSM checkpoint

```powershell
cd "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment3"
powershell -File scripts/cluster/sync_to_cluster.ps1 -ClusterUser atml_team044
```

Or upload only scripts:

```powershell
scp -r scripts src configs atml_team044@conduit2.hpc.uni-saarland.de:~/tml26_task3/
scp results/checkpoints/fgsm_at_resnet18.pt `
  atml_team044@conduit2.hpc.uni-saarland.de:~/tml26_task3/results/checkpoints/
```

### B) On cluster (SSH)

```bash
cd ~/tml26_task3

find scripts/cluster -type f \( -name '*.sh' -o -name '*.sub' \) -exec sed -i 's/\r$//' {} +
chmod +x scripts/cluster/*.sh

bash scripts/cluster/verify_setup.sh

sed -i 's/YOUR_CLUSTER_USER/atml_team044/g' scripts/cluster/condor/train_pgd_at*.sub
mkdir -p runlogs

condor_submit scripts/cluster/condor/train_pgd_at.sub
condor_q
```

### C) Monitor

```bash
tail -f ~/tml26_task3/runlogs/pgd_at_*.log
cat ~/tml26_task3/results/runs/pgd_at_resnet18/progress.json
```

If job stops: `condor_submit scripts/cluster/condor/train_pgd_at_resume.sub`

---

## Settings (quality-first)

| Parameter | Value | Why |
|-----------|-------|-----|
| Init | `fgsm_at_resnet18.pt` | Step 2 warm-start |
| Train attack | PGD-**7**, eps=8/255, alpha=2/255 | Madry AT (not 20 — too slow) |
| Eval | PGD-**20** full val | Strong local metric |
| Epochs | **80** | Robustness needs time |
| LR | 0.01, milestones 50/70 | Standard AT schedule |
| Batch | 128 | Stable on cluster GPU |
| Best checkpoint | max **unified** score | Matches server metric |

**Expected runtime:** ~3–6 hours on cluster GPU.

---

## After training

```bash
grep "TRAINING COMPLETE" runlogs/pgd_at_*.log
cat results/runs/pgd_at_resnet18/progress.json
```

Download to PC:

```powershell
scp atml_team044@conduit2.hpc.uni-saarland.de:~/tml26_task3/results/checkpoints/pgd_at_resnet18.pt `
  results/checkpoints/

python scripts/eval_model.py results/checkpoints/pgd_at_resnet18.pt --device cuda
```

**Submit only if** unified **> 0.486** (your ERM LB score).

---

## Local run (laptop)

```powershell
python scripts/train_pgd_at.py --device cuda `
  --init results/checkpoints/fgsm_at_resnet18.pt `
  --epochs 80 --save-every 5
```

---

## Files

| File | Role |
|------|------|
| `scripts/train_pgd_at.py` | Main training |
| `configs/pgd_at.yaml` | Reference hyperparameters |
| `scripts/cluster/run_pgd_at.sh` | Cluster wrapper |
| `scripts/cluster/condor/train_pgd_at.sub` | Condor submit |
| `scripts/cluster/condor/train_pgd_at_resume.sub` | Resume job |

See also: `docs/STEP2_ROBUSTNESS_STRATEGY.md`
