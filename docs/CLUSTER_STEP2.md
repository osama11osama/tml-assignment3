# Step 2 — HPC Cluster Runbook (FGSM-AT)

> Saarland University GPU cluster (`conduit2.hpc.uni-saarland.de`) via HTCondor + Docker.  
> Same pattern as Assignment 1 & 2. Reference: course `HPC_Cluster_Guide.pdf`.

---

## Overview

| Item | Value |
|------|-------|
| Script | `scripts/train_fgsm_at.py` |
| Cluster wrapper | `scripts/cluster/run_fgsm_at.sh` |
| Condor job | `scripts/cluster/condor/train_fgsm_at.sub` |
| Init weights | `results/checkpoints/baseline_erm_resnet18.pt` (Step 1 ERM) |
| Output | `results/checkpoints/fgsm_at_resnet18.pt` |
| Run dir | `results/runs/fgsm_at_resnet18/` |
| Expected time | ~2–4 h (50 epochs, batch 128) |

---

## A) One-time setup on cluster (SSH)

```bash
ssh YOUR_CLUSTER_USER@conduit2.hpc.uni-saarland.de

mkdir -p ~/tml26_task3/{data,results/checkpoints,results/runs,runlogs,src,scripts,configs}
cd ~/tml26_task3

# Download training data (~127 MB)
bash scripts/cluster/cluster_download_data.sh
# Or manual:
# wget -nc -O data/train.npz \
#   "https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz"
```

Replace `YOUR_CLUSTER_USER` with your CMS cluster login (not necessarily `team_XLVII`).

---

## B) Upload code + ERM checkpoint from PC

**Option 1 — PowerShell sync script (recommended):**

```powershell
cd "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment3"
powershell -File scripts/cluster/sync_to_cluster.ps1 -ClusterUser YOUR_CLUSTER_USER
```

**Option 2 — Manual scp:**

```powershell
$U = "YOUR_CLUSTER_USER"
$H = "conduit2.hpc.uni-saarland.de"
$R = "${U}@${H}:~/tml26_task3"
$LOCAL = "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment3"

scp -r "$LOCAL\src" "$LOCAL\scripts" "$LOCAL\configs" "$R/"
scp "$LOCAL\results\checkpoints\baseline_erm_resnet18.pt" "${R}/results/checkpoints/"
```

**Option 3 — Git clone on cluster (code only):**

```bash
git clone git@github.com:osama11osama/tml-assignment3.git ~/tml26_task3_code
cp -r ~/tml26_task3_code/{src,scripts,configs} ~/tml26_task3/
# Still need: data/train.npz + baseline_erm_resnet18.pt
```

---

## C) Fix line endings (if scripts fail with `\r`)

```bash
cd ~/tml26_task3
sed -i 's/\r$//' scripts/cluster/*.sh
chmod +x scripts/cluster/*.sh
```

---

## D) Verify before GPU submit

```bash
cd ~/tml26_task3
bash scripts/cluster/verify_setup.sh
```

Must show `OK data/train.npz` and `OK baseline_erm_resnet18.pt`.

---

## E) Submit GPU job (Condor)

```bash
cd ~/tml26_task3
mkdir -p runlogs

# Edit username in condor file (once):
sed -i 's/YOUR_CLUSTER_USER/'"$USER"'/g' scripts/cluster/condor/train_fgsm_at.sub
sed -i 's/YOUR_CLUSTER_USER/'"$USER"'/g' scripts/cluster/condor/train_fgsm_at_resume.sub

condor_submit scripts/cluster/condor/train_fgsm_at.sub
condor_q
```

**Monitor:**

```bash
condor_q
tail -f runlogs/fgsm_at.*.out
cat results/runs/fgsm_at_resnet18/progress.json
```

**If job stops early (walltime):**

```bash
condor_submit scripts/cluster/condor/train_fgsm_at_resume.sub
```

Resume reads `results/runs/fgsm_at_resnet18/last.pt` automatically.

---

## F) Interactive GPU session (alternative to Condor)

If you have an interactive GPU allocation:

```bash
cd ~/tml26_task3
bash scripts/cluster/run_fgsm_at.sh
# Or resume:
bash scripts/cluster/run_fgsm_at.sh --resume
```

---

## G) Pull results back to PC

```powershell
$U = "YOUR_CLUSTER_USER"
scp "${U}@conduit2.hpc.uni-saarland.de:~/tml26_task3/results/checkpoints/fgsm_at_resnet18.pt" `
  "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment3\results\checkpoints\"

scp "${U}@conduit2.hpc.uni-saarland.de:~/tml26_task3/results/runs/fgsm_at_resnet18/progress.json" `
  "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment3\results\runs\fgsm_at_resnet18\"
```

**Local eval on PC:**

```powershell
python scripts/eval_model.py results/checkpoints/fgsm_at_resnet18.pt --device cuda
```

---

## H) Local run (laptop GPU — same script)

```powershell
cd Assignment3
.\.venv\Scripts\Activate.ps1
python scripts/train_fgsm_at.py --device cuda `
  --init results/checkpoints/baseline_erm_resnet18.pt `
  --epochs 50 --save-every 5
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `train.npz not found` | `bash scripts/cluster/cluster_download_data.sh` |
| Init missing | Upload `baseline_erm_resnet18.pt` via scp |
| `bash\r: No such file` | `sed -i 's/\r$//' scripts/cluster/*.sh` |
| Job idle long | `condor_q -better-analyze <jobid>` |
| OOM on GPU | Set `TML3_BATCH=64` before run script |
| Slow robust eval | Normal — PGD-20 on full val takes minutes |

---

## After Step 2

When `progress.json` shows `status: finished` and robust >> 0.24%:

1. Optionally submit `fgsm_at_resnet18.pt` via GUI (60 min cooldown).
2. Proceed to **Step 3 — PGD-AT** (main leaderboard improvement).

See `docs/STEP2_ROBUSTNESS_STRATEGY.md`.
