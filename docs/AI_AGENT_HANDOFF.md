# AI Agent Handoff — TML Assignment 3 (Adversarial Robustness)

> **Purpose:** Give this file to a new AI agent (Cursor, ChatGPT, etc.) so it understands the full project, what happened in the recent conversation, and what to do next.  
> **Last updated:** 2026-06-14 (v1.2 submittable — TRADES β=8 LB **0.582405**; tag `v1.2-submittable`)  
> **Student / cluster user:** `atml_team044`  
> **Leaderboard team name:** `team_XLVII` (different from cluster username)

---

## 1. Project overview

### Course & assignment

- **Course:** Trustworthy Machine Learning, SS2026, Saarland University  
- **Assignment:** Assignment 3 — **Adversarial Robustness**  
- **Task ID (submission server):** `03-robustness`  
- **Deadline:** 16 June 2026, 23:59 (leaderboard + CMS ZIP report)

### Goal

Train a **ResNet** classifier on 32×32 RGB images (9 classes) that performs well on:

1. **Clean** validation/test images  
2. **Adversarial** images (ℓ∞ perturbations, ε = 8/255)

### Scoring metric

```
Unified Score = 0.5 × clean_accuracy + 0.5 × robustness_accuracy
```

Both metrics matter equally. A model with 97% clean and 0% robust scores ~0.49 unified — same as 0% clean and 97% robust.

### Data & submission

| Resource | Location |
|----------|----------|
| Dataset | [HuggingFace: SprintML/tml26_task3](https://huggingface.co/datasets/SprintML/tml26_task3) |
| Local data | `data/train.npz` (50k samples, 9 classes, 3×32×32) |
| Leaderboard | http://34.63.153.158/leaderboard_page |
| Submit artifact | `.pt` PyTorch `state_dict` (ResNet18/34/50) |
| Submit script | `submission.py` or GUI `_private/tools/tml_submit_gui.py` |
| API key | `TML_API_KEY` in `.env` (CMS) |

### Workspace paths

| Machine | Path |
|---------|------|
| **Windows (local)** | `C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment3` |
| **HPC cluster** | `~/tml26_task3` on `conduit2.hpc.uni-saarland.de` |
| **Local GPU** | NVIDIA GeForce RTX 5060 Laptop, PyTorch `2.11.0+cu128` |
| **Python venv** | `.venv/` |

---

## 2. Theory & strategy (paper-backed)

Full strategy doc: `docs/STEP2_ROBUSTNESS_STRATEGY.md`  
Learning notebook: `_private/AI cach/assignment3_master_guide.ipynb`

### Three-phase training pipeline

| Step | Method | Paper | Purpose |
|------|--------|-------|---------|
| **1 — ERM** | Standard training on clean images | Baseline | High clean accuracy (~97%); ~0% robust |
| **2 — FGSM-AT** | Adversarial training with 1-step FGSM | Goodfellow et al., *Explaining and Harnessing Adversarial Examples* | Warm-up; lifts robust slightly; **not for final submit** |
| **3 — PGD-AT** | Adversarial training with multi-step PGD | Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* | **Main method** for leaderboard improvement |

### Attack parameters (assignment standard)

| Parameter | Value |
|-----------|-------|
| ε (epsilon) | `8/255` ≈ 0.03137 |
| α (step size) | `2/255` ≈ 0.00784 |
| PGD train steps | 7 (main) or 10 (trial variant) |
| PGD eval steps | 20 (local + validation during training) |

### What NOT to do

From Athalye et al., *Obfuscated Gradients Give a False Sense of Security*:

- Do **not** rely on input preprocessing / gradient masking tricks without proper adversarial training  
- Do **not** overfit to one local PGD config — server attack params are hidden (Tutorial 6)  
- Do **not** submit FGSM-AT if unified < ERM baseline

---

## 3. Results so far

### Step 1 — ERM (COMPLETE)

| Metric | Local val | Server LB |
|--------|-----------|-----------|
| Clean | **97.28%** | ~97% |
| Robust (PGD-20) | **0.24%** | ~0% |
| **Unified** | ~0.487 | **0.486371** |
| Rank | — | ~31/34 |
| Checkpoint | `results/checkpoints/baseline_erm_resnet18.pt` | |
| Run dir | `results/runs/erm_resnet18/` | |
| Submission ID | 2721 | |

Details: `docs/STEP1_ERM_RESULTS.md`

### Step 2 — FGSM-AT (COMPLETE on cluster)

| Metric | Value |
|--------|-------|
| Epochs | 50 (~37 min on cluster) |
| Clean | **72.66%** |
| Robust | **9.24%** |
| **Unified** | **0.4095** |
| Verdict | **Do NOT submit** — worse than ERM (0.486) |
| Checkpoint | `results/checkpoints/fgsm_at_resnet18.pt` |
| Run dir | `results/runs/fgsm_at_resnet18/` |

FGSM-AT is used only as **initialization** for Step 3 PGD-AT.

### Step 3 — PGD-AT (COMPLETE)

| Metric | Local val @80 | Server LB |
|--------|---------------|-----------|
| Clean | 67.88% | — |
| Robust | 45.98% | — |
| **Unified** | 0.5693 | **0.575136** |
| Rank | — | ~24/35 |

**Submittable snapshot:** `docs/SUBMISSION_SNAPSHOT.md` (v1.0)  
**Full results:** `docs/STEP3_PGD_RESULTS.md`  
**Checkpoint:** `results/checkpoints/pgd_at_resnet18.pt`  
**Resume for Phase 4:** `results/runs/pgd_at_resnet18/last.pt` (epoch 80)

### Phase 4 — Extend PGD-AT (NEXT on cluster)

**Goal:** push LB beyond 0.575 (stretch 0.70 unlikely; top ~0.63)  
**Method:** resume `pgd_at` epoch 80 → **120**, PGD-**10** steps  
**Runbook:** `docs/CLUSTER_STEP4.md`

---

## 4. HPC cluster constraints & workflow

### Saarland HPC (Condor + Docker)

- **Host:** `conduit2.hpc.uni-saarland.de`  
- **User:** `atml_team044`  
- **Universe:** Docker (`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`)  
- **Critical limit:** **~1 hour per GPU job** (`+MaxRuntime = 3600`)  
- Full 80-epoch PGD-AT cannot run in one job — must use **chunked training with `--resume`**

### Chunk workflow (4 × 1 hour)

| Chunk job | Target epoch | Condor file |
|-----------|--------------|-------------|
| 1 | 20 | `scripts/cluster/condor/train_pgd_at_chunk20.sub` |
| 2 | 40 | `scripts/cluster/condor/train_pgd_at_chunk40.sub` |
| 3 | 60 | `scripts/cluster/condor/train_pgd_at_chunk60.sub` |
| 4 | 80 | `scripts/cluster/condor/train_pgd_at_chunk80.sub` |

**GPU optimizations in chunk jobs (environment vars in `.sub` files):**

| Setting | Value |
|---------|-------|
| `TML3_BATCH` | 256 |
| `TML3_NUM_WORKERS` | 4 |
| `TML3_ROBUST_EVERY` | 0 (PGD-20 eval only at chunk target epoch) |
| `TML3_SAVE_EVERY` | 3 |
| Soft timeout | 3300 s (~55 min) via `timeout` in `run_pgd_at_chunk.sh` |
| `cudnn.benchmark` | enabled in `train_pgd_at.py` |

### Cluster scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/cluster/sync_to_cluster.ps1` | Upload code + checkpoints from Windows |
| `scripts/cluster/run_pgd_at_chunk.sh` | Run one chunk toward target epoch |
| `scripts/cluster/check_pgd_progress.sh` | Show progress + next chunk |
| `scripts/cluster/submit_pgd_next.sh` | Auto-submit next chunk |
| `scripts/cluster/submit_pgd_chain.sh` | DAG: all 4 chunks sequentially (1 GPU) |
| `scripts/cluster/submit_pgd_parallel_trial.sh` | 2 GPUs: PGD-7 vs PGD-10 trial at epoch 20 |
| `scripts/cluster/compare_pgd_trials.sh` | Compare trial results, pick winner |
| `scripts/cluster/condor/pgd_at_chain.dag` | HTCondor DAG for auto-chaining |

Full runbook: `docs/CLUSTER_STEP3.md`

### CRLF fix (Windows → Linux)

After uploading scripts, always on cluster:

```bash
find scripts/cluster -type f \( -name '*.sh' -o -name '*.sub' \) -exec sed -i 's/\r$//' {} +
chmod +x scripts/cluster/*.sh
sed -i 's/YOUR_CLUSTER_USER/atml_team044/g' scripts/cluster/condor/train_pgd_at*.sub
```

---

## 5. What happened in this conversation (chronological)

### Problem discovered

The original Step 3 job (**49473**) was submitted as a **single 80-epoch run** with `+MaxRuntime = 172800` (48h), but the cluster **actually enforces ~1 hour per GPU job**. That job would have been killed mid-training.

### Solution built

1. **Chunked PGD training** — split 80 epochs into 4 jobs (20/40/60/80) with `--resume` from `last.pt`  
2. **GPU tuning** — batch 256, 4 workers, skip mid-chunk robust eval  
3. **Auto DAG chain** — `submit_pgd_chain.sh` runs all 4 chunks without manual resubmit  
4. **Parallel trial** — `submit_pgd_parallel_trial.sh` runs PGD-7 vs PGD-10 on **2 GPUs** for epoch 20 comparison

### User actions (2026-06-12 ~00:52)

1. Uploaded cluster scripts via `scp -r scripts/cluster ...`  
2. Fixed line endings and username in `.sub` files  
3. **Mistake:** ran **both** `submit_pgd_chain.sh` AND `submit_pgd_parallel_trial.sh` at once  
   - This created **conflicting jobs** writing to the same `pgd_at` run directory  
4. Cleaned up:
   - `condor_rm 49473.0` — old long job (marked for removal)  
   - `condor_rm 49477` — DAG chain (marked for removal)  
   - `condor_rm 49480` — DAG child (already gone)  
5. **Final clean state:** only 2 jobs running

### Active cluster jobs (as of last user update)

| Job ID | Tag | Config | Status |
|--------|-----|--------|--------|
| **49478** | `pgd_at` | PGD-7, batch 256, chunk → epoch 20 | **Running** |
| **49479** | `pgd_at_s10` | PGD-10, batch 256, chunk → epoch 20 | **Running** |

### Training observations from logs

Both jobs **resumed from epoch 2** (leftover checkpoint from killed job 49473):

| Run | Speed | Clean acc (early) | Notes |
|-----|-------|-------------------|-------|
| PGD-7 (`49478`) | ~54 s/epoch | ~61% | 176 batches/epoch (= batch 256) |
| PGD-10 (`49479`) | ~75 s/epoch | ~61% | Slower due to 10 PGD steps |

Log files on cluster:

```
runlogs/pgd_at_chunk20_20260611_225222.log      ← monitor this one (49478, current)
runlogs/pgd_at_chunk20_20260611_225213.log      ← older/stale from overlap
runlogs/pgd_at_s10_chunk20_20260611_225213.log  ← 49479
```

`robust n/a` during epochs 1–19 is **expected** (`TML3_ROBUST_EVERY=0`). Robust PGD-20 eval runs at **epoch 20**.

### Important rule learned

**Never run `submit_pgd_chain.sh` and `submit_pgd_parallel_trial.sh` together.**  
They both start a `pgd_at` chunk-20 job → race condition on `results/runs/pgd_at_resnet18/last.pt`.

**Choose ONE path:**

- **Path A (current):** Parallel trial → compare → continue winner's chunks  
- **Path B:** DAG chain only → 4 sequential chunks, no comparison

---

## 6. Current state (snapshot)

```
Phase                          Status
─────────────────────────────────────────────────────
Step 0 Setup                   DONE
Step 1 ERM                     DONE — LB 0.486371
Step 2 FGSM-AT                 DONE — warm-up only
Step 3 PGD-AT                  DONE — LB 0.575136 (80 epochs)
Phase 4 Extend (80→120)        READY — not started on cluster
Submittable snapshot           v1.0 — docs/SUBMISSION_SNAPSHOT.md
CMS report + ZIP               NOT STARTED
```

**Best LB:** **0.582405** (`trades_b8_resnet18.pt`, TRADES β=8, ResNet18)  
**Do not:** Re-submit `trades_b8` or weaker models; next submit only if local unified > **0.5824 + margin (~0.004)**

---

## 7. Next steps (for the next agent / user)

### Phase 4 on cluster (push LB toward 0.60+)

Sync scripts from Windows, then on cluster:

```bash
cd ~/tml26_task3
find scripts/cluster -type f \( -name '*.sh' -o -name '*.sub' \) -exec sed -i 's/\r$//' {} +
chmod +x scripts/cluster/*.sh
sed -i 's/YOUR_CLUSTER_USER/atml_team044/g' scripts/cluster/condor/train_pgd_at_chunk*.sub

export TML3_FINAL_EPOCHS=120
condor_submit scripts/cluster/condor/train_pgd_at_chunk100.sub
# after finish:
condor_submit scripts/cluster/condor/train_pgd_at_chunk120.sub
```

Or: `bash scripts/cluster/submit_pgd_extend_next.sh`

### After Phase 4

```powershell
scp ... pgd_at_resnet18.pt results/checkpoints/
python scripts/eval_model.py results/checkpoints/pgd_at_resnet18.pt --architecture resnet18
# Submit via GUI only if unified > 0.575136
```

### CMS (deadline 16 June 2026)

- [x] Submittable code snapshot — git tag v1.0-submittable / `docs/SUBMISSION_SNAPSHOT.md`
- [ ] 2-page PDF report
- [ ] CMS ZIP (no weights, no data)

### If targeting ~0.70 (stretch)

1. Phase 4 extend (R18, epoch 120, PGD-10)  
2. ResNet-34 full pipeline  
3. TRADES (not implemented yet)

---

## 8. Key code files

### Training

| File | Purpose |
|------|---------|
| `scripts/train_standard.py` | Step 1 ERM |
| `scripts/train_fgsm_at.py` | Step 2 FGSM-AT |
| `scripts/train_pgd_at.py` | Step 3 PGD-AT (main) |
| `src/train_utils.py` | Checkpoints, resume, FGSM/PGD train loops |
| `src/attacks.py` | FGSM, PGD implementations |
| `src/eval_utils.py` | Clean + robust accuracy |
| `scripts/eval_model.py` | Local evaluation |

### Configs

| File | Purpose |
|------|---------|
| `configs/standard_erm.yaml` | ERM hyperparameters |
| `configs/fgsm_at.yaml` | FGSM-AT |
| `configs/pgd_at.yaml` | PGD-AT (80 epochs, batch 128 default; cluster uses 256) |

### Checkpoints

| File | Description |
|------|-------------|
| `results/checkpoints/baseline_erm_resnet18.pt` | Step 1 — submitted to LB |
| `results/checkpoints/fgsm_at_resnet18.pt` | Step 2 — init for PGD |
| `results/checkpoints/pgd_at_resnet18.pt` | Step 3 target (in progress) |
| `results/runs/pgd_at_resnet18/last.pt` | Resume checkpoint on cluster |
| `results/runs/pgd_at_resnet18/progress.json` | Metrics + status |

### Submit GUI (bug fixed in this conversation)

- **File:** `_private/tools/tml_submit_gui.py`  
- **Bug fixed:** Successful LB submits showed **Rejected** because `parse_server_eval()` ignored cached history status  
- **History:** `%APPDATA%\tml_submit_gui\task3\submit_history.jsonl`

---

## 9. Errors encountered & fixes (reference)

| Issue | Fix |
|-------|-----|
| GUI shows Rejected after successful submit | Fixed `parse_server_eval` + use `row.eval_status` from history |
| `sync_to_cluster.ps1` PowerShell parse error | Removed nested quotes |
| Cluster `set: -\r` / `$'\r'` errors | `sed -i 's/\r$//'` on `.sh`/`.sub` files |
| 1-hour job limit kills long PGD run | Chunk workflow + resume |
| Ran chain + parallel trial together | `condor_rm` conflicting jobs; use ONE path only |
| Missing `return` in `train_one_epoch` | Fixed in `src/train_utils.py` |

---

## 10. Prompt for ChatGPT / new agent

Copy-paste this to onboard a fresh agent:

---

You are helping with **Trustworthy ML Assignment 3 (Adversarial Robustness)**.

**Goal:** Maximize `0.5 × clean_acc + 0.5 × robust_acc` on a 9-class ResNet for 32×32 images.

**Current status:**
- **Best LB: 0.575136** — `pgd_at_resnet18.pt` (PGD-AT, 80 epochs, ResNet18)
- Submittable snapshot: `docs/SUBMISSION_SNAPSHOT.md` (v1.0)
- Phase 4 ready: resume epoch 80 → 120, PGD-10 (`docs/CLUSTER_STEP4.md`)
- Cluster: `atml_team044@conduit2`, `~/tml26_task3`

**Do not:** submit if unified ≤ 0.575; use FGSM-AT as final model.

**Read first:** `docs/SUBMISSION_SNAPSHOT.md`, `docs/CLUSTER_STEP4.md`, `docs/STEP3_PGD_RESULTS.md`

**Workspace:** `Assignment3/` under Trustworthy ML Assignments folder on Windows; cluster at `~/tml26_task3`.

---

## 11. Related documentation index

| Document | Content |
|----------|---------|
| `README.md` | Project overview (partially outdated progress table — update after Step 3) |
| `docs/STEP1_ERM_RESULTS.md` | ERM results + LB submit |
| `docs/STEP2_ROBUSTNESS_STRATEGY.md` | Paper-backed AT strategy |
| `docs/CLUSTER_STEP2.md` | FGSM-AT cluster runbook |
| `docs/CLUSTER_STEP3.md` | PGD-AT cluster runbook (chunks, GPU tuning, parallel trial) |
| `docs/AI_AGENT_HANDOFF.md` | This file |
| `docs/STEP3_PGD_RESULTS.md` | PGD-AT 80-epoch results + LB |
| `docs/TRADES_B8_RESULTS.md` | **Full doc: β=8 pipeline, LB 0.5824, reproduce** |
| `docs/SUBMISSION_SNAPSHOT.md` | **CMS snapshot — points to trades_b8** |
| `docs/CLUSTER_STEP4.md` | Phase 4 extend training (→ epoch 120) |
| `_private/AI cach/assignment3_master_guide.ipynb` | Full learning guide |

---

## 12. Sleep checkpoint — 2026-06-14 ~06:35

**When user asks "what next?" — remind them of this section and ask for cluster results.**

### Best LB so far (do not lose this)

| Checkpoint | Server LB | Notes |
|------------|-----------|-------|
| **`trades_b8_resnet18.pt`** | **0.582405** | **CURRENT BEST — TRADES β=8, 20ep (`docs/TRADES_B8_RESULTS.md`)** |
| `trades_r18_resnet18.pt` | 0.575571 | TRADES β=6 fallback |
| `pgd_at_resnet18.pt` | 0.575136 | v1.0-submittable fallback |

### What we did today

1. Week 1: ERM R34 chunks 20→40→60→80; TRADES R18 40 ep → LB 0.575571
2. Added `sync_from_cluster.ps1`, `eval_archive.py`, `results/archive/`
3. Launched **parallel tracks** (`submit_parallel_tracks.sh max4`) — jobs **49897–49901**

### Cluster jobs submitted (check `condor_q` / logs on wake)

| Job | Track | Tag | Expected outcome |
|-----|-------|-----|------------------|
| 49897 | ERM R34 chunk 60 | `erm_r34` | Likely **done** |
| 49898 | ERM R34 chunk 80 | `erm_r34` | **Done** — clean ~97%, robust ~0% |
| 49899 | TRADES R34 chunk 20 | `trades_r34` | **Running** — init ERM R34; high KL epoch 1 is normal |
| 49900 | TRADES β=4 pilot 20 ep | `trades_b4` | **Running** — compare only |
| 49901 | TRADES β=8 pilot 20 ep | `trades_b8` | **Running** — compare only |

### What we expect when cluster finishes

| Track | Success looks like | LB submit? |
|-------|-------------------|------------|
| ERM R34 → 100 | clean >90%, robust ~0% | **No** (init only) |
| TRADES R34 ep 20→40→… | unified val **> 0.575571** | **Yes if beats trades_r18** |
| trades_b4 / trades_b8 | β sweep from trades_r18 | **b8 submitted — LB 0.582405; b4 skip (~0.536 local)** |

### Pending cluster commands (after jobs finish)

```bash
cd ~/tml26_task3
condor_q
# ERM (if epoch 80 done, not 100):
condor_submit scripts/cluster/condor/train_erm_r34_chunk100.sub
# TRADES R34 (after chunk 20):
condor_submit scripts/cluster/condor/train_trades_r34_chunk40.sub
# Optional 5th GPU:
bash scripts/cluster/submit_parallel_tracks.sh extend   # TRADES R18 → 60→80
```

### Ask user to paste these on wake-up

1. Output of `condor_q` (empty or not)
2. Tail of finished logs:
   - `runlogs/trades_r34_chunk20_*.log`
   - `runlogs/trades_b4_chunk20_*.log`
   - `runlogs/trades_b8_chunk20_*.log`
   - `runlogs/erm_r34_chunk100_*.log` (if submitted)
3. Or: `TML3_TAG=trades_r34 TML3_ARCH=resnet34 TML3_FINAL_EPOCHS=80 bash scripts/cluster/check_train_progress.sh`

### Copy locally + test (Windows)

```powershell
cd Assignment3
powershell -ExecutionPolicy Bypass -File scripts/cluster/sync_from_cluster.ps1 -ClusterUser atml_team044 -Force
python scripts/eval_archive.py
```

Then GUI: Refresh → **Local eval** → **Submit to LB only if unified > ~0.586** (local overestimates server by ~0.003)

- `trades_r34_resnet34.pt` → architecture **resnet34**
- `trades_r18_resnet18.pt` / `trades_b4` / `trades_b8` → **resnet18**

### Deadline reminder

**16 June 2026** — CMS report (PDF) + ZIP still needed.

---

*End of handoff document.*
