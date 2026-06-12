# Trustworthy ML 2026 — Assignment 3: Adversarial Robustness

Train a ResNet image classifier that stays accurate on **clean** and **adversarially perturbed** 32×32 inputs (9 classes).

**Metric:** `Score = 0.5 × clean_accuracy + 0.5 × robustness_accuracy`  
**Deadline:** 16 June 2026, 23:59 (leaderboard + CMS ZIP)  
**Task ID:** `03-robustness`

> **Private repo** — includes `_private/` learning docs and plans. Do not publish API keys or model weights.

**Data:** [SprintML/tml26_task3](https://huggingface.co/datasets/SprintML/tml26_task3)  
**Leaderboard:** http://34.63.153.158/leaderboard_page

---

## Progress

| Phase | Status |
|-------|--------|
| Step 0 — Setup, download, verify | **Done** |
| Step 1 — Standard training baseline (ERM) | **Done** — LB **0.486371** (`docs/STEP1_ERM_RESULTS.md`) |
| Step 2 — FGSM adversarial training | **Done** — unified 0.410 (warm-up only) |
| Step 3 — PGD adversarial training | **Done** — 80 epochs; LB **0.575136** (`docs/STEP3_PGD_RESULTS.md`) |
| **Submittable snapshot (CMS)** | **`docs/SUBMISSION_SNAPSHOT.md`** — v1.0, best: `pgd_at_resnet18.pt` |
| Phase 4 — Extend PGD-AT (optional) | **Ready** — epoch 80→120 (`docs/CLUSTER_STEP4.md`) |
| CMS report + ZIP | Not started |

---

## Step 0 — What was done (2026-06-11)

This step established the project infrastructure. **No model training yet** — only data, code layout, and validation.

### 1. Environment

- Python virtual environment: `.venv/`
- Dependencies: `requirements.txt` (torch, torchvision, numpy, huggingface_hub, requests, python-dotenv, …)
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU
- **PyTorch:** `2.11.0+cu128` (CUDA 12.8) — required for adversarial training later

If you recreate the venv, install the CUDA build explicitly:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Data download

- Source: HuggingFace dataset `SprintML/tml26_task3`
- Script: `scripts/download_data.py`
- Output: `data/train.npz` (~127 MB, gitignored)

**Dataset summary** (from `scripts/explore_data.py`):

| Property | Value |
|----------|-------|
| Samples | 50,000 |
| Image shape | `(N, 3, 32, 32)` — RGB, uint8 in file |
| Preprocessing | divide by `255.0` → float in `[0, 1]` |
| Classes | 9 (labels `0` … `8`) |
| Class balance | min 4,424 / max 7,127 per class (ratio ≈ 0.62) |

### 3. Code added

| Path | Purpose |
|------|---------|
| `src/paths.py` | Repo paths, constants (`NUM_CLASSES=9`) |
| `src/data.py` | Load `train.npz`, train/val `DataLoader` (90/10 split) |
| `src/model.py` | `make_model()` for `resnet18` / `resnet34` / `resnet50` |
| `scripts/download_data.py` | Download from HuggingFace → `data/train.npz` |
| `scripts/explore_data.py` | Print dataset statistics |
| `scripts/verify_setup.py` | End-to-end sanity checks |
| `task_template.py` | Official-style load + model I/O + save checkpoint |
| `submission.py` | Upload `.pt` to server; supports `--validate-only` and `.env` |
| `.env.example` | Template for `TML_API_KEY` |

Official HuggingFace copies of `task_template.py` / `submission.py` are kept in `hf_download/` for reference. The repo versions add path handling, validation, and dotenv support.

### 4. Verification results

All checks passed (`python scripts/verify_setup.py`):

- `data/train.npz` loads correctly
- Image tensor shape `(3, 32, 32)`, labels in `[0, 8]`
- Train/val loaders: ~352 / ~40 batches (batch size 128)
- All three architectures produce output shape `(1, 9)` on GPU
- `task_template.py` saves `results/checkpoints/template_sanity.pt`
- `submission.py --validate-only` accepts the sanity checkpoint

### 5. What is NOT done yet

- No trained model (only random-weights sanity checkpoint)
- No adversarial attack or adversarial training code
- No leaderboard upload
- No CMS report

---

## Repository layout

```
tml-assignment3/
├── README.md
├── SUBMIT.md
├── requirements.txt
├── .env.example
├── Assignment_3_-_Robustness.pdf
├── task_template.py
├── submission.py
├── src/
│   ├── paths.py
│   ├── data.py
│   └── model.py
├── scripts/
│   ├── download_data.py
│   ├── explore_data.py
│   └── verify_setup.py
├── data/
│   └── train.npz              # gitignored — run download_data.py
├── results/checkpoints/       # gitignored .pt files
├── hf_download/               # HF cache (reference templates)
├── _private/
│   └── tools/
│       ├── tml_submit_gui.py      # submit GUI + history
│       └── launch_submit_gui.ps1
```

---

## Submit GUI (same pattern as Assignment 2)

Desktop tool for uploading `.pt` checkpoints — validate, submit, cooldown, history, queue, leaderboard.

```powershell
powershell -ExecutionPolicy Bypass -File _private/tools/launch_submit_gui.ps1
```

| Feature | Assignment 2 | Assignment 3 |
|---------|--------------|--------------|
| Artifact | CSV (`id,score`) | `.pt` state_dict |
| Extra field | — | `model_name` (resnet18/34/50) |
| Scan folder | `results/submissions` | `results/checkpoints` |
| Task ID | `19-stolen-model-detection` | `03-robustness` |
| History path | `%APPDATA%\tml_submit_gui\task2\` | `%APPDATA%\tml_submit_gui\task3\` |

The GUI reuses your **TML API key** from Assignment 1/2 settings if already saved.

---

## Quick start (reproduce Step 0)

```powershell
cd tml-assignment3
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/download_data.py
python scripts/explore_data.py
python scripts/verify_setup.py
python task_template.py

copy .env.example .env          # add TML_API_KEY from CMS (for upload later)
python submission.py --validate-only results/checkpoints/template_sanity.pt --model-name resnet18
```

Expected final line from `verify_setup.py`:

```
All checks passed. Ready for baseline training (Stage 1).
```

---

## Step 1 results (2026-06-11)

**Step 1 is NOT complete.** Only short tests were run. Full details: [`docs/STEP1_ERM_RESULTS.md`](docs/STEP1_ERM_RESULTS.md)

| Run | Epochs | Clean (val) | Robust (PGD-20) | Unified | Status |
|-----|--------|-------------|-----------------|---------|--------|
| exp001 smoke | 2 | **55.34%** | 3.16% | 0.293 | Pipeline OK |
| exp002 resume test | 3 | 40.86% | 6.22% | 0.235 | Resume OK; do not submit |
| exp003 full ERM | 100 | — | — | — | **You still need to run this** |

GPU verified: RTX 5060, PyTorch `2.11.0+cu128`, training uses `cuda`.

---

## Step 1 — ERM baseline training (you run this)

> **Policy:** Long GPU jobs (many epochs) are **run by you**, not started automatically by the agent.
> Scripts save resume checkpoints every few epochs so a crash does not restart from epoch 1.

```powershell
# Start (100 epochs, saves resume ckpt every 5 epochs)
python scripts/train_standard.py --device cuda --epochs 100 --save-every 5

# If interrupted — continue from last saved epoch
python scripts/train_standard.py --device cuda --epochs 100 --resume

# After training
python scripts/eval_model.py results/checkpoints/baseline_erm_resnet18.pt --architecture resnet18
python submission.py --validate-only results/checkpoints/baseline_erm_resnet18.pt --model-name resnet18
```

**Run folder** (`results/runs/erm_resnet18/`):

| File | Purpose |
|------|---------|
| `last.pt` | Full resume state (model + optimizer + epoch) |
| `best.pt` | Best val-clean weights for submission |
| `progress.json` | Metrics history + status (`running` / `finished`) |

**Console steps:** `[SETUP]` → `[TRAIN]` (batch %) → `[EVAL]` → `[SAVE]` → `[DONE]`

| Script | Purpose |
|--------|---------|
| `scripts/train_standard.py` | ERM training with resume |
| `scripts/eval_model.py` | Clean + PGD robust eval |
| `src/train_utils.py` | Shared checkpoint / progress helpers |

## Next step — Step 2 (FGSM adversarial training)

After ERM baseline completes: `scripts/train_fgsm_at.py` → then PGD-AT.

For theory and full roadmap see `_private/AI cach/assignment3_master_guide.ipynb`.

---

## Related repos

- [tml-assignment1](https://github.com/osama11osama/tml-assignment1) — Membership Inference
- [tml-assignment2](https://github.com/osama11osama/tml-assignment2) — Stolen Model Detection
- [tml-assignment3](https://github.com/osama11osama/tml-assignment3) — this repo (private)
