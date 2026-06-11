# Trustworthy ML 2026 — Assignment 3: Adversarial Robustness

Train a ResNet image classifier that stays accurate on **clean** and **adversarially perturbed** 32×32 inputs (9 classes).

**Metric:** `Score = 0.5 × clean_accuracy + 0.5 × robustness_accuracy`  
**Deadline:** 16 June 2026, 23:59 (leaderboard + CMS ZIP)

> **Private repo** — includes `_private/` learning docs and plans. Do not publish API keys or model weights.

---

## Repository layout

```
tml-assignment3/
├── README.md                        ← this file
├── SUBMIT.md                        ← CMS + leaderboard checklist
├── Assignment_3_-_Robustness.pdf    ← official spec
├── _private/                        ← workshop docs, master guide notebook
│   └── AI cach/
│       ├── assignment3_master_guide.ipynb   ← START HERE
│       ├── 00_project_scan.md
│       └── 01_master_plan.md
├── task_template.py                 ← from HuggingFace (after download)
├── submission.py                    ← upload .pt to server
├── data/train.npz                   ← gitignored — download locally
├── src/                             ← training code (future)
├── scripts/                         ← train / eval (future)
├── configs/                         ← hyperparameters (future)
└── report/                          ← 2-page ICLR report (future)
```

**Data:** [SprintML/tml26_task3](https://huggingface.co/datasets/SprintML/tml26_task3)  
**Leaderboard:** http://34.63.153.158/leaderboard_page

---

## Status (2026-06-07)

| Phase | Status |
|-------|--------|
| Documentation & master guide | Done |
| HuggingFace starter download | Not started |
| Baseline / adversarial training | Not started |
| Leaderboard submission | Not started |
| CMS report + ZIP | Not started |

---

## Quick start (when implementation begins)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download course files from HuggingFace into repo root
# huggingface-cli download SprintML/tml26_task3 --local-dir .

copy .env.example .env   # add TML_API_KEY from CMS
python task_template.py  # verify data + model shape
```

Planned pipeline: standard training → FGSM-AT → PGD-AT → submit best `state_dict` via `submission.py`.

See `_private/AI cach/assignment3_master_guide.ipynb` for the full zero-to-hero workshop.

---

## Related repos

- [tml-assignment1](https://github.com/osama11osama/tml-assignment1) — Membership Inference
- [tml-assignment2](https://github.com/osama11osama/tml-assignment2) — Stolen Model Detection
