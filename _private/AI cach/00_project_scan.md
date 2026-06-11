# Project scan — Assignment 3 (Robustness)

> Scan date: 2026-06-07  
> Status: **documentation phase only — no implementation started**

---

## Task summary

| Item | Detail |
|------|--------|
| **Goal** | Train an image classifier robust to adversarial attacks |
| **Input shape** | `3 × 32 × 32` (RGB, uint8 in `.npz`) |
| **Output** | 9 raw logits (classes 0–8) |
| **Architectures** | `resnet18`, `resnet34`, `resnet50` (torchvision only) |
| **Training data** | 50,000 labeled images in `train.npz` |
| **Metric** | `Score = 0.5 × clean_accuracy + 0.5 × robustness_accuracy` |
| **Hard gate** | Clean accuracy must be **> 50%** or submission rejected |
| **Submit artifact** | `.pt` file = `state_dict` only (not full model) |
| **Leaderboard** | http://34.63.153.158/leaderboard_page |
| **Deadline** | **16 June 2026, 23:59** (leaderboard + CMS — no extensions) |
| **Weight** | 10% of course grade |
| **HuggingFace data** | [SprintML/tml26_task3](https://huggingface.co/datasets/SprintML/tml26_task3) (~127 MB) |

---

## What you HAVE (after download)

| Asset | Expected path | Notes |
|-------|---------------|-------|
| Training images + labels | `data/train.npz` | 50k × 32×32×3, uint8, labels 0–8 |
| Loading example | `task_template.py` | Model definition, save format, assertions |
| Upload script | `submission.py` | API key, model path, architecture name |
| Assignment PDF | `Assignment_3_-_Robustness.pdf` | Official spec |

## What you DON'T have

| Missing | Why it matters |
|---------|----------------|
| Private test set | Final score computed server-side |
| Exact attack parameters (ε, steps, norm) | Must generalize, not overfit one attack |
| Attack implementation on server | You defend blind — train with strong local attacks |
| Ground-truth robust labels | Only server knows robust accuracy |

---

## On disk right now

```
Assignment3/
├── Assignment_3_-_Robustness.pdf     ✅ official spec
└── _private/
    └── AI cach/                        ✅ learning docs + PDF extracts
        ├── README.md
        ├── 00_project_scan.md          ← this file
        ├── 01_master_plan.md
        ├── pdf_extracts/
        └── assignment3_master_guide.ipynb  ⏳ generated next
```

**Not yet present (expected after HF download + setup):**

```
Assignment3/
├── task_template.py
├── submission.py
├── requirements.txt
├── data/train.npz
├── src/                    ← your training code (future)
├── scripts/                ← train, eval, submit helpers (future)
├── configs/                ← hyperparameters (future)
├── results/                ← checkpoints, logs (future)
├── report/                 ← LaTeX 2-page report (future)
├── README.md               ← public repo readme (future)
└── .gitignore
```

---

## Key differences from Assignment 1 & 2

| | Assignment 1 (MIA) | Assignment 2 (Stolen) | **Assignment 3 (Robustness)** |
|--|-------------------|----------------------|------------------------------|
| Role | Attacker | Victim / forensic analyst | **Defender / trainer** |
| Output | CSV scores per sample | CSV scores per suspect | **Model weights (.pt)** |
| Metric | TPR @ 5% FPR | TPR @ 5% FPR | **0.5 clean + 0.5 robust acc** |
| Main skill | Attack design | Detection / comparison | **Adversarial training** |
| Data size | Model + probe sets | 360 suspect models | **50k training images** |
| Compute | Shadow models on HPC | Feature extraction GPU | **Long GPU training (PGD-AT)** |

---

## Detection / defense approaches (learning map)

| Stage | Method | Purpose | Expected outcome |
|-------|--------|---------|------------------|
| Baseline | Standard training (ERM) | Sanity check | High clean, **near-zero robust** |
| v001 | FGSM adversarial training | Fast first defense | Some robust, moderate clean drop |
| v002 | PGD adversarial training (Madry) | Strong standard defense | Better robust, lower clean |
| v003 | PGD-AT + tuning (ε, steps, aug) | Optimize unified score | Balance clean/robust trade-off |
| v004 | Larger arch (ResNet-34/50) | More capacity for robustness | May help both metrics |
| v005 | Efficient AT (TRADES, FreeAT, etc.) | Scale training | Only if compute-limited |

---

## Status checklist

| Step | Status |
|------|--------|
| Read assignment PDF + Tutorial 6 | ✅ extracted to `pdf_extracts/` |
| Read lecture 05 (Robustness) | ✅ study notes exist in course hub |
| Download HF starter files | ⏳ not done |
| Local baseline (standard training) | ⏳ not started |
| Local adversarial training | ⏳ not started |
| Local robust eval (PGD) | ⏳ not started |
| First leaderboard submission | ⏳ not started |
| CMS report + ZIP | ⏳ not started |

---

## Common rejection reasons (from official PDF)

- Clean accuracy below 50%
- Wrong output shape (must be 9 logits)
- Wrong input shape (must accept 3×32×32)
- Saved full model instead of `state_dict`
- Custom architecture not in allowed list
- `model-name` field doesn't match actual architecture

---

## Submission hygiene

- API key in `.env` only — never commit
- Cooldown: **60 min** success / **2 min** on error
- Public leaderboard = 30% — don't overfit
- CMS ZIP: code + 2-page report — **no model weights**
- Leaderboard keeps **best score per team** only
