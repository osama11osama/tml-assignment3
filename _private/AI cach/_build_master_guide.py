"""Build assignment3_master_guide.ipynb — documentation only, no implementation."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent / "assignment3_master_guide.ipynb"
cells = []


def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})


def code(source: str):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source.splitlines(True),
    })


md("""# Assignment 3 — Master Learning Guide
## Adversarial Robustness | Trustworthy ML 2026

> **Zero-to-hero workshop.** Read top-to-bottom once, then use sections as reference.
> **No implementation in this session** — theory, planning, and reading map only.

| | |
|---|---|
| **Your role** | Defender — train a robust classifier |
| **Goal** | High clean + high robust accuracy on 9-class 32×32 images |
| **Metric** | Score = 0.5 × clean_acc + 0.5 × robust_acc |
| **Submit** | `.pt` state_dict (ResNet-18/34/50) |
| **Deadline** | **16 June 2026, 23:59** (leaderboard + CMS ZIP) |
| **Location** | `_private/AI cach/` (local only — never submit this) |

### How to use this notebook

1. Run the **setup cell** below.
2. Read **Parts 1–4** — understand the assignment story.
3. Run **visualization cells** — intuitive metric & trade-off.
4. Read **Parts 5–7** — attacks, defenses, file map.
5. Read **Parts 8–10** — workflows & checklist.
6. Finish with **Part 11** — your personal action plan.
""")

code("""# Setup — run first (works even before data download)
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

NOTEBOOK_DIR = Path.cwd()
ROOT = NOTEBOOK_DIR.parents[1] if NOTEBOOK_DIR.name == "AI cach" else NOTEBOOK_DIR
sys.path.insert(0, str(ROOT))

print("Project root:", ROOT)
print("Assignment PDF:", (ROOT / "Assignment_3_-_Robustness.pdf").exists())
print("train.npz downloaded:", (ROOT / "data" / "train.npz").exists())
print("task_template.py:", (ROOT / "task_template.py").exists())

plt.rcParams.update({"figure.figsize": (10, 5), "font.size": 11})
""")

md("""---
# Part 1 — The Story (Threat Model)

Think like a **security engineer deploying ML**, not only an ML trainer.

## Scenario

You deploy an image classifier (9 classes, 32×32 RGB). An attacker adds a **tiny perturbation** to an image — invisible to humans — and your model misclassifies it. This is an **evasion attack** at inference time.

Your job: **train** a model that stays correct on both clean and perturbed inputs.

```
┌─────────────────────────────────────────────────────────────────┐
│  YOU (DEFENDER)                                                  │
│  train.npz (50k) ──► adversarial training ──► Robust ResNet     │
│                           │                                      │
│                           ▼ submit state_dict (.pt)              │
│  Server ──► clean test + hidden adversarial attacks              │
│  Score = 0.5 × clean_acc + 0.5 × robust_acc                     │
│  (30% public / 70% private — don't overfit public!)             │
└─────────────────────────────────────────────────────────────────┘

ATTACKER (server, hidden):
  x* = x + δ   where ||δ|| ≤ ε
  goal: fool your model
```

## White-box vs black-box (lecture recap)

| Setting | Attacker knows | This assignment |
|---------|----------------|-----------------|
| **White-box** | Weights, gradients | Server attacks your submitted weights |
| **Black-box** | API queries only | Not your training setup — but know for exam |
| **Transfer** | Surrogate model | Perturbations transfer across models |

**Key insight:** You don't know server's exact ε, norm, or attack code. Train with **strong diverse local attacks** (PGD) for generalization.

## Why this matters (real world)

- Medical imaging: benign-looking scan → wrong diagnosis
- Autonomous driving: sticker on sign → misread speed limit
- Security: evade malware / face recognition classifiers

Report conclusion must discuss these implications (see `report-plan.md`).
""")

md("""---
# Part 2 — Official Assignment Specification

*Sources: Assignment_3_-_Robustness.pdf, Tutorial 6*

## What you HAVE

| Asset | Location (after HF download) |
|-------|-------------------------------|
| Training data | `data/train.npz` — 50k images, uint8, 9 classes |
| Template | `task_template.py` |
| Submit script | `submission.py` |
| HF dataset | [SprintML/tml26_task3](https://huggingface.co/datasets/SprintML/tml26_task3) |

## What you DON'T have

- Private test set
- Server attack parameters
- Server attack source code

## Model constraints

| Rule | Value |
|------|-------|
| Architectures | `resnet18`, `resnet34`, `resnet50` (torchvision) |
| Input | 3 × 32 × 32 |
| Output | 9 **logits** (not probabilities) |
| Save format | `state_dict` only in `.pt` |
| Minimum clean acc | **> 50%** or rejected |

## CMS deliverables (in addition to leaderboard)

- ZIP: code + **2-page report** (ICLR template, no abstract)
- README: reproduce best result only
- **No** model weights in ZIP
""")

md("""---
# Part 3 — Course Materials Map

Read in priority order. Full details in course hub docs — not duplicated here.

| Priority | Document | Path | Why read |
|----------|----------|------|----------|
| P0 | Assignment PDF | `Assignment_3_-_Robustness.pdf` | Official rules |
| P0 | Tutorial 6 | `Tutorials/Tutorial_6_-_Assignment_3.pdf` | Submit walkthrough |
| P1 | Lecture 05 study notes | `TML_Summury/.../lectures/05-robustness/study-notes.md` | Full theory |
| P1 | Glossary | `assignment-03-robustness/glossary.md` | Terms + Arabic |
| P2 | Madry PGD paper | openreview.net/forum?id=rJzIBfZAb | Main defense |
| P2 | Goodfellow FGSM | arxiv.org/abs/1412.6572 | First AT baseline |
| P3 | Athalye obfuscated gradients | proceedings.mlr.press/v80/athalye18a.html | Defense pitfalls |
| P3 | Tutorial 5 slides | Robustness section | Attack evolution |

**Extracted text:** `pdf_extracts/` in this folder (searchable).
""")

md("""---
# Part 4 — Project Status

> See `00_project_scan.md` for live checklist.

| Step | Status |
|------|--------|
| Documentation | ✅ complete (2026-06-07) |
| HF download | ⏳ not done |
| Baseline ERM | ⏳ not started |
| FGSM-AT | ⏳ not started |
| PGD-AT | ⏳ not started |
| Leaderboard submit | ⏳ not started |
| CMS report + ZIP | ⏳ not started |

## Comparison to your previous assignments

| | A1 MIA | A2 Stolen | **A3 Robustness** |
|--|--------|-----------|-------------------|
| Output | CSV | CSV | **Model .pt** |
| Role | Attacker | Forensic analyst | **Defender** |
| Core method | Shadow models, LiRA | Weight/logit similarity | **Adversarial training** |
| Metric | TPR@5%FPR | TPR@5%FPR | **0.5 clean + 0.5 robust** |
""")

md("""---
# Part 5 — Attacks (What the Server Does to You)

You won't implement the server attack — but you **must understand** it to defend.

## FGSM (one step)

$$x^* = x + \\varepsilon \\cdot \\mathrm{sign}(\\nabla_x L(x, y, \\theta))$$

- Fast, weak — good for **starting** adversarial training
- Tutorial 6 recommends as entry point

## PGD (multi-step — strong benchmark)

$$x^{(t+1)} = \\Pi_{\\|x-x_0\\|\\leq\\varepsilon} \\left( x^{(t)} + \\alpha \\cdot \\mathrm{sign}(\\nabla_x L(x^{(t)}, y, \\theta)) \\right)$$

- Random start inside ε-ball (R+PGD)
- **Use PGD for evaluation** even if training with FGSM
- Madry et al.: PGD is "strongest" first-order attack — train against it

## Attack strength knobs

| Knob | Effect |
|------|--------|
| ε larger | Stronger attack, harder to defend |
| More steps | Stronger attack, slower |
| L∞ vs L2 | Assignment likely L∞ (standard for CIFAR) |

## Adaptive attacks warning (Athalye)

Defenses that hide gradients (input masking, weird ops) look robust but fail when attacker adapts. **Stick to adversarial training** — principled defense.
""")

md("""---
# Part 6 — Defenses (What You Will Implement Later)

## Stage 0: Standard training (ERM)

- Minimize loss on clean data only
- **Expected:** high clean, robust ≈ random (1/9 ≈ 11%)

## Stage 1: FGSM adversarial training

Each batch:
1. Compute FGSM perturbation
2. Train on adversarial images

## Stage 2: PGD adversarial training (main)

Min-max optimization:

$$\\min_\\theta \\; \\mathbb{E}_{(x,y)} \\left[ \\max_{\\|\\delta\\|\\leq\\varepsilon} L(\\theta, x+\\delta, y) \\right]$$

Each batch:
1. **Inner loop:** PGD to find worst-case x_adv
2. **Outer loop:** SGD step on x_adv

## Stage 3: Tune for unified score

Don't maximize robust alone — **50% weight on clean**.

Strategies:
- Adjust ε (smaller ε → higher clean, lower robust)
- Mixed batches (clean + adversarial)
- Bigger model (ResNet-34/50)
- Longer training

See `implementation-roadmap.md` for full stage list.
""")

code("""# Part 6b — Visualize clean vs robust trade-off (simulated)
# Illustrates WHY unified score matters — not real experiment data

clean = np.array([95, 90, 85, 75, 70, 60, 55])
robust = np.array([5, 15, 25, 40, 50, 55, 58])
unified = 0.5 * clean/100 + 0.5 * robust/100

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].scatter(clean, robust, s=80, c=unified, cmap='viridis')
for c, r, u in zip(clean, robust, unified):
    axes[0].annotate(f'{u:.2f}', (c, r), textcoords='offset points', xytext=(4, 4), fontsize=9)
axes[0].set_xlabel('Clean accuracy (%)')
axes[0].set_ylabel('Robust accuracy (%)')
axes[0].set_title('Trade-off: each point is a training config')
axes[0].axvline(50, color='red', ls='--', alpha=0.7, label='50% clean gate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

labels = ['ERM', 'FGSM', 'FGSM+', 'PGD', 'PGD+', 'PGD++', 'Best?']
axes[1].bar(labels, unified, color=plt.cm.viridis(unified / unified.max()))
axes[1].set_ylabel('Unified score')
axes[1].set_title('Score = 0.5 × clean + 0.5 × robust (simulated)')
axes[1].set_ylim(0, 0.75)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

print('Best simulated unified score:', unified.max())
print('Notice: highest clean (ERM) is NOT best unified score.')
""")

md("""---
# Part 7 — Metric Explained

$$\\text{Score} = 0.5 \\times \\text{clean\\_accuracy} + 0.5 \\times \\text{robustness\\_accuracy}$$

| Term | Meaning |
|------|---------|
| Clean accuracy | Correct on normal test images |
| Robustness accuracy | Correct on adversarially perturbed test images |
| Gate | Clean must be **> 50%** |

**Example:** 80% clean + 40% robust → score = 0.5×0.8 + 0.5×0.4 = **0.60**

**Leaderboard:** 30% public / 70% private — optimize for generalization, not public overfitting.
""")

code("""# Unified score contour — pick your operating point
clean_grid = np.linspace(50, 100, 100)
robust_grid = np.linspace(0, 100, 100)
C, R = np.meshgrid(clean_grid, robust_grid)
Z = 0.5 * C/100 + 0.5 * R/100

plt.figure(figsize=(8, 5))
cs = plt.contourf(C, R, Z, levels=20, cmap='RdYlGn')
plt.colorbar(cs, label='Unified score')
plt.contour(C, R, Z, levels=[0.5, 0.55, 0.6, 0.65], colors='black', linewidths=0.8)
plt.xlabel('Clean accuracy (%)')
plt.ylabel('Robust accuracy (%)')
plt.title('Unified score landscape (score lines labeled)')
plt.axvline(50, color='red', ls='--', label='Rejection gate')
plt.legend()
plt.tight_layout()
plt.show()
""")

md("""---
# Part 8 — File Map (Planned Layout)

```
Assignment3/
├── Assignment_3_-_Robustness.pdf
├── task_template.py          ← from HuggingFace
├── submission.py             ← from HuggingFace
├── data/train.npz            ← from HuggingFace (gitignored)
├── src/                      ← model, data loading (future)
├── scripts/                  ← train, eval, attacks (future)
├── configs/                  ← hyperparameters (future)
├── results/checkpoints/      ← saved models (gitignored)
├── report/                   ← LaTeX CMS report (future)
├── README.md                 ← public reproduce guide (future)
└── _private/AI cach/         ← YOU ARE HERE (never submit)
```

**Course hub docs:** `TML_Summury/.../assignment-03-robustness/` (8 focused markdown files).
""")

md("""---
# Part 9 — Local GPU Workflow (When Implementation Starts)

```powershell
cd "...\\Assignments\\Assignment3"
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# 1) Download data from HuggingFace
# huggingface-cli download SprintML/tml26_task3 --local-dir .

# 2) Verify template
python task_template.py

# 3) Train baseline (future)
python scripts/train_standard.py --device cuda

# 4) Train PGD-AT (future)
python scripts/train_pgd_at.py --device cuda --config configs/pgd.yaml

# 5) Submit
# Edit submission.py: API_KEY, MODEL_PATH, MODEL_NAME
python submission.py
```

**Local eval before submit:** always run PGD attack on validation set with **more steps than training**.
""")

md("""---
# Part 10 — HPC Cluster Workflow

PGD-AT can take **10–20×** longer than normal training. Use cluster for full runs.

See `hpc-runbook.md` in course hub for:
- HTCondor submit templates
- Checkpoint/resume strategy
- Parallel hyperparameter sweeps
- scp best model back for submission

Reference: `Tutorials/HPC_Cluster_Guide.pdf`
""")

md("""---
# Part 11 — Learning Checklist

## Concepts (exam + report)

- [ ] Define adversarial example
- [ ] Explain ε-bounded perturbation
- [ ] White-box vs black-box attacks
- [ ] FGSM vs PGD
- [ ] Adversarial training min-max formulation
- [ ] Clean vs robust accuracy trade-off
- [ ] Why obfuscated gradients fail (Athalye)
- [ ] Certified robustness (high level)
- [ ] Public vs private leaderboard

## Skills (implementation phase)

- [ ] Load train.npz and preprocess uint8 → tensor
- [ ] Adapt ResNet for 32×32 / 9 classes
- [ ] Implement FGSM + PGD attack
- [ ] Implement PGD adversarial training loop
- [ ] Save/load state_dict correctly
- [ ] Local robust evaluation
- [ ] Submit via submission.py

## Deliverables

- [ ] Best `.pt` on leaderboard
- [ ] 2-page ICLR report on CMS
- [ ] Public GitHub repo (no weights)
- [ ] README with reproduce steps only
""")

md("""---
# Part 12 — Action Plan (Personal Timeline)

**Today (7 Jun):** ✅ Read this notebook + course hub docs

| Date | Action |
|------|--------|
| 7–8 Jun | Download HF files; verify task_template |
| 9 Jun | ERM baseline + local PGD eval |
| 10–11 Jun | FGSM-AT + first leaderboard submit |
| 12–14 Jun | PGD-AT + hyperparameter tuning |
| 15 Jun | Final model selection |
| **16 Jun** | **Deadline 23:59** — last submit + CMS ZIP |

**Next command when ready to implement:**
```powershell
huggingface-cli download SprintML/tml26_task3 --local-dir "path\\to\\Assignment3"
```

---

*Documentation session 2026-06-07 — implementation intentionally deferred.*
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {nb_path} ({len(cells)} cells)")
