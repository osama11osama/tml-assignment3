# Master plan — Assignment 3: zero to CMS ZIP

> Goal: maximize **unified score** + deep understanding + 2-page report  
> **Current phase: documentation only — no code yet**

---

## Phase 0 — Foundation (NOW) ✅

- [x] Extract assignment PDF + Tutorial 6 + lecture slides
- [x] Create `_private/AI cach/` documentation structure
- [x] Create course hub docs in `TML_Summury/.../assignment-03-robustness/`
- [x] Master guide notebook
- [ ] Download HuggingFace starter files
- [ ] Set up public repo layout (mirror Assignment 2 structure)

---

## Phase 1 — Environment & data

```powershell
# Future commands (not run yet)
cd "...\Assignments\Assignment3"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download from HuggingFace: SprintML/tml26_task3
# Expected: train.npz, task_template.py, submission.py
```

Checklist:
- [ ] Verify `train.npz` loads (50k images, 9 classes)
- [ ] Run assertions in `task_template.py`
- [ ] Set `TML_API_KEY` in `.env`
- [ ] Confirm GPU available (`cuda`)

---

## Phase 2 — Experiments (planned order)

### Exp 000 — Standard training (baseline)

**Purpose:** establish clean accuracy ceiling and confirm robust acc ≈ 0 under PGD.

| Hyperparameter | Starting guess |
|----------------|----------------|
| Architecture | ResNet-18 (CIFAR-adapted) |
| Optimizer | SGD + momentum |
| Epochs | 100–200 |
| Augmentation | Random crop + flip |

**Script (future):** `scripts/train_standard.py`  
**Output:** `results/checkpoints/baseline.pt`

### Exp 001 — FGSM adversarial training

**Purpose:** fast first robust model; tutorial recommends as starting point.

- Generate FGSM perturbations during training
- Train on adversarial examples
- Evaluate with local PGD attack

**Script (future):** `scripts/train_fgsm_at.py`

### Exp 002 — PGD adversarial training (Madry et al.)

**Purpose:** main workhorse — stronger adversarial examples → better robustness.

- Inner loop: multi-step PGD to maximize loss
- Outer loop: SGD on adversarial batch
- Tune ε, step size α, number of steps

**Script (future):** `scripts/train_pgd_at.py`

### Exp 003 — Hyperparameter search

Tune for **unified score**, not robust alone:

| Knob | Effect |
|------|--------|
| ε (perturbation budget) | Higher ε → harder training → more robust, lower clean |
| PGD steps | More steps → stronger attack → better robust generalization |
| Learning rate schedule | Stability vs convergence |
| Weight decay | Regularization |
| Architecture | ResNet-34/50 for capacity |
| Data augmentation | May help clean without hurting robust |

### Exp 004 — Advanced (only if needed)

- TRADES (robustness + natural accuracy trade-off)
- Early stopping on validation robust acc
- Ensemble / wider models
- Input normalization matching server expectations

---

## Phase 3 — Evaluation loop (local before submit)

```
1. Train → save state_dict
2. Local eval:
   - clean accuracy on held-out split (from train or val split)
   - robust accuracy under PGD (multiple ε values)
3. Run task_template.py assertions
4. submission.py --validate-only (if available)
5. Submit to server
6. Log public score; iterate
```

**Critical:** local attacks will NOT match server exactly. Use **strong, diverse** local eval.

---

## Phase 4 — Submission loop

1. Save best `state_dict` → `results/checkpoints/best.pt`
2. Set in `submission.py`: `API_KEY`, `MODEL_PATH`, `MODEL_NAME`
3. Run `python submission.py`
4. Wait 60 min cooldown
5. Compare leaderboard; keep best checkpoint

---

## Phase 5 — CMS deliverables

```
report/
├── paper.tex          # ICLR template, no abstract, max 2 pages
├── figures/           # clean vs robust trade-off plot
└── references.bib

ZIP contents:
├── All code (reproducible)
├── README.md          # ONLY how to reproduce best result
├── report.pdf
└── NO model weights, NO train.npz, NO .env
```

Report sections (from official PDF):
1. Introduction — task in your own words
2. Main body — approach, hyperparameters, leaderboard results
3. Conclusion — **practical implications of adversarial vulnerability in real systems**

---

## Priority order

```
P0  Read all docs + master guide notebook     ← you are here
P1  Download HF files + verify task_template
P2  Baseline standard training + local PGD eval
P3  FGSM adversarial training + first submit
P4  PGD adversarial training + hyperparameter tuning
P5  Final model + CMS report + ZIP
```

---

## Timeline (deadline: 16 June 2026)

| Week | Focus |
|------|-------|
| 7–8 Jun | Read, setup, baseline |
| 9–11 Jun | FGSM-AT + first submissions |
| 12–14 Jun | PGD-AT + tuning |
| 15–16 Jun | Final submit + CMS ZIP + report |

---

*Updated: 2026-06-07 — documentation phase*
