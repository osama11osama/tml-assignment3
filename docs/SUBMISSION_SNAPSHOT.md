# Submission Snapshot — v1.0 (CMS + Leaderboard)

> **Release:** `v1.0-submittable`  
> **Date:** 2026-06-12  
> **Assignment:** TML SS2026 — Task `03-robustness`  
> **Deadline:** 16 June 2026, 23:59

This document describes the **current best leaderboard result** you can cite in the CMS report and reproduce from this repo.

---

## Best model (leaderboard)

| Field | Value |
|-------|-------|
| **Checkpoint** | `results/checkpoints/pgd_at_resnet18.pt` |
| **Architecture** | `resnet18` |
| **Method** | PGD adversarial training (Madry et al.) |
| **Training** | 80 epochs; init from FGSM-AT; PGD-7 train / PGD-20 eval |
| **Server unified score** | **0.575136** |
| **Team name** | `team_XLVII` |
| **Rank (approx.)** | ~24 / 35 |

---

## Score progression

| Submit | Model | Server unified |
|--------|-------|----------------|
| 1 | ERM baseline | 0.486371 |
| 2 | PGD-AT (epoch 20) | 0.535790 |
| 3 | **PGD-AT (epoch 80)** | **0.575136** ← **report this** |

---

## Local validation (epoch 80, full val PGD-20)

| Metric | Value |
|--------|-------|
| Clean accuracy | 67.88% |
| Robust accuracy | 45.98% |
| Unified | 0.5693 |

Server score (0.575) can differ slightly — hidden test attack parameters (Tutorial 6).

---

## How to reproduce (CMS README section)

1. **Data:** `python scripts/download_data.py` → `data/train.npz`
2. **Step 1 ERM:** `python scripts/train_standard.py --device cuda --epochs 100`
3. **Step 2 FGSM-AT:** `python scripts/train_fgsm_at.py --device cuda --init results/checkpoints/baseline_erm_resnet18.pt`
4. **Step 3 PGD-AT:** `python scripts/train_pgd_at.py --device cuda --init results/checkpoints/fgsm_at_resnet18.pt --epochs 80`
   - Or cluster chunks: `docs/CLUSTER_STEP3.md`
5. **Eval:** `python scripts/eval_model.py results/checkpoints/pgd_at_resnet18.pt --architecture resnet18`
6. **Submit:** `submission.py` or `_private/tools/launch_submit_gui.ps1`

**Attack params (assignment standard):** ε = 8/255, α = 2/255, PGD-20 for evaluation.

---

## CMS ZIP contents (checklist)

Include:

- [ ] Source code (`src/`, `scripts/`, `configs/`)
- [ ] `README.md` + this file (`docs/SUBMISSION_SNAPSHOT.md`)
- [ ] 2-page PDF report (ICLR template, no abstract)
- [ ] Matriculation number + CMS team ID in report

Do **not** include:

- [ ] `data/train.npz`, `.pt` weights, `.env`, `.venv/`

---

## Report talking points

1. **Baseline ERM:** high clean (~97%), ~0% robust → unified ~0.49.
2. **FGSM-AT:** lifts robust slightly but hurts clean too much (unified 0.41).
3. **PGD-AT:** main method; 80-epoch schedule with FGSM init; unified **0.575** on server.
4. **Trade-off:** adversarial training reduces clean accuracy but greatly increases robustness.
5. **References:** Goodfellow FGSM, Madry PGD-AT, Athalye obfuscated gradients (no masking tricks).

---

## Version history

| Version | Date | Best LB | Notes |
|---------|------|---------|-------|
| **v1.0** | 2026-06-12 | **0.575136** | PGD-AT ResNet18, 80 epochs — **submittable now** |
| v1.1 (planned) | — | target 0.60+ | Phase 4: extend to epoch 120, PGD-10 — `docs/CLUSTER_STEP4.md` |

---

*For full training logs and cluster workflow, see `docs/STEP3_PGD_RESULTS.md` and `docs/AI_AGENT_HANDOFF.md`.*
