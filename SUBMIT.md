# Submission checklist — Assignment 3

## Current best (v1.0 — submittable)

| Field | Value |
|-------|-------|
| Checkpoint | `pgd_at_resnet18.pt` |
| Architecture | `resnet18` |
| Method | PGD adversarial training, 80 epochs |
| **Server unified** | **0.575136** |
| Full details | `docs/SUBMISSION_SNAPSHOT.md` |

---

## Leaderboard (model `.pt`)

- [x] Model: `resnet18`
- [x] Clean accuracy **> 50%** on server
- [x] Best submit: PGD-AT **0.575136**
- [ ] Phase 4 submit if unified **> 0.575** (after epoch 120)
- [ ] 60-minute cooldown between successful uploads

## CMS ZIP (deadline: 16 Jun 2026 23:59)

- [ ] Code + 2-page report (ICLR template, no abstract)
- [ ] Matriculation number + CMS team ID in report
- [ ] README + `docs/SUBMISSION_SNAPSHOT.md` explain how to reproduce **best LB result**
- [ ] **No** model weights, `train.npz`, `.env`, or `.venv` in ZIP

## This private repo

Safe to keep `_private/` and documentation here.  
Never commit `.env` or trained `.pt` checkpoints.
