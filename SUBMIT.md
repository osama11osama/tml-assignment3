# Submission checklist — Assignment 3

## Leaderboard (model `.pt`)

- [ ] Model: `resnet18`, `resnet34`, or `resnet50`
- [ ] Input `3×32×32`, output **9 logits**
- [ ] Clean accuracy **> 50%** on server
- [ ] Save `state_dict` only (not full model)
- [ ] `submission.py`: `API_KEY`, `MODEL_PATH`, `MODEL_NAME` set
- [ ] 60-minute cooldown between successful uploads

## CMS ZIP (deadline: 16 Jun 2026 23:59)

- [ ] Code + 2-page report (ICLR template, no abstract)
- [ ] Matriculation number + CMS team ID in report
- [ ] README explains how to reproduce **best leaderboard result only**
- [ ] **No** model weights, `train.npz`, `.env`, or `.venv` in ZIP

## This private repo

Safe to keep `_private/` and documentation here.  
Never commit `.env` or trained `.pt` checkpoints.
