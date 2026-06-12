# Experiment notes — Assignment 3

## Summary table

| ID | Method | Epochs | Clean (val) | Robust PGD-20 | Unified | Submitted? | Notes |
|----|--------|--------|-------------|---------------|---------|------------|-------|
| exp001 | ERM smoke | 2 | **55.34%** | 3.16% | 0.293 | No | Best clean so far; short run |
| exp002 | ERM resume test | 3 | 40.86% | 6.22% | 0.235 | No | Resume OK; overwrote submit ckpt |
| exp003 | ERM full | 100 | — | — | — | No | **TODO — completes Step 1** |

Full write-up: [docs/STEP1_ERM_RESULTS.md](../docs/STEP1_ERM_RESULTS.md)

---

## Ideas queue

| Priority | Experiment | Status |
|----------|------------|--------|
| P0 | ERM 100 epochs | Pending |
| P1 | FGSM adversarial training | Not started |
| P2 | PGD adversarial training | Not started |
| P3 | First leaderboard submit | Not started |
