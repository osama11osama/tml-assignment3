# Experiment notes — Assignment 3

## Summary table

| ID | Method | Arch | Epochs | Clean | Robust | Unified (local) | **LB** | Submitted? |
|----|--------|------|--------|-------|--------|-----------------|--------|------------|
| exp001 | ERM | R18 | 100 | 97.28% | 0.24% | 0.487 | **0.486** | Yes |
| exp002 | FGSM-AT | R18 | 50 | 72.66% | 9.24% | 0.410 | — | No |
| exp003 | PGD-AT | R18 | 80 | 67.88% | 45.98% | 0.569 | **0.575** | Yes |
| exp004 | PGD-AT extend | R18 | 120 | 67.32% | 46.76% | 0.570 | — | No (plateau) |
| exp005 | TRADES β=6 | R18 | 40 | 74.24% | 41.76% | ~0.580 | **0.575571** | Yes |
| **exp007** | **TRADES β=8** | **R18** | **20** | **72.34%** | **44.82%** | **0.5858 est.** | **0.582405** | **Yes — BEST** |
| exp006a | TRADES β=4 | R18 | 20 | — | — | ~0.536 | — | No (worse) |
| exp006 | ERM | R34 | 100 | ~97% | ~0.2% | ~0.49 | — | Init only |

---

## Best result — TRADES β=8 (2026-06-14)

- **Checkpoint:** `trades_b8_resnet18.pt`
- **Init:** `trades_r18_resnet18.pt` (not PGD directly)
- **Key change:** `trades_beta=8`, `lr=0.003`, 20 epochs
- **Server LB:** 0.582405 (+0.006834 vs TRADES β=6)
- **Doc:** `docs/TRADES_B8_RESULTS.md`
- **Git tag:** `v1.2-submittable`

**Local vs server gap:** quick eval 0.5858 → server 0.5824 (~−0.0034). Use local scores to **rank**, not to predict exact LB.

---

## Week 1 tracks (2026-06-14)

| Tag | Track | Status |
|-----|-------|--------|
| `trades_r18` | TRADES β=6, 40ep from PGD | Done — LB 0.575571 |
| **`trades_b8`** | **TRADES β=8, 20ep from trades_r18** | **Done — LB 0.582405 (BEST)** |
| `trades_b4` | TRADES β=4 pilot | Done — local ~0.536, skip LB |
| `erm_r34` | ERM ResNet-34 100ep | Done — init for Week 2 |
| `trades_r34` | TRADES on ERM R34 | In progress / Week 2 |

Runbook: `docs/CLUSTER_WEEK1.md` | Best result: `docs/TRADES_B8_RESULTS.md`

---

## Ideas queue

| Priority | Experiment | Status |
|----------|------------|--------|
| P0 | TRADES β=8 submit | **Done — LB 0.5824** |
| P1 | TRADES R34 (after ERM R34) | Week 2 — target > 0.582 |
| P2 | Extend β=8 to 40 ep | Only if cluster time |
| P3 | CMS PDF report | **Still needed** |
