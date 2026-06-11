# AI cach — Assignment 3 documentation index

> **START HERE:** open `assignment3_master_guide.ipynb` and read top-to-bottom once.

| File | Purpose |
|------|---------|
| **`assignment3_master_guide.ipynb`** | Full workshop: theory, metric, attacks, defenses, plan |
| `00_project_scan.md` | Current project state |
| `01_master_plan.md` | Phased roadmap (no code yet) |
| `CHANGELOG.md` | What changed each session |
| `pdf_extracts/` | Text from assignment PDF, Tutorial 6, lecture 05 |

## Course hub (structured docs, no duplication)

Also see the parallel doc set (one topic per file):

`..\..\..\TML_Summury and bette understand Material\assignments\assignment-03-robustness\`

| Doc | Read when |
|-----|-----------|
| `README.md` | Navigation hub |
| `task-summary.md` | What the task is in plain language |
| `glossary.md` | Terms + Arabic equivalents |
| `references.md` | Papers and external links |
| `implementation-roadmap.md` | Future attack/defense stages |
| `submission-checklist.md` | Before every upload |
| `report-plan.md` | Writing the 2-page CMS report |
| `hpc-runbook.md` | GPU cluster for long training |
| `experiment-summary.md` | Log results as you go |

## End goal

**Assignment 3:** train ResNet-18/34/50 on 9-class 32×32 images → submit `.pt` state dict → maximize **Score = 0.5 × clean_acc + 0.5 × robust_acc** → CMS ZIP with code + report.

*Created: 2026-06-07 — documentation phase only (no implementation yet)*
