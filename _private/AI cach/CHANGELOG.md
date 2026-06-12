# Changelog — Assignment 3 documentation

## Session 1 — 2026-06-07

**User request:** Create full documentation/plan for Assignment 3 (Robustness), same structure as Assignments 1 & 2. No implementation yet.

**Created:**
- `_private/README.md` — local folder index
- `_private/AI cach/README.md` — documentation hub
- `_private/AI cach/00_project_scan.md` — project state
- `_private/AI cach/01_master_plan.md` — phased roadmap
- `_private/AI cach/pdf_extracts/` — text from 4 course PDFs
- `_private/AI cach/assignment3_master_guide.ipynb` — zero-to-hero notebook
- `_private/AI cach/_build_master_guide.py` — notebook generator
- `TML_Summury/.../assignment-03-robustness/` — full doc set (8 files)

## Session 2 — 2026-06-11

**Step 0 implementation:**
- Downloaded `train.npz`, `src/`, `scripts/`, verify pipeline
- CUDA PyTorch on RTX 5060

**Submit GUI (Assignment 2 pattern):**
- `_private/tools/tml_submit_gui.py` — Task 3 model upload GUI
- `_private/tools/launch_submit_gui.ps1`
- Config/history: `%APPDATA%\tml_submit_gui\task3\`

## Session 3 — 2026-06-11

**Documented Step 1 results:**
- `docs/STEP1_ERM_RESULTS.md` — full experiment log
- `experiments/exp_notes.md` — summary table
- Updated README + TML experiment-summary

**Step 1 verdict:** NOT complete (only 2–3 epoch tests; 100-epoch run pending)

**Recorded results:**
- exp001: clean 55.34%, robust 3.16%, unified 0.293 (2 epochs)
- exp002: clean 40.86%, robust 6.22%, unified 0.235 (3 epochs, resume test)

**GPU:** RTX 5060 + cuda confirmed; no full training running at doc time

**Not done yet:**
- ERM 100 epochs (exp003)
- FGSM / PGD adversarial training
- Leaderboard submission

**Sources used:**
- `Assignment_3_-_Robustness.pdf`
- `Tutorial_6_-_Assignment_3.pdf`
- `Tutorial_5_on_Model_Stealing_II_and_Robustness_(Slides).pdf`
- `Notes/05-AdversariaML_Robustness.pdf`
- Existing lecture study notes in course hub
