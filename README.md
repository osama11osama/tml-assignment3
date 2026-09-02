# Trustworthy Machine Learning — Adversarial Robustness

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-adversarial%20training-red)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/topic-adversarial%20robustness-purple)](#project-overview)
[![Use](https://img.shields.io/badge/use-research%20%26%20education-orange)](#responsible-use--legal-notice)

A research project for studying **adversarial robustness in image classification**.

The project investigates how standard and adversarial training strategies affect the trade-off between clean accuracy and robustness against bounded adversarial perturbations. It was developed in an academic Trustworthy Machine Learning setting and is published as a **research, reproducibility, and portfolio artifact**.

> This repository is intended for lawful education, research, benchmarking, and authorized ML-security evaluation only.

---

## Project overview

Modern neural networks can achieve strong accuracy while remaining sensitive to carefully constructed perturbations that are small under a chosen norm constraint.

This project explores the question:

> **How can we train a classifier that remains useful on clean data while becoming more resistant to adversarial examples?**

The experimental progression includes:

- standard empirical risk minimization (ERM);
- FGSM adversarial training;
- PGD adversarial training;
- TRADES-style robust optimization;
- robustness evaluation under iterative attacks;
- comparison of clean/robust accuracy trade-offs;
- reproducible experiment configurations and checkpoints.

---

## Why this matters

Adversarial robustness is relevant to:

- trustworthy machine learning;
- safety-critical ML deployment;
- robustness benchmarking;
- secure perception systems;
- model evaluation under worst-case perturbations;
- understanding the difference between average-case accuracy and adversarial behavior.

Adversarial examples are also a useful scientific tool for studying model decision boundaries and failure modes.

---

## Experimental setup

The original task used small RGB images with 9 output classes and ResNet-family classifiers.

The evaluation metric balanced clean and robust performance:

```text
score = 0.5 × clean_accuracy + 0.5 × robustness_accuracy
```

This encourages models that do not maximize robustness by sacrificing all normal predictive performance.

---

## Training approaches

### 1. Standard training (ERM)

A conventional clean-data baseline establishes how well the architecture performs without robustness-specific optimization.

### 2. FGSM adversarial training

Fast Gradient Sign Method (FGSM) provides a lightweight way to expose the model to adversarially perturbed samples during training.

It is useful as a simple baseline, although single-step training can be insufficient against stronger iterative attacks.

### 3. PGD adversarial training

Projected Gradient Descent (PGD) adversarial training uses iterative perturbation generation and is a stronger robustness baseline.

Conceptually:

```text
clean batch
    ↓
generate bounded adversarial examples with PGD
    ↓
train model on adversarial / mixed objective
    ↓
evaluate clean + robust accuracy
```

### 4. TRADES

TRADES explicitly optimizes the trade-off between natural accuracy and robustness by combining standard classification loss with a robustness-oriented divergence term.

This repository includes multiple TRADES configurations and experiments with different beta values and model variants.

---

## Recorded experimental progress

Examples of recorded results from the original evaluation environment include:

| Experiment | Recorded result |
|---|---:|
| Standard ERM baseline | 0.486371 |
| PGD adversarial training | 0.575136 |
| TRADES β=6 | 0.575571 |
| TRADES β=8 ResNet-18 | **0.582405** |

These values belong to the original task/evaluation setup and should not be interpreted as universal robustness guarantees.

Robustness numbers are meaningful only together with the exact:

- attack parameters;
- perturbation budget;
- preprocessing;
- dataset;
- model checkpoint;
- norm definition;
- number of attack steps;
- evaluation implementation.

---

## Repository structure

```text
tml-assignment3/
├── README.md
├── requirements.txt
├── .env.example
├── configs/                 # experiment configurations
├── src/                     # models, data and training utilities
├── scripts/                 # training / evaluation workflows
├── docs/                    # experiment and cluster notes
├── results/                 # local/generated result structure
└── hf_download/             # reference task helpers where applicable
```

The repository also contains historical development notes from the original academic workflow. Before republishing or redistributing third-party course material, verify that you have the right to do so.

---

## Getting started

Create and activate a Python environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install PyTorch appropriate for your hardware, then install project dependencies:

```powershell
pip install torch torchvision
pip install -r requirements.txt
```

For CUDA systems, use the PyTorch build that matches your installed environment.

The typical workflow is:

```text
prepare authorized dataset
        ↓
verify preprocessing/model pipeline
        ↓
train clean baseline
        ↓
train adversarial variants
        ↓
evaluate clean and robust accuracy
        ↓
compare configurations
```

---

## Configuration-driven experiments

The repository contains configuration files for several training strategies, including:

```text
configs/
├── standard_erm.yaml
├── fgsm_at.yaml
├── pgd_at.yaml
├── pgd_at_extend.yaml
├── trades_r18.yaml
├── trades_b8.yaml
├── trades_r34.yaml
└── erm_r34.yaml
```

Keeping configurations separate from the implementation makes it easier to reproduce and compare experiments without changing code for every run.

---

## Reproducibility considerations

Robust training is especially sensitive to implementation details. Results may change with:

- random seed;
- model initialization;
- train/validation split;
- optimizer and learning-rate schedule;
- attack step size;
- attack iteration count;
- perturbation radius;
- data augmentation;
- batch size;
- PyTorch/CUDA version;
- GPU numerical behavior.

For meaningful comparisons, record the full threat model and evaluation configuration rather than reporting only a single robustness number.

---

## Defensive perspective

Adversarial robustness research helps developers avoid a common mistake: assuming that high clean accuracy implies reliable behavior under deliberate input manipulation.

Practical lessons include:

- define the threat model before claiming robustness;
- evaluate with sufficiently strong attacks;
- avoid reporting robustness against only the attack used during training;
- watch for gradient masking or evaluation artifacts;
- preserve clean accuracy measurements alongside robust accuracy;
- test multiple attack settings when possible;
- treat empirical robustness as conditional evidence, not a proof of security.

No training method in this repository should be interpreted as making a model universally secure.

---

## Responsible use & legal notice

This repository is provided for **lawful educational, academic, defensive, benchmarking, and authorized ML-security research purposes only**.

By using the project, you are responsible for ensuring compliance with applicable laws, institutional rules, licenses, contracts, dataset terms, model terms, privacy requirements, and authorization scopes.

You must not use this repository to:

- attack or interfere with third-party systems without authorization;
- bypass access controls or platform restrictions;
- use adversarial techniques to cause harm or evade legitimate safety/security controls;
- evaluate models or datasets you do not have permission to access;
- misrepresent experimental robustness as a formal security guarantee;
- treat publication of this repository as authorization to test third-party services.

The author does **not** authorize unlawful, abusive, or unauthorized use and is not responsible for how third parties choose to use or modify this work.

### Disclaimer of warranty and liability

All software, configurations, research notes, experiments, and results are provided **"as is"**, without warranties or guarantees of any kind.

Robustness results are environment- and threat-model-specific and may be incomplete, outdated, non-reproducible, or unsuitable for a particular operational purpose.

To the maximum extent permitted by applicable law, the author shall not be liable for damages, claims, service disruption, model failure, data loss, financial loss, regulatory consequences, or other outcomes arising from use, misuse, modification, or redistribution of this project.

Nothing in this repository constitutes legal advice, safety certification, or a guarantee that a model is secure. No disclaimer can guarantee complete exclusion of liability in every jurisdiction.

---

## Academic integrity

This repository originated from academic work and is published for learning, reproducibility, and portfolio purposes.

If you are currently taking a course with the same or a similar task:

- follow your institution's academic-integrity rules;
- do not submit this implementation or derived solutions as your own where reuse is prohibited;
- cite reused code, experiments, and ideas where appropriate;
- ask the course staff if consulting public repositories is allowed.

A public repository does not override academic rules.

---

## Publication / copyright note

Code written for this project can be documented and shared subject to the rights you hold in it. However, course PDFs, lecture/tutorial extracts, third-party datasets, model weights, and copied reference material may have separate copyright or redistribution terms.

Before making the repository broadly public, review such files and remove anything you do not have permission to redistribute.

---

## Author

**Osama Altamar**  
Cybersecurity and software engineering — interests include adversarial machine learning, ML security, privacy, secure systems, and defensive research.

GitHub: [@osama11osama](https://github.com/osama11osama)

---

## Related projects

- [`tml-assignment1`](https://github.com/osama11osama/tml-assignment1) — membership inference / ML privacy
- [`tml-assignment2`](https://github.com/osama11osama/tml-assignment2) — stolen/derived model detection
- [`Sidechannel-timing-attack-starter`](https://github.com/osama11osama/Sidechannel-timing-attack-starter) — timing side-channel demonstration
