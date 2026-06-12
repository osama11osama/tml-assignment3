#!/usr/bin/env bash
# Verify Assignment 3 cluster layout before submitting GPU jobs.
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

echo "=== Assignment 3 cluster setup check ==="
echo "Base: $BASE"

fail=0
check_file() {
  if [[ -f "$1" ]]; then
    echo "  OK  $1"
  else
    echo "  MISSING  $1"
    fail=1
  fi
}

check_file "data/train.npz"
check_file "scripts/train_fgsm_at.py"
check_file "scripts/train_pgd_at.py"
check_file "scripts/train_standard.py"
check_file "src/attacks.py"
check_file "src/train_utils.py"

INIT="${TML3_INIT:-results/checkpoints/fgsm_at_resnet18.pt}"
if [[ -f "$INIT" ]]; then
  echo "  OK  $INIT (PGD-AT warm-start)"
elif [[ -f results/checkpoints/baseline_erm_resnet18.pt ]]; then
  echo "  WARN  fgsm_at missing; baseline_erm_resnet18.pt available as fallback init"
else
  echo "  WARN  No init checkpoint - upload fgsm_at or baseline ERM"
fi

mkdir -p results/runs results/checkpoints runlogs

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- nvidia-smi ---"
  nvidia-smi -L || true
else
  echo "  (no nvidia-smi on login node - normal)"
fi

if [[ "$fail" -ne 0 ]]; then
  echo "Setup incomplete. See docs/CLUSTER_STEP2.md"
  exit 1
fi
echo "Setup OK."
