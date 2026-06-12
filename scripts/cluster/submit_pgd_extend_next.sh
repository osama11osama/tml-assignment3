#!/usr/bin/env bash
# Submit next Phase-4 chunk (epochs 100 or 120) after pgd_at finished epoch 80.
# Usage: bash scripts/cluster/submit_pgd_extend_next.sh
set -eu

export TML3_FINAL_EPOCHS="${TML3_FINAL_EPOCHS:-120}"
export TML3_TAG="${TML3_TAG:-pgd_at}"
export TML3_PGD_TRAIN_STEPS="${TML3_PGD_TRAIN_STEPS:-10}"

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

bash scripts/cluster/check_pgd_progress.sh "$TML3_TAG" | tee /tmp/pgd_extend_next.txt
NEXT=$(grep "Next 1-hour job target: epoch" /tmp/pgd_extend_next.txt | grep -o '[0-9]*$' || true)

if [[ -z "$NEXT" ]]; then
  echo "Nothing to submit (already at epoch $TML3_FINAL_EPOCHS?)."
  exit 0
fi

SUB="scripts/cluster/condor/train_pgd_at_chunk${NEXT}.sub"
if [[ ! -f "$SUB" ]]; then
  echo "Missing $SUB — use manual condor_submit for epoch $NEXT"
  exit 1
fi

echo "Phase 4: submitting $SUB (tag=$TML3_TAG -> epoch $NEXT, PGD-${TML3_PGD_TRAIN_STEPS})"
condor_submit "$SUB"
condor_q
