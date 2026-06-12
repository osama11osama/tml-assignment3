#!/usr/bin/env bash
# Submit the next PGD-AT 1-hour chunk (after previous job finished or was killed).
# Usage: TML3_TAG=pgd_at bash submit_pgd_next.sh
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

TAG="${TML3_TAG:-pgd_at}"

bash scripts/cluster/check_pgd_progress.sh "$TAG" | tee /tmp/pgd_next_${TAG}.txt
NEXT=$(grep "Next 1-hour job target: epoch" /tmp/pgd_next_${TAG}.txt | grep -o '[0-9]*$' || true)

if [[ -z "$NEXT" ]]; then
  echo "Nothing to submit for tag=$TAG."
  exit 0
fi

if [[ "$TAG" == "pgd_at" ]]; then
  SUB="scripts/cluster/condor/train_pgd_at_chunk${NEXT}.sub"
else
  SUB="scripts/cluster/condor/train_pgd_at_chunk${NEXT}_${TAG}.sub"
  if [[ ! -f "$SUB" ]]; then
    echo "No variant sub file: $SUB"
    echo "Create it from train_pgd_at_chunk20_s10.sub template, or use main tag pgd_at."
    exit 1
  fi
fi

if [[ ! -f "$SUB" ]]; then
  echo "Missing $SUB"
  exit 1
fi

echo "Submitting $SUB (tag=$TAG -> epoch $NEXT)"
condor_submit "$SUB"
condor_q
