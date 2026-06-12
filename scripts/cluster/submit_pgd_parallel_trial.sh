#!/usr/bin/env bash
# Submit 2 GPU jobs IN PARALLEL — compare PGD-7 vs PGD-10 after epoch 20.
# Uses 2 cluster GPUs at once. Pick winner, then continue that tag's chunks.
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

echo "Submitting parallel epoch-20 trials (2 GPUs)..."
J1=$(condor_submit scripts/cluster/condor/train_pgd_at_chunk20.sub | grep -o '[0-9]*' | tail -1)
J2=$(condor_submit scripts/cluster/condor/train_pgd_at_chunk20_s10.sub | grep -o '[0-9]*' | tail -1)

echo "  Main (PGD-7, batch 256): job $J1  -> tag pgd_at"
echo "  Trial (PGD-10):            job $J2  -> tag pgd_at_s10"
echo ""
condor_q
echo ""
echo "When both finish (~1h):"
echo "  bash scripts/cluster/compare_pgd_trials.sh"
echo "Then continue winner:"
echo "  bash scripts/cluster/submit_pgd_next.sh"
echo "  # or for s10 winner: TML3_TAG=pgd_at_s10 bash scripts/cluster/submit_pgd_next.sh"
