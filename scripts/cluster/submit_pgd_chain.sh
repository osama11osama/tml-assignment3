#!/usr/bin/env bash
# Auto-chain all 4 PGD chunks (sequential on 1 GPU — no manual resubmit).
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

DAG="scripts/cluster/condor/pgd_at_chain.dag"
if [[ ! -f "$DAG" ]]; then
  echo "Missing $DAG"
  exit 1
fi

echo "Submitting PGD-AT DAG (4×1h, auto-resume between chunks)..."
cd scripts/cluster/condor
condor_submit_dag -force pgd_at_chain.dag
cd "$BASE"
echo ""
echo "Monitor:"
echo "  condor_q -dagman"
echo "  condor_q"
echo "  bash scripts/cluster/check_pgd_progress.sh"
echo "  tail -f runlogs/pgd_at_chunk*.log"
