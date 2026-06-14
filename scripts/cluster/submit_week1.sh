#!/usr/bin/env bash
# Submit Week 1 parallel tracks on cluster (2 GPUs if available).
#
# Track A: ERM ResNet-34 chunk 20
# Track B: TRADES R18 chunk 20 (init pgd_at on cluster)
#
# Usage: bash scripts/cluster/submit_week1.sh
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

sed -i 's/YOUR_CLUSTER_USER/atml_team044/g' scripts/cluster/condor/train_erm_r34_chunk*.sub 2>/dev/null || true
sed -i 's/YOUR_CLUSTER_USER/atml_team044/g' scripts/cluster/condor/train_trades_r18_chunk*.sub 2>/dev/null || true

echo "=== Week 1 — submitting Track A (ERM R34) + Track B (TRADES R18) ==="
echo ""

if [[ ! -f results/checkpoints/pgd_at_resnet18.pt ]]; then
  echo "WARN: missing results/checkpoints/pgd_at_resnet18.pt — TRADES init may fail"
fi

condor_submit scripts/cluster/condor/train_erm_r34_chunk20.sub
condor_submit scripts/cluster/condor/train_trades_r18_chunk20.sub

echo ""
condor_q
echo ""
echo "Monitor:"
echo "  tail -f runlogs/erm_r34_chunk20_*.log"
echo "  tail -f runlogs/trades_r18_chunk20_*.log"
echo ""
echo "After each finishes, submit next chunk manually or see docs/CLUSTER_WEEK1.md"
