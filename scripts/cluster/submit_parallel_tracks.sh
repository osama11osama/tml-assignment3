#!/usr/bin/env bash
# Submit multiple parallel GPU tracks (each uses a DIFFERENT tag — never duplicate tags).
#
# Usage:
#   bash scripts/cluster/submit_parallel_tracks.sh          # default: max4
#   bash scripts/cluster/submit_parallel_tracks.sh max4     # 4 GPUs — recommended
#   bash scripts/cluster/submit_parallel_tracks.sh extend   # TRADES R18 60+80 only
#   bash scripts/cluster/submit_parallel_tracks.sh sweep    # beta 4 + beta 8 pilots
#   bash scripts/cluster/submit_parallel_tracks.sh r34      # TRADES R34 chunk 20
#   bash scripts/cluster/submit_parallel_tracks.sh erm      # ERM R34 chunk 80 only
#
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"
USER="${TML3_CLUSTER_USER:-atml_team044}"
MODE="${1:-max4}"

fix_subs() {
  sed -i "s/YOUR_CLUSTER_USER/${USER}/g" scripts/cluster/condor/train_erm_r34_chunk*.sub 2>/dev/null || true
  sed -i "s/YOUR_CLUSTER_USER/${USER}/g" scripts/cluster/condor/train_trades_r18_chunk*.sub 2>/dev/null || true
  sed -i "s/YOUR_CLUSTER_USER/${USER}/g" scripts/cluster/condor/train_trades_r34_chunk*.sub 2>/dev/null || true
  sed -i "s/YOUR_CLUSTER_USER/${USER}/g" scripts/cluster/condor/train_trades_b4_r18_chunk*.sub 2>/dev/null || true
  sed -i "s/YOUR_CLUSTER_USER/${USER}/g" scripts/cluster/condor/train_trades_b8_r18_chunk*.sub 2>/dev/null || true
}

fix_subs

echo "=== Parallel tracks mode: $MODE ==="
echo ""
echo "RULE: never run two jobs on the SAME tag at once."
echo "      trades_r18 | erm_r34 | trades_r34 | trades_b4 | trades_b8 are independent."
echo ""

submit() {
  local sub="$1"
  echo ">> condor_submit $sub"
  condor_submit "$sub"
}

case "$MODE" in
  erm)
    submit scripts/cluster/condor/train_erm_r34_chunk80.sub
    ;;
  extend)
    submit scripts/cluster/condor/train_trades_r18_chunk60.sub
    echo "After 60 finishes: condor_submit scripts/cluster/condor/train_trades_r18_chunk80.sub"
    ;;
  r34)
    submit scripts/cluster/condor/train_trades_r34_chunk20.sub
    echo "After 20 finishes: condor_submit scripts/cluster/condor/train_trades_r34_chunk40.sub"
    ;;
  sweep)
    submit scripts/cluster/condor/train_trades_b4_r18_chunk20.sub
    submit scripts/cluster/condor/train_trades_b8_r18_chunk20.sub
    ;;
  max4|all|*)
    echo "Launching 4 parallel tracks (use up to 4 free GPUs):"
    echo "  GPU1 ERM R34 -> epoch 80"
    echo "  GPU2 TRADES R34 -> epoch 20 (init erm_r34 ep60+)"
    echo "  GPU3 TRADES beta=4 pilot (20 ep)"
    echo "  GPU4 TRADES beta=8 pilot (20 ep)"
    echo ""
    echo "Optional 5th GPU (separate): bash scripts/cluster/submit_parallel_tracks.sh extend"
    echo ""
    submit scripts/cluster/condor/train_erm_r34_chunk80.sub
    submit scripts/cluster/condor/train_trades_r34_chunk20.sub
    submit scripts/cluster/condor/train_trades_b4_r18_chunk20.sub
    submit scripts/cluster/condor/train_trades_b8_r18_chunk20.sub
    ;;
esac

echo ""
condor_q
echo ""
echo "Monitor (pick logs as they appear):"
echo "  tail -f runlogs/erm_r34_chunk80_*.log"
echo "  tail -f runlogs/trades_r34_chunk20_*.log"
echo "  tail -f runlogs/trades_b4_chunk20_*.log"
echo "  tail -f runlogs/trades_b8_chunk20_*.log"
echo ""
echo "After jobs finish — pull + eval on Windows:"
echo "  powershell -File scripts/cluster/sync_from_cluster.ps1 -ClusterUser $USER -Force"
echo "  python scripts/eval_archive.py"
echo ""
echo "Submit to LB only if local unified > 0.575571 (current best: trades_r18)"
