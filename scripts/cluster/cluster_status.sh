#!/usr/bin/env bash
# Full cluster dashboard: jobs, all tracks, log tails, next steps.
#
# Usage:
#   cd ~/tml26_task3 && bash scripts/cluster/cluster_status.sh
#
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v python3 || command -v python)"
fi

LB_BEST="0.575571"
LB_CKPT="trades_r18_resnet18.pt"

echo "============================================================"
echo " TML Assignment 3 — Cluster status"
echo " $(date -Iseconds) | base=$BASE"
echo "============================================================"
echo ""

echo "=== Condor jobs (your queue) ==="
condor_q -nobatch 2>/dev/null || condor_q
echo ""

# tag:arch:final_epochs:chunks_csv:checkpoint_guess:next_sub_hint
TRACKS=(
  "erm_r34:resnet34:100:20,40,60,80,100:baseline_erm_r34_resnet34.pt:train_erm_r34_chunk"
  "trades_r34:resnet34:80:20,40,60,80:trades_r34_resnet34.pt:train_trades_r34_chunk"
  "trades_r18:resnet18:80:20,40,60,80:trades_r18_resnet18.pt:train_trades_r18_chunk"
  "trades_b4:resnet18:20:20:trades_b4_resnet18.pt:train_trades_b4_r18_chunk"
  "trades_b8:resnet18:20:20:trades_b8_resnet18.pt:train_trades_b8_r18_chunk"
  "pgd_at:resnet18:80:20,40,60,80:pgd_at_resnet18.pt:train_pgd_at_chunk"
)

print_track() {
  local spec="$1"
  IFS=':' read -r TAG ARCH FINAL CHUNKS CKPT SUBPFX <<< "$spec"
  local PROGRESS="results/runs/${TAG}_${ARCH}/progress.json"

  echo "------------------------------------------------------------"
  echo "Track: $TAG ($ARCH) → target epoch $FINAL"
  echo "Checkpoint: results/checkpoints/$CKPT"

  if [[ ! -f "$PROGRESS" ]]; then
    echo "  Status: NOT STARTED"
    local FIRST="${CHUNKS%%,*}"
    echo "  Next: condor_submit scripts/cluster/condor/${SUBPFX}${FIRST}.sub"
    echo "  (or: TML3_TAG=$TAG TML3_ARCH=$ARCH bash scripts/cluster/run_*_chunk.sh $FIRST)"
    return
  fi

  read -r STATUS LAST BEST CLEAN ROB <<< "$("$PYTHON" - "$PROGRESS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
last = p.get("last_epoch") or p.get("epochs_completed") or 0
best = p.get("best_unified_score", p.get("final_unified_score"))
status = p.get("status", "unknown")
clean = p.get("best_clean_acc", p.get("final_clean_acc"))
rob = p.get("final_robust_acc", p.get("last_robust_acc"))
def fmt(x):
    if x is None: return "?"
    if isinstance(x, float) and x <= 1: return f"{x*100:.2f}%"
    return str(x)
print(status, last, best if best is not None else "?", fmt(clean), fmt(rob))
PY
)"

  echo "  Status: $STATUS | epoch $LAST / $FINAL | unified: $BEST"
  echo "  clean: $CLEAN | robust: $ROB"

  # Latest training log for this tag
  local LOG
  LOG=$(ls -t runlogs/${TAG}_chunk*_[0-9]*.log runlogs/${TAG}_chunk*.log 2>/dev/null | head -1 || true)
  if [[ -n "$LOG" && -f "$LOG" ]]; then
    echo "  Latest log: $LOG"
    grep -E '^\[DONE\]|^TRAINING COMPLETE|Chunk complete|Training exited|Stopped at soft timeout' "$LOG" 2>/dev/null | tail -5 | sed 's/^/    /' || true
  fi

  if [[ "$STATUS" == "finished" && "$LAST" -ge "$FINAL" ]]; then
    if [[ "$TAG" == erm* ]]; then
      echo "  => DONE (ERM baseline — do NOT submit to LB)"
    elif [[ "$BEST" != "?" ]]; then
      "$PYTHON" - "$BEST" "$LB_BEST" <<'PY'
import sys
try:
    u = float(sys.argv[1])
    lb = float(sys.argv[2])
    if u > lb:
        print(f"  => DONE — unified {u:.4f} BEATS LB {lb} — sync + eval + submit!")
    else:
        print(f"  => DONE — unified {u:.4f} <= LB {lb} — keep {sys.argv[2]} for LB")
except (TypeError, ValueError):
    print("  => DONE — run local eval before LB submit")
PY
    else
      echo "  => DONE — sync locally and eval before LB"
    fi
    return
  fi

  local NEXT=""
  IFS=',' read -ra ARR <<< "$CHUNKS"
  for T in "${ARR[@]}"; do
    [[ "$T" -gt "$FINAL" ]] && continue
    if [[ "$LAST" -lt "$T" ]]; then
      NEXT="$T"
      break
    fi
  done
  [[ -z "$NEXT" && "$LAST" -lt "$FINAL" ]] && NEXT="$FINAL"

  if [[ -n "$NEXT" ]]; then
    local SUB="scripts/cluster/condor/${SUBPFX}${NEXT}.sub"
    if [[ -f "$SUB" ]]; then
      echo "  Next: condor_submit $SUB"
    else
      echo "  Next: TML3_TAG=$TAG TML3_ARCH=$ARCH TML3_FINAL_EPOCHS=$FINAL bash scripts/cluster/run_trades_chunk.sh $NEXT"
      echo "        (no .sub for epoch $NEXT — create or use run_*_chunk.sh)"
    fi
  fi
}

echo "=== All training tracks ==="
for spec in "${TRACKS[@]}"; do
  print_track "$spec"
done

echo ""
echo "============================================================"
echo " RECOMMENDED NEXT STEPS"
echo "============================================================"

# ERM done → push TRADES R34
if [[ -f results/runs/erm_r34_resnet34/progress.json ]]; then
  ERM_LAST=$("$PYTHON" -c "import json;p=json.load(open('results/runs/erm_r34_resnet34/progress.json'));print(p.get('last_epoch') or p.get('epochs_completed') or 0)")
  if [[ "$ERM_LAST" -ge 100 ]]; then
    echo "1. ERM R34 complete (100 ep) — init ready for TRADES R34"
    TR34_LAST=0
    if [[ -f results/runs/trades_r34_resnet34/progress.json ]]; then
      TR34_LAST=$("$PYTHON" -c "import json;p=json.load(open('results/runs/trades_r34_resnet34/progress.json'));print(p.get('last_epoch') or p.get('epochs_completed') or 0)")
    fi
    if [[ "$TR34_LAST" -lt 40 ]]; then
      echo "   condor_submit scripts/cluster/condor/train_trades_r34_chunk40.sub"
    elif [[ "$TR34_LAST" -lt 80 ]]; then
      echo "   Continue TRADES R34 toward epoch 80 (chunk subs or run_trades_chunk.sh)"
    fi
  fi
fi

echo "2. When condor_q empty — pull to Windows:"
echo "   powershell -File scripts/cluster/sync_from_cluster.ps1 -ClusterUser atml_team044 -Force"
echo "   python scripts/eval_archive.py"
echo ""
echo "3. LB submit ONLY if unified > $LB_BEST ($LB_CKPT)"
echo "4. Optional 5th GPU: bash scripts/cluster/submit_parallel_tracks.sh extend  # TRADES R18 -> 60"
echo ""
echo "Quick log tails:"
echo "  tail -20 runlogs/trades_r34_chunk*.log"
echo "  tail -20 runlogs/trades_b4_chunk*.log"
echo "  tail -20 runlogs/trades_b8_chunk*.log"
echo "============================================================"
