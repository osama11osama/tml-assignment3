#!/usr/bin/env bash
# TRADES training in 1-hour Condor chunks (resume toward TARGET epoch).
#
# Usage:
#   TML3_INIT=results/checkpoints/pgd_at_resnet18.pt bash scripts/cluster/run_trades_chunk.sh 20
#   TML3_TAG=trades_r18 bash scripts/cluster/run_trades_chunk.sh 40
#
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"
mkdir -p runlogs results/checkpoints results/runs

TARGET="${1:-20}"
if ! [[ "$TARGET" =~ ^[0-9]+$ ]] || [[ "$TARGET" -lt 1 ]]; then
  echo "Usage: $0 TARGET_EPOCH   (e.g. 20, 40, 80)"
  exit 2
fi

TAG="${TML3_TAG:-trades_r18}"
ARCH="${TML3_ARCH:-resnet18}"
RUN_DIR="results/runs/${TAG}_${ARCH}"
mkdir -p "$RUN_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"

INIT="${TML3_INIT:-results/checkpoints/pgd_at_resnet18.pt}"
BATCH="${TML3_BATCH:-128}"
NUM_WORKERS="${TML3_NUM_WORKERS:-4}"
PGD_TRAIN_STEPS="${TML3_PGD_TRAIN_STEPS:-10}"
TRADES_BETA="${TML3_TRADES_BETA:-6.0}"
ROBUST_EVERY="${TML3_ROBUST_EVERY:-0}"
SAVE_EVERY="${TML3_SAVE_EVERY:-5}"
LR="${TML3_LR:-0.005}"
LAST_PT="${RUN_DIR}/last.pt"
PROGRESS="${RUN_DIR}/progress.json"

RESUME=""
INIT_ARG=()
if [[ -f "$LAST_PT" ]]; then
  RESUME="--resume"
  echo "Found $LAST_PT — resuming toward epoch $TARGET"
else
  if [[ -f "$INIT" ]]; then
    INIT_ARG=(--init "$INIT")
    echo "Starting from init $INIT toward epoch $TARGET"
  else
    echo "WARN: no last.pt and no init checkpoint ($INIT)"
  fi
fi

if [[ -f "$PROGRESS" ]]; then
  DONE=$("$PYTHON" - "$PROGRESS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(p.get("last_epoch") or p.get("epochs_completed") or 0)
PY
)
  if [[ -n "$DONE" && "$DONE" -ge "$TARGET" ]]; then
    echo "Already at epoch $DONE >= target $TARGET. Nothing to do."
    exit 0
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="runlogs/${TAG}_chunk${TARGET}_${STAMP}.log"
TIMEOUT_SEC="${TML3_TIMEOUT_SEC:-3300}"

echo "TRADES chunk start $(date -Iseconds) | tag=$TAG target epoch $TARGET" | tee "$LOG"
echo "init=${INIT_ARG[*]:-resume} beta=$TRADES_BETA pgd_train=$PGD_TRAIN_STEPS lr=$LR batch=$BATCH" | tee -a "$LOG"

set +e
timeout "$TIMEOUT_SEC" "$PYTHON" scripts/train_trades.py \
  --device cuda \
  --architecture "$ARCH" \
  --tag "$TAG" \
  --epochs "$TARGET" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --pgd-train-steps "$PGD_TRAIN_STEPS" \
  --trades-beta "$TRADES_BETA" \
  --robust-every "$ROBUST_EVERY" \
  --save-every "$SAVE_EVERY" \
  --num-workers "$NUM_WORKERS" \
  "${INIT_ARG[@]}" \
  $RESUME \
  2>&1 | tee -a "$LOG"
EXIT=$?
set -e

if [[ "$EXIT" -eq 124 ]]; then
  echo "Stopped at soft timeout (${TIMEOUT_SEC}s). Resume with next chunk." | tee -a "$LOG"
elif [[ "$EXIT" -ne 0 ]]; then
  echo "Training exited with code $EXIT" | tee -a "$LOG"
  exit "$EXIT"
fi

echo "Chunk complete (target was epoch $TARGET)." | tee -a "$LOG"
TML3_TAG="$TAG" TML3_ARCH="$ARCH" TML3_FINAL_EPOCHS="${TML3_FINAL_EPOCHS:-40}" \
  bash scripts/cluster/check_train_progress.sh | tee -a "$LOG"
