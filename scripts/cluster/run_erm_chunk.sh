#!/usr/bin/env bash
# ERM training in 1-hour Condor chunks (resume toward TARGET epoch).
#
# Usage:
#   bash scripts/cluster/run_erm_chunk.sh 20
#   TML3_TAG=erm_r34 TML3_ARCH=resnet34 bash scripts/cluster/run_erm_chunk.sh 40
#
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"
mkdir -p runlogs results/checkpoints results/runs

TARGET="${1:-20}"
if ! [[ "$TARGET" =~ ^[0-9]+$ ]] || [[ "$TARGET" -lt 1 ]]; then
  echo "Usage: $0 TARGET_EPOCH   (e.g. 20, 40, 60, 80, 100)"
  exit 2
fi

TAG="${TML3_TAG:-erm_r34}"
ARCH="${TML3_ARCH:-resnet34}"
RUN_DIR="results/runs/${TAG}_${ARCH}"
mkdir -p "$RUN_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"

BATCH="${TML3_BATCH:-256}"
NUM_WORKERS="${TML3_NUM_WORKERS:-4}"
SAVE_EVERY="${TML3_SAVE_EVERY:-5}"
LAST_PT="${RUN_DIR}/last.pt"
PROGRESS="${RUN_DIR}/progress.json"

RESUME=""
if [[ -f "$LAST_PT" ]]; then
  RESUME="--resume"
  echo "Found $LAST_PT — resuming toward epoch $TARGET"
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

echo "ERM chunk start $(date -Iseconds) | tag=$TAG arch=$ARCH target epoch $TARGET" | tee "$LOG"
echo "batch=$BATCH workers=$NUM_WORKERS save_every=$SAVE_EVERY" | tee -a "$LOG"

set +e
timeout "$TIMEOUT_SEC" "$PYTHON" scripts/train_standard.py \
  --device cuda \
  --architecture "$ARCH" \
  --tag "$TAG" \
  --epochs "$TARGET" \
  --batch-size "$BATCH" \
  --save-every "$SAVE_EVERY" \
  --robust-every 0 \
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
TML3_TAG="$TAG" TML3_ARCH="$ARCH" TML3_FINAL_EPOCHS="${TML3_FINAL_EPOCHS:-100}" \
  bash scripts/cluster/check_train_progress.sh | tee -a "$LOG"
