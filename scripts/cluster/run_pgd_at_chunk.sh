#!/usr/bin/env bash
# PGD-AT in 1-hour Condor chunks (same full training, split with --resume).
#
# GPU tuning (cluster defaults):
#   TML3_BATCH=256          — saturate GPU (was 128)
#   TML3_NUM_WORKERS=4      — parallel data loading
#   TML3_ROBUST_EVERY=0     — PGD-20 eval only at chunk target (saves ~30% wall time)
#   TML3_TAG=pgd_at         — run dir tag (use different tags for parallel trials)
#
# Usage (on cluster):
#   bash scripts/cluster/run_pgd_at_chunk.sh 20
#   TML3_TAG=pgd_at_s10 TML3_PGD_TRAIN_STEPS=10 bash scripts/cluster/run_pgd_at_chunk.sh 20
#   bash scripts/cluster/submit_pgd_next.sh
#
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"
mkdir -p runlogs results/checkpoints

TARGET="${1:-20}"
if ! [[ "$TARGET" =~ ^[0-9]+$ ]] || [[ "$TARGET" -lt 1 ]]; then
  echo "Usage: $0 TARGET_EPOCH   (e.g. 20, 40, 60, 80)"
  exit 2
fi

TAG="${TML3_TAG:-pgd_at}"
ARCH="${TML3_ARCH:-resnet18}"
RUN_DIR="results/runs/${TAG}_${ARCH}"
mkdir -p "$RUN_DIR"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"

INIT="${TML3_INIT:-results/checkpoints/fgsm_at_resnet18.pt}"
BATCH="${TML3_BATCH:-256}"
NUM_WORKERS="${TML3_NUM_WORKERS:-4}"
PGD_TRAIN_STEPS="${TML3_PGD_TRAIN_STEPS:-7}"
ROBUST_EVERY="${TML3_ROBUST_EVERY:-0}"
SAVE_EVERY="${TML3_SAVE_EVERY:-3}"
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
    echo "WARN: no last.pt and no init checkpoint"
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

echo "PGD-AT chunk start $(date -Iseconds) | tag=$TAG target epoch $TARGET" | tee "$LOG"
echo "Python: $PYTHON ($("$PYTHON" --version 2>&1))" | tee -a "$LOG"
echo "batch=$BATCH workers=$NUM_WORKERS pgd_train=$PGD_TRAIN_STEPS robust_every=$ROBUST_EVERY save_every=$SAVE_EVERY" | tee -a "$LOG"
echo "resume=${RESUME:-no}" | tee -a "$LOG"

TIMEOUT_SEC="${TML3_TIMEOUT_SEC:-3300}"

set +e
timeout "$TIMEOUT_SEC" "$PYTHON" scripts/train_pgd_at.py \
  --device cuda \
  --architecture "$ARCH" \
  --tag "$TAG" \
  --epochs "$TARGET" \
  --batch-size "$BATCH" \
  --pgd-train-steps "$PGD_TRAIN_STEPS" \
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
TML3_TAG="$TAG" bash scripts/cluster/check_pgd_progress.sh | tee -a "$LOG"
