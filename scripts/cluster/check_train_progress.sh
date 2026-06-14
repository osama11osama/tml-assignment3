#!/usr/bin/env bash
# Generic training progress + next chunk hint (ERM / TRADES / PGD).
# Usage: TML3_TAG=erm_r34 TML3_FINAL_EPOCHS=100 TML3_CHUNKS=20,40,60,80,100 bash check_train_progress.sh
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v python3 || command -v python)"
fi

TAG="${1:-${TML3_TAG:-pgd_at}}"
ARCH="${TML3_ARCH:-resnet18}"
FINAL="${TML3_FINAL_EPOCHS:-80}"
IFS=',' read -ra CHUNKS <<< "${TML3_CHUNKS:-20,40,60,80,100,120}"
PROGRESS="results/runs/${TAG}_${ARCH}/progress.json"
CKPT_GUESS="results/checkpoints/${TAG}_${ARCH}.pt"
if [[ "$TAG" == erm* ]]; then
  CKPT_GUESS="results/checkpoints/baseline_${TAG}_${ARCH}.pt"
fi

echo "=== Training progress (tag=$TAG arch=$ARCH) ==="

if [[ ! -f "$PROGRESS" ]]; then
  echo "Status: not started"
  echo "Next chunk target: epoch ${CHUNKS[0]}"
  exit 0
fi

read -r STATUS LAST BEST <<< "$("$PYTHON" - "$PROGRESS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
last = p.get("last_epoch") or p.get("epochs_completed") or 0
best = p.get("best_unified_score", p.get("final_unified_score", "?"))
status = p.get("status", "unknown")
print(status, last, best if best is not None else "?")
PY
)"

echo "Status: $STATUS | last_epoch: $LAST / $FINAL | best_unified: $BEST"
echo "Checkpoint: $CKPT_GUESS"

if [[ "$STATUS" == "finished" ]] && [[ "$LAST" -ge "$FINAL" ]]; then
  echo ""
  echo "DONE — download checkpoint and run eval_model.py locally."
  exit 0
fi

NEXT=""
for T in "${CHUNKS[@]}"; do
  if [[ "$T" -gt "$FINAL" ]]; then
    continue
  fi
  if [[ "$LAST" -lt "$T" ]]; then
    NEXT="$T"
    break
  fi
done

if [[ -z "$NEXT" ]] && [[ "$LAST" -lt "$FINAL" ]]; then
  NEXT="$FINAL"
fi

echo ""
echo "Next 1-hour job target: epoch $NEXT"
echo "  condor_submit scripts/cluster/condor/train_${TAG}_chunk${NEXT}.sub"
echo "  (or generic run_*_chunk.sh $NEXT with TML3_TAG=$TAG)"
