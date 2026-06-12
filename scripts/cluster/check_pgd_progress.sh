#!/usr/bin/env bash
# Show PGD-AT progress and the next 1-hour chunk to submit.
# Usage: bash check_pgd_progress.sh [TAG]   (default tag: pgd_at)
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v python3 || command -v python)"
fi

TAG="${1:-${TML3_TAG:-pgd_at}}"
ARCH="${TML3_ARCH:-resnet18}"
FINAL="${TML3_FINAL_EPOCHS:-${TML3_EPOCHS:-80}}"
CHUNKS=(20 40 60 80 100 120)
PROGRESS="results/runs/${TAG}_${ARCH}/progress.json"
CKPT="results/checkpoints/${TAG}_${ARCH}.pt"

echo "=== PGD-AT progress (tag=$TAG) ==="

if [[ ! -f "$PROGRESS" ]]; then
  echo "Status: not started"
  echo "Next chunk: epoch 20 (first job)"
  echo ""
  echo "  condor_submit scripts/cluster/condor/train_pgd_at_chunk20.sub"
  if [[ "$TAG" != "pgd_at" ]]; then
    echo "  (or variant sub for tag $TAG)"
  fi
  exit 0
fi

STATUS=$(grep -o '"status": "[^"]*"' "$PROGRESS" | tail -1 | cut -d'"' -f4 || echo unknown)
read -r LAST BEST <<< "$("$PYTHON" - "$PROGRESS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
last = p.get("last_epoch") or p.get("epochs_completed") or 0
best = p.get("best_unified_score")
if best is None:
    best = p.get("final_unified_score", "?")
print(last, best if best is not None else "?")
PY
)"

echo "Status: $STATUS | last_epoch: $LAST / $FINAL | best_unified: $BEST"
echo "Checkpoint: $CKPT"

if [[ "$STATUS" == "finished" ]] && [[ "$LAST" -ge "$FINAL" ]]; then
  echo ""
  echo "DONE — download ${TAG}_${ARCH}.pt and run eval_model.py locally."
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
echo ""
if [[ "$TAG" == "pgd_at" ]]; then
  echo "  condor_submit scripts/cluster/condor/train_pgd_at_chunk${NEXT}.sub"
  if [[ "$FINAL" -gt 80 ]]; then
    echo "  bash scripts/cluster/submit_pgd_extend_next.sh"
  else
    echo "  bash scripts/cluster/submit_pgd_next.sh"
  fi
else
  echo "  TML3_TAG=$TAG bash scripts/cluster/submit_pgd_next.sh"
fi
echo "  condor_q"
echo "  tail -f runlogs/${TAG}_chunk${NEXT}_*.log"
