#!/usr/bin/env bash
# Run FGSM-AT (Step 2) on a GPU node or inside Condor Docker job.
#
# Usage:
#   bash scripts/cluster/run_fgsm_at.sh
#   bash scripts/cluster/run_fgsm_at.sh --resume
#   TML3_INIT=results/checkpoints/baseline_erm_resnet18.pt bash scripts/cluster/run_fgsm_at.sh
#
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"
mkdir -p runlogs results/runs results/checkpoints

PYTHON="${PYTHON:-/opt/conda/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v python3 || command -v python)"
fi

export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"

RESUME=""
INIT="${TML3_INIT:-results/checkpoints/baseline_erm_resnet18.pt}"
ARCH="${TML3_ARCH:-resnet18}"
EPOCHS="${TML3_EPOCHS:-50}"
BATCH="${TML3_BATCH:-128}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME="--resume"; shift ;;
    --init) INIT="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="runlogs/fgsm_at_${STAMP}.log"

echo "FGSM-AT start $(date -Iseconds)" | tee "$LOG"
echo "Python: $PYTHON ($("$PYTHON" --version 2>&1))" | tee -a "$LOG"
echo "Base: $BASE | arch=$ARCH epochs=$EPOCHS init=$INIT resume=${RESUME:-no}" | tee -a "$LOG"

INIT_ARG=()
if [[ -n "$INIT" && -f "$INIT" && -z "$RESUME" ]]; then
  INIT_ARG=(--init "$INIT")
elif [[ -n "$INIT" && ! -f "$INIT" && -z "$RESUME" ]]; then
  echo "WARN: init checkpoint missing ($INIT) - training from random init" | tee -a "$LOG"
fi

"$PYTHON" scripts/train_fgsm_at.py \
  --device cuda \
  --architecture "$ARCH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --save-every 5 \
  --num-workers 2 \
  "${INIT_ARG[@]}" \
  $RESUME \
  2>&1 | tee -a "$LOG"
