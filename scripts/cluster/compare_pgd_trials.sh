#!/usr/bin/env bash
# Compare parallel PGD trials after epoch-20 jobs finish.
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(command -v python3 || command -v python)"
fi

read_progress() {
  local tag="$1"
  local f="results/runs/${tag}_resnet18/progress.json"
  if [[ ! -f "$f" ]]; then
    echo "MISSING"
    return
  fi
  "$PYTHON" - "$f" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
last = p.get("last_epoch") or p.get("epochs_completed") or 0
best = p.get("best_unified_score", p.get("final_unified_score", 0))
clean = p.get("last_clean_acc", p.get("final_clean_acc", 0))
robust = p.get("last_robust_acc", p.get("final_robust_acc", 0))
print(f"{last}|{best}|{clean}|{robust}")
PY
}

echo "=== PGD trial comparison (epoch ~20) ==="
echo ""

for TAG in pgd_at pgd_at_s10; do
  P=$(read_progress "$TAG")
  if [[ "$P" == "MISSING" ]]; then
    echo "$TAG: not started / no progress.json"
    continue
  fi
  IFS='|' read -r LAST BEST CLEAN ROBUST <<< "$P"
  echo "$TAG:"
  echo "  last_epoch=$LAST  best_unified=$BEST  clean=$CLEAN  robust=$ROBUST"
  echo ""
done

P_MAIN=$(read_progress pgd_at)
P_S10=$(read_progress pgd_at_s10)
if [[ "$P_MAIN" == "MISSING" || "$P_S10" == "MISSING" ]]; then
  echo "Wait until both jobs finish, then re-run."
  exit 0
fi

IFS='|' read -r _ BEST_MAIN _ _ <<< "$P_MAIN"
IFS='|' read -r _ BEST_S10 _ _ <<< "$P_S10"

WINNER="pgd_at"
if awk "BEGIN {exit !($BEST_S10 > $BEST_MAIN)}"; then
  WINNER="pgd_at_s10"
fi

echo "Recommended winner: $WINNER (best_unified)"
echo ""
if [[ "$WINNER" == "pgd_at" ]]; then
  echo "Continue: bash scripts/cluster/submit_pgd_next.sh"
else
  echo "Continue: TML3_TAG=pgd_at_s10 bash scripts/cluster/submit_pgd_next.sh"
fi
