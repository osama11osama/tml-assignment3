#!/usr/bin/env bash
# Download Assignment 3 train.npz on the cluster (run once).
set -eu

BASE="${TML3_BASE:-$HOME/tml26_task3}"
mkdir -p "$BASE/data"
cd "$BASE"

URL="https://huggingface.co/datasets/SprintML/tml26_task3/resolve/main/train.npz"
OUT="data/train.npz"

if [[ -f "$OUT" ]]; then
  echo "Already present: $OUT ($(du -h "$OUT" | cut -f1))"
  exit 0
fi

echo "Downloading train.npz ..."
wget -q --show-progress -O "$OUT" "$URL"
echo "Done: $OUT ($(du -h "$OUT" | cut -f1))"
