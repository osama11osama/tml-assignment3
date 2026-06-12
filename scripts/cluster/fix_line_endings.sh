#!/usr/bin/env bash
# Fix Windows CRLF in cluster scripts (run once on cluster after scp from Windows).
set -eu
BASE="${TML3_BASE:-$HOME/tml26_task3}"
cd "$BASE"
find scripts/cluster -type f \( -name '*.sh' -o -name '*.sub' \) -exec sed -i 's/\r$//' {} +
chmod +x scripts/cluster/*.sh
echo "Line endings fixed under scripts/cluster/"
