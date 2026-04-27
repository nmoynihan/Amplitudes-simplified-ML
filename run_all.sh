#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train_eval}"

for script in run1.sh run2.sh run3.sh run4.sh run5.sh run6.sh run7.sh; do
  echo "==== Running $script $MODE ===="
  "./$script" "$MODE"
done
