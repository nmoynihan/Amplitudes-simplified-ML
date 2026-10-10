#!/bin/bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TASK_DIR/evaluate.sh" \
  --existing-raw-csv "$TASK_DIR/sqed_4pt_physical_prepared.csv" \
  --output-stem sqed_4pt_physical_prepared_eval \
  "$@"
