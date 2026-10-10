#!/bin/bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /Users/kymani/miniforge3/envs/scattering/bin/python \
  "$TASK_DIR/../evaluate_model.py" \
  --model-path /Users/kymani/Documents/SQED_KYMANI/SQED/5PT_Model/best_model.pt \
  --data-source csv \
  --existing-raw-csv "$TASK_DIR/sqed_5pt_physical.csv" \
  --n-particles 5 \
  --numeric-backend sqed \
  --numeric-pol-modes coulomb covariant \
  --numeric-samples 5 \
  --tokenizer-max-particles 8 \
  --existing-csv-max-rows 1 \
  --decoding-method greedy \
  --max-decode-tokens 512 \
  --batch-size 1 \
  --no-plots \
  --output-stem sqed_5pt_physical_eval \
  "$@"
