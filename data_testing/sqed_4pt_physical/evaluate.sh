#!/bin/bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /Users/kymani/miniforge3/envs/scattering/bin/python \
  "$TASK_DIR/../evaluate_model.py" \
  --model-path /Users/kymani/Documents/SQED_KYMANI/SQED/4PT_Model/best_model.pt \
  --device cpu \
  --data-source csv \
  --existing-raw-csv "$TASK_DIR/sqed_4pt_physical.csv" \
  --n-particles 4 \
  --numeric-backend sqed \
  --numeric-pol-modes coulomb covariant \
  --numeric-samples 5 \
  --tokenizer-max-particles 8 \
  --existing-csv-max-rows 1 \
  --decoding-method greedy \
  --max-decode-tokens 256 \
  --batch-size 1 \
  --no-plots \
  --output-stem sqed_4pt_physical_eval \
  "$@"
