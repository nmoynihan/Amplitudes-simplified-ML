#!/bin/bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON=/Users/kymani/miniforge3/envs/scattering/bin/python
"$TASK_PYTHON" "$TASK_DIR/prepare.py"
bash "$TASK_DIR/../evaluate.sh" \
  --existing-raw-csv "$TASK_DIR/sqed_5pt_prepared.csv" \
  --decoding-method beam --beam-size 20 --rerank-numerical \
  --device cpu --max-decode-tokens 256 \
  --output-stem sqed_5pt_prepared_beam20
"$TASK_PYTHON" "$TASK_DIR/verify_result.py"
