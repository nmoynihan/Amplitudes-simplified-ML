#!/usr/bin/env bash
set -euo pipefail

# Evaluate the four SQED 4pt seed one-shot datasets and write one comparison CSV.
#
# Configurable environment variables:
#   MODEL_PATH       default: models/unit_500k/best_model.pt
#   DEVICE           default: auto
#   DECODING_METHOD  default: beam
#   BEAM_SIZE        default: 50
#   BATCH_SIZE       default: 8
#   PLOTS            default: 0
#   SAMPLES          default: 100
#   DATASETS         default: "1 2 3 4"
#   SUMMARY_PLOTS    default: 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-models/unit_500k/best_model.pt}"
DEVICE="${DEVICE:-auto}"
DECODING_METHOD="${DECODING_METHOD:-beam}"
BEAM_SIZE="${BEAM_SIZE:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PLOTS="${PLOTS:-0}"
SAMPLES="${SAMPLES:-100}"
DATASETS="${DATASETS:-1 2 3 4}"
#DATASETS="${DATASETS:-4}"
SUMMARY_PLOTS="${SUMMARY_PLOTS:-1}"

OUTPUT_DIR="data_testing/outputs"
REPORT_PATH="${OUTPUT_DIR}/sqed_4ptseed_oneshot_comparison.csv"
TMP_DIR="$(mktemp -d /tmp/sqed_4ptseed_eval.XXXXXX)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"

plot_flag="--no-plots"
if [[ "$PLOTS" == "1" ]]; then
  plot_flag="--plots"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model checkpoint not found: $MODEL_PATH" >&2
  exit 1
fi

summary_paths=()

read -r -a DATASET_IDS <<< "$DATASETS"

for n in "${DATASET_IDS[@]}"; do
  gz_path="data/sqed/sqed_4ptseed_oneshot_${n}.csv.gz"
  csv_path="${TMP_DIR}/sqed_4ptseed_oneshot_${n}.csv"
  output_stem="sqed_4ptseed_oneshot_${n}"
  summary_path="${OUTPUT_DIR}/${output_stem}_summary.csv"

  if [[ ! -f "$gz_path" ]]; then
    echo "ERROR: dataset not found: $gz_path" >&2
    exit 1
  fi

  echo "=== Dataset ${n}: ${gz_path} ==="
  gzip -cd "$gz_path" > "$csv_path"

  python3 data_testing/evaluate_model.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --data-source csv \
    --existing-raw-csv "$csv_path" \
    --existing-csv-max-rows "$SAMPLES" \
    --decoding-method "$DECODING_METHOD" \
    --beam-size "$BEAM_SIZE" \
    --rerank-numerical \
    --batch-size "$BATCH_SIZE" \
    --output-stem "$output_stem" \
    --simple-summary \
    "$plot_flag"

  if [[ ! -f "$summary_path" ]]; then
    echo "ERROR: expected summary was not written: $summary_path" >&2
    exit 1
  fi

  summary_paths+=("${n}:${summary_path}")
  echo ""
done

python3 - "$REPORT_PATH" "${summary_paths[@]}" <<'PY'
import csv
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
summary_specs = sys.argv[2:]

rows = []
fieldnames = None
for spec in summary_specs:
    dataset, path_str = spec.split(":", 1)
    path = Path(path_str)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no CSV header")
        if fieldnames is None:
            fieldnames = ["dataset", *reader.fieldnames]
        for row in reader:
            rows.append({"dataset": dataset, **row})

if fieldnames is None:
    raise SystemExit("No summary rows found")

with report_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote comparison report to {report_path}")
PY

if [[ "$SUMMARY_PLOTS" == "1" ]]; then
  python3 data_testing/plot_sqed_4ptseed.py --outputs-dir "$OUTPUT_DIR"
fi

echo "All SQED 4pt seed evaluations complete."
