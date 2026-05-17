#!/usr/bin/env bash
set -euo pipefail

# Staged SQED 4pt seed evaluation.
#
# The script runs deterministic beam stages only on rows that failed the
# previous stage, then optionally tries stochastic nucleus rescue attempts.
#
# Configurable environment variables:
#   MODEL_PATH              default: models/unit_500k/best_model.pt
#   DEVICE                  default: auto
#   DATASETS                default: "1 2 3 4"
#   SAMPLES                 default: 100
#   BEAM_STAGES             default: "5 50 200"
#   MAX_BEAM                default: 250
#   BATCH_SIZE              default: 8
#   PLOTS                   default: 0
#   HUMAN_CSV               default: 1
#   NUCLEUS_ATTEMPTS        default: 0
#   NUCLEUS_BEAM_SIZE       default: last value in BEAM_STAGES
#   NUCLEUS_P               default: 0.99
#   NUCLEUS_TEMPERATURES    default: "0.8 1.0 1.2 1.5"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-models/unit_500k/best_model.pt}"
DEVICE="${DEVICE:-auto}"
DATASETS="${DATASETS:-1 2 3 4}"
SAMPLES="${SAMPLES:-100}"
BEAM_STAGES="${BEAM_STAGES:-5 50 200}"
MAX_BEAM="${MAX_BEAM:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PLOTS="${PLOTS:-0}"
HUMAN_CSV="${HUMAN_CSV:-1}"
NUCLEUS_ATTEMPTS="${NUCLEUS_ATTEMPTS:-20}"
NUCLEUS_P="${NUCLEUS_P:-0.99}"
NUCLEUS_TEMPERATURES="${NUCLEUS_TEMPERATURES:-0.8 1.0 1.2 1.4}"

OUTPUT_DIR="data_testing/outputs/staged_sqed_4ptseed"
STAGE_REPORT="${OUTPUT_DIR}/stage_report.csv"
FINAL_REPORT="${OUTPUT_DIR}/final_report.csv"
TMP_DIR="$(mktemp -d /tmp/sqed_4ptseed_staged.XXXXXX)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model checkpoint not found: $MODEL_PATH" >&2
  exit 1
fi

read -r -a DATASET_IDS <<< "$DATASETS"
read -r -a BEAMS <<< "$BEAM_STAGES"
read -r -a NUCLEUS_TEMPS <<< "$NUCLEUS_TEMPERATURES"

if [[ "${#BEAMS[@]}" -eq 0 ]]; then
  echo "ERROR: BEAM_STAGES must contain at least one beam size" >&2
  exit 1
fi
if ! [[ "$MAX_BEAM" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_BEAM must be an integer, got: $MAX_BEAM" >&2
  exit 1
fi
if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SAMPLES must be an integer for staged evaluation, got: $SAMPLES" >&2
  exit 1
fi
if ! [[ "$NUCLEUS_ATTEMPTS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NUCLEUS_ATTEMPTS must be an integer, got: $NUCLEUS_ATTEMPTS" >&2
  exit 1
fi
if [[ "$NUCLEUS_ATTEMPTS" -gt 0 && "${#NUCLEUS_TEMPS[@]}" -eq 0 ]]; then
  echo "ERROR: NUCLEUS_TEMPERATURES must contain at least one value when NUCLEUS_ATTEMPTS > 0" >&2
  exit 1
fi

LAST_BEAM="${BEAMS[$((${#BEAMS[@]} - 1))]}"
NUCLEUS_BEAM_SIZE="${NUCLEUS_BEAM_SIZE:-$LAST_BEAM}"

check_beam() {
  local value="$1"
  local label="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $label must be an integer, got: $value" >&2
    exit 1
  fi
  if (( value > MAX_BEAM )); then
    echo "ERROR: $label=$value exceeds MAX_BEAM=$MAX_BEAM" >&2
    exit 1
  fi
}

for beam in "${BEAMS[@]}"; do
  check_beam "$beam" "beam stage"
done
check_beam "$NUCLEUS_BEAM_SIZE" "NUCLEUS_BEAM_SIZE"

plot_flag="--no-plots"
if [[ "$PLOTS" == "1" ]]; then
  plot_flag="--plots"
fi

human_flag="--human-csv"
if [[ "$HUMAN_CSV" != "1" ]]; then
  human_flag="--no-human-csv"
fi

printf 'dataset,stage,method,beam_size,temperature,input_examples,stage_successes,cumulative_successes,remaining_failures,cumulative_accuracy_pct\n' > "$STAGE_REPORT"
printf 'dataset,total_examples,beam_stages,beam_successes,nucleus_attempts,nucleus_successes,cumulative_successes,remaining_failures,final_cumulative_accuracy_pct\n' > "$FINAL_REPORT"

read_summary() {
  python3 - "$1" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    row = next(csv.DictReader(handle))

print(row["total_examples"], row["top1_num_eq_scrambled"])
PY
}

write_failures_csv() {
  python3 - "$1" "$2" <<'PY'
import csv
import sys

detail_path, out_path = sys.argv[1], sys.argv[2]
count = 0
with open(detail_path, newline="", encoding="utf-8") as src, open(
    out_path, "w", newline="", encoding="utf-8"
) as dst:
    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=["simple", "scrambled"])
    writer.writeheader()
    for row in reader:
        if int(row["top1_num_eq_scrambled"]) == 0:
            writer.writerow(
                {
                    "simple": row["target_simple"],
                    "scrambled": row["input_scrambled"],
                }
            )
            count += 1
print(count)
PY
}

effective_csv_rows() {
  local csv_path="$1"
  local rows
  rows="$(($(wc -l < "$csv_path") - 1))"
  if (( rows > SAMPLES )); then
    echo "$SAMPLES"
  else
    echo "$rows"
  fi
}

run_stage() {
  local method="$1"
  local beam_size="$2"
  local temperature="$3"
  local input_csv="$4"
  local output_stem="$5"
  local args=(
    python3 data_testing/evaluate_model.py
    --model-path "$MODEL_PATH"
    --device "$DEVICE"
    --data-source csv
    --existing-raw-csv "$input_csv"
    --existing-csv-max-rows "$SAMPLES"
    --no-dedupe
    --decoding-method "$method"
    --beam-size "$beam_size"
    --rerank-numerical
    --batch-size "$BATCH_SIZE"
    --output-stem "$output_stem"
    --simple-summary
    "$human_flag"
    "$plot_flag"
  )

  if [[ "$method" == "nucleus" ]]; then
    args+=(--p-nucleus "$NUCLEUS_P" --temperature-nucleus "$temperature")
  fi

  "${args[@]}"
}

for dataset in "${DATASET_IDS[@]}"; do
  source_gz="data/sqed/sqed_4ptseed_oneshot_${dataset}.csv.gz"
  if [[ ! -f "$source_gz" ]]; then
    echo "ERROR: dataset not found: $source_gz" >&2
    exit 1
  fi

  dataset_tmp="${TMP_DIR}/dataset_${dataset}"
  mkdir -p "$dataset_tmp"
  current_csv="${dataset_tmp}/stage_0_input.csv"
  gzip -cd "$source_gz" > "$current_csv"

  total_examples=""
  cumulative_successes=0
  remaining_failures=0
  beam_successes=()
  nucleus_successes=()

  echo "=== Dataset ${dataset}: staged beam search ==="

  for beam in "${BEAMS[@]}"; do
    input_count="$(effective_csv_rows "$current_csv")"
    if (( input_count <= 0 )); then
      echo "No failures remain before beam=$beam; skipping remaining beam stages."
      break
    fi

    output_stem="staged_sqed_4ptseed/dataset_${dataset}_beam_${beam}"
    summary_path="data_testing/outputs/${output_stem}_summary.csv"
    detail_path="data_testing/outputs/${output_stem}_beam_results.csv"
    next_csv="${dataset_tmp}/after_beam_${beam}_failures.csv"

    echo "--- Dataset ${dataset} | beam=${beam} | input rows=${input_count} ---"
    run_stage "beam" "$beam" "" "$current_csv" "$output_stem"

    read -r stage_total stage_successes < <(read_summary "$summary_path")
    if [[ -z "$total_examples" ]]; then
      total_examples="$stage_total"
    fi
    cumulative_successes=$((cumulative_successes + stage_successes))
    remaining_failures="$(write_failures_csv "$detail_path" "$next_csv")"
    beam_successes+=("${beam}:${stage_successes}")

    accuracy="$(python3 - "$cumulative_successes" "$total_examples" <<'PY'
import sys
successes = int(sys.argv[1])
total = int(sys.argv[2])
print(f"{100 * successes / total:.6f}" if total else "0.000000")
PY
)"
    printf '%s,beam_%s,beam,%s,,%s,%s,%s,%s,%s\n' \
      "$dataset" "$beam" "$beam" "$stage_total" "$stage_successes" \
      "$cumulative_successes" "$remaining_failures" "$accuracy" >> "$STAGE_REPORT"

    current_csv="$next_csv"
  done

  for ((attempt = 1; attempt <= NUCLEUS_ATTEMPTS; attempt++)); do
    input_count="$(effective_csv_rows "$current_csv")"
    if (( input_count <= 0 )); then
      echo "No failures remain before nucleus attempt ${attempt}; skipping remaining attempts."
      break
    fi

    temp_index=$(((attempt - 1) % ${#NUCLEUS_TEMPS[@]}))
    temperature="${NUCLEUS_TEMPS[$temp_index]}"
    temp_label="${temperature//./p}"
    output_stem="staged_sqed_4ptseed/dataset_${dataset}_nucleus_${attempt}_t${temp_label}"
    summary_path="data_testing/outputs/${output_stem}_summary.csv"
    detail_path="data_testing/outputs/${output_stem}_nucleus_results.csv"
    next_csv="${dataset_tmp}/after_nucleus_${attempt}_failures.csv"

    echo "--- Dataset ${dataset} | nucleus attempt=${attempt} | temp=${temperature} | input rows=${input_count} ---"
    run_stage "nucleus" "$NUCLEUS_BEAM_SIZE" "$temperature" "$current_csv" "$output_stem"

    read -r stage_total stage_successes < <(read_summary "$summary_path")
    if [[ -z "$total_examples" ]]; then
      total_examples="$stage_total"
    fi
    cumulative_successes=$((cumulative_successes + stage_successes))
    remaining_failures="$(write_failures_csv "$detail_path" "$next_csv")"
    nucleus_successes+=("${attempt}@${temperature}:${stage_successes}")

    accuracy="$(python3 - "$cumulative_successes" "$total_examples" <<'PY'
import sys
successes = int(sys.argv[1])
total = int(sys.argv[2])
print(f"{100 * successes / total:.6f}" if total else "0.000000")
PY
)"
    printf '%s,nucleus_%s,nucleus,%s,%s,%s,%s,%s,%s,%s\n' \
      "$dataset" "$attempt" "$NUCLEUS_BEAM_SIZE" "$temperature" "$stage_total" \
      "$stage_successes" "$cumulative_successes" "$remaining_failures" "$accuracy" >> "$STAGE_REPORT"

    current_csv="$next_csv"
  done

  if [[ -z "$total_examples" ]]; then
    total_examples=0
  fi
  final_accuracy="$(python3 - "$cumulative_successes" "$total_examples" <<'PY'
import sys
successes = int(sys.argv[1])
total = int(sys.argv[2])
print(f"{100 * successes / total:.6f}" if total else "0.000000")
PY
)"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$dataset" \
    "$total_examples" \
    "\"${BEAMS[*]}\"" \
    "\"${beam_successes[*]:-}\"" \
    "$NUCLEUS_ATTEMPTS" \
    "\"${nucleus_successes[*]:-}\"" \
    "$cumulative_successes" \
    "$remaining_failures" \
    "$final_accuracy" >> "$FINAL_REPORT"

  echo "Dataset ${dataset}: ${cumulative_successes}/${total_examples} correct (${final_accuracy}%)."
  echo ""
done

echo "Wrote stage report to ${STAGE_REPORT}"
echo "Wrote final report to ${FINAL_REPORT}"
