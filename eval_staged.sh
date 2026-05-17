#!/usr/bin/env bash
set -euo pipefail

# Generic staged one-shot evaluator.
#
# Use either:
#   INPUT_CSV=data/my_eval.csv ./eval_staged.sh
# or:
#   DATA_SOURCE=generate SAMPLES=100 ./eval_staged.sh
#
# Configurable environment variables:
#   MODEL_PATH              default: models/unit_500k/best_model.pt
#   DEVICE                  default: auto
#   DATA_SOURCE             default: csv; set to generate to ignore INPUT_CSV
#   INPUT_CSV               optional CSV or CSV.GZ with simple,scrambled columns
#   RUN_ID                  default: staged_eval
#   OUTPUT_DIR              default: data_testing/outputs/staged_eval
#   SAMPLES                 default: 100
#   BEAM_STAGES             default: "5 50 200"
#   MAX_BEAM                default: 250
#   BATCH_SIZE              default: 8
#   HUMAN_CSV               default: 1
#   PLOTS                   default: 0
#   NUCLEUS_ATTEMPTS        default: 0
#   NUCLEUS_BEAM_SIZE       default: last value in BEAM_STAGES
#   NUCLEUS_P               default: 0.99
#   NUCLEUS_TEMPERATURES    default: "0.8 1.0 1.2 1.5"
#
# Generation-only knobs:
#   N_PARTICLES             default: 4
#   SEED                    default: 123
#   MIN_SCR                 default: 1
#   MAX_SCR                 default: 4
#   MIN_TERMS               default: 1
#   MAX_TERMS               default: 3
#   MAX_TOKENS              optional; default lets evaluator use dataset max length
#   TOKENIZER_MAX_PARTICLES default: 8
#   SCRAMBLES               optional space-separated scramble labels

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-models/unit_500k/best_model.pt}"
DEVICE="${DEVICE:-auto}"
DATA_SOURCE="${DATA_SOURCE:-csv}" # Set to generate to generate data instead of reading from CSV.
#INPUT_CSV="${INPUT_CSV:-${EVAL_TEST_CSV:-data/sqed_oneshot_150.csv}}" # Paolo generated data
INPUT_CSV="${INPUT_CSV:-${EVAL_TEST_CSV:-data/sqed_4ptseed_oneshot.csv}}" # Actual 4pt data 
RUN_ID="${RUN_ID:-staged_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-data_testing/outputs/staged_eval}"
SAMPLES="${SAMPLES:-100}"
BEAM_STAGES="${BEAM_STAGES:-5 50 100}"
MAX_BEAM="${MAX_BEAM:-180}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HUMAN_CSV="${HUMAN_CSV:-1}"
PLOTS="${PLOTS:-0}"
NUCLEUS_ATTEMPTS="${NUCLEUS_ATTEMPTS:-4}"
NUCLEUS_P="${NUCLEUS_P:-0.99}"
NUCLEUS_TEMPERATURES="${NUCLEUS_TEMPERATURES:-0.8 1.0 1.2 1.5}"

N_PARTICLES="${N_PARTICLES:-4}"
SEED="${SEED:-123}"
MIN_SCR="${MIN_SCR:-1}"
MAX_SCR="${MAX_SCR:-4}"
MIN_TERMS="${MIN_TERMS:-1}"
MAX_TERMS="${MAX_TERMS:-3}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TOKENIZER_MAX_PARTICLES="${TOKENIZER_MAX_PARTICLES:-8}"
SCRAMBLES="${SCRAMBLES:-}"

case "$DATA_SOURCE" in
  csv|generate) ;;
  *)
    echo "ERROR: DATA_SOURCE must be 'csv' or 'generate', got: $DATA_SOURCE" >&2
    exit 1
    ;;
esac

if [[ "$DATA_SOURCE" == "csv" && -z "$INPUT_CSV" ]]; then
  echo "ERROR: INPUT_CSV is required when DATA_SOURCE=csv" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model checkpoint not found: $MODEL_PATH" >&2
  exit 1
fi

if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SAMPLES must be an integer, got: $SAMPLES" >&2
  exit 1
fi
if ! [[ "$MAX_BEAM" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_BEAM must be an integer, got: $MAX_BEAM" >&2
  exit 1
fi
if ! [[ "$NUCLEUS_ATTEMPTS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NUCLEUS_ATTEMPTS must be an integer, got: $NUCLEUS_ATTEMPTS" >&2
  exit 1
fi

read -r -a BEAMS <<< "$BEAM_STAGES"
read -r -a NUCLEUS_TEMPS <<< "$NUCLEUS_TEMPERATURES"

if [[ "${#BEAMS[@]}" -eq 0 ]]; then
  echo "ERROR: BEAM_STAGES must contain at least one beam size" >&2
  exit 1
fi
if [[ "$NUCLEUS_ATTEMPTS" -gt 0 && "${#NUCLEUS_TEMPS[@]}" -eq 0 ]]; then
  echo "ERROR: NUCLEUS_TEMPERATURES must contain values when NUCLEUS_ATTEMPTS > 0" >&2
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

TMP_DIR="$(mktemp -d /tmp/eval_staged.XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"

STAGE_REPORT="${OUTPUT_DIR}/${RUN_ID}_stage_report.csv"
FINAL_REPORT="${OUTPUT_DIR}/${RUN_ID}_final_report.csv"
OUTPUT_PREFIX="${OUTPUT_DIR#data_testing/outputs}/$RUN_ID"
OUTPUT_PREFIX="${OUTPUT_PREFIX#/}"

plot_flag="--no-plots"
if [[ "$PLOTS" == "1" ]]; then
  plot_flag="--plots"
fi

human_flag="--human-csv"
if [[ "$HUMAN_CSV" != "1" ]]; then
  human_flag="--no-human-csv"
fi

SCRAMBLE_ARGS=()
if [[ -n "$SCRAMBLES" ]]; then
  read -r -a SCRAMBLE_NAMES <<< "$SCRAMBLES"
  SCRAMBLE_ARGS=(--scrambles "${SCRAMBLE_NAMES[@]}")
fi

MAX_TOKEN_ARGS=()
if [[ -n "$MAX_TOKENS" ]]; then
  MAX_TOKEN_ARGS=(--max-tokens "$MAX_TOKENS")
fi

printf 'run_id,stage,method,beam_size,temperature,input_examples,stage_successes,cumulative_successes,remaining_failures,cumulative_accuracy_pct\n' > "$STAGE_REPORT"
printf 'run_id,data_source,input_csv,total_examples,beam_stages,beam_successes,nucleus_attempts,nucleus_successes,cumulative_successes,remaining_failures,final_cumulative_accuracy_pct\n' > "$FINAL_REPORT"

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

accuracy_pct() {
  python3 - "$1" "$2" <<'PY'
import sys
successes = int(sys.argv[1])
total = int(sys.argv[2])
print(f"{100 * successes / total:.6f}" if total else "0.000000")
PY
}

prepare_input_csv() {
  local src="$1"
  local dst="$2"
  case "$src" in
    *.gz) gzip -cd "$src" > "$dst" ;;
    *) cp "$src" "$dst" ;;
  esac
}

run_stage() {
  local method="$1"
  local beam_size="$2"
  local temperature="$3"
  local output_stem="$4"
  shift 4

  local args=(
    python3 data_testing/evaluate_model.py
    --model-path "$MODEL_PATH"
    --device "$DEVICE"
    --n-particles "$N_PARTICLES"
    --num-samples "$SAMPLES"
    --seed "$SEED"
    --min-scr "$MIN_SCR"
    --max-scr "$MAX_SCR"
    --min-terms "$MIN_TERMS"
    --max-terms "$MAX_TERMS"
    --tokenizer-max-particles "$TOKENIZER_MAX_PARTICLES"
    --decoding-method "$method"
    --beam-size "$beam_size"
    --rerank-numerical
    --batch-size "$BATCH_SIZE"
    --output-stem "$output_stem"
    --simple-summary
    "$human_flag"
    "$plot_flag"
    "${MAX_TOKEN_ARGS[@]}"
    "$@"
    "${SCRAMBLE_ARGS[@]}"
  )

  if [[ "$method" == "nucleus" ]]; then
    args+=(--p-nucleus "$NUCLEUS_P" --temperature-nucleus "$temperature")
  fi

  "${args[@]}"
}

current_csv=""
if [[ "$DATA_SOURCE" == "csv" ]]; then
  if [[ ! -f "$INPUT_CSV" ]]; then
    echo "ERROR: input CSV not found: $INPUT_CSV" >&2
    exit 1
  fi
  current_csv="${TMP_DIR}/stage_0_input.csv"
  prepare_input_csv "$INPUT_CSV" "$current_csv"
fi

total_examples=""
cumulative_successes=0
remaining_failures=0
beam_successes=()
nucleus_successes=()
first_stage=1

echo "Run ID: $RUN_ID"
echo "Data source: $DATA_SOURCE"
if [[ "$DATA_SOURCE" == "csv" ]]; then
  echo "Input CSV: $INPUT_CSV"
fi
echo "Beam stages: ${BEAMS[*]}"
echo "Nucleus attempts: $NUCLEUS_ATTEMPTS"
echo "Output dir: $OUTPUT_DIR"

for beam in "${BEAMS[@]}"; do
  if [[ "$first_stage" == "1" && "$DATA_SOURCE" == "generate" ]]; then
    input_count="$SAMPLES"
    data_args=(--data-source generate)
  else
    input_count="$(effective_csv_rows "$current_csv")"
    data_args=(--data-source csv --existing-raw-csv "$current_csv" --existing-csv-max-rows "$SAMPLES" --no-dedupe)
  fi

  if (( input_count <= 0 )); then
    echo "No failures remain before beam=$beam; skipping remaining beam stages."
    break
  fi

  output_stem="${OUTPUT_PREFIX}_beam_${beam}"
  summary_path="data_testing/outputs/${output_stem}_summary.csv"
  detail_path="data_testing/outputs/${output_stem}_beam_results.csv"
  next_csv="${TMP_DIR}/after_beam_${beam}_failures.csv"

  echo "--- beam=${beam} | input rows=${input_count} ---"
  run_stage "beam" "$beam" "" "$output_stem" "${data_args[@]}"

  read -r stage_total stage_successes < <(read_summary "$summary_path")
  if [[ -z "$total_examples" ]]; then
    total_examples="$stage_total"
  fi
  cumulative_successes=$((cumulative_successes + stage_successes))
  remaining_failures="$(write_failures_csv "$detail_path" "$next_csv")"
  beam_successes+=("${beam}:${stage_successes}")
  current_csv="$next_csv"
  first_stage=0

  accuracy="$(accuracy_pct "$cumulative_successes" "$total_examples")"
  printf '%s,beam_%s,beam,%s,,%s,%s,%s,%s,%s\n' \
    "$RUN_ID" "$beam" "$beam" "$stage_total" "$stage_successes" \
    "$cumulative_successes" "$remaining_failures" "$accuracy" >> "$STAGE_REPORT"
done

for ((attempt = 1; attempt <= NUCLEUS_ATTEMPTS; attempt++)); do
  if [[ "$first_stage" == "1" && "$DATA_SOURCE" == "generate" ]]; then
    input_count="$SAMPLES"
    data_args=(--data-source generate)
  else
    input_count="$(effective_csv_rows "$current_csv")"
    data_args=(--data-source csv --existing-raw-csv "$current_csv" --existing-csv-max-rows "$SAMPLES" --no-dedupe)
  fi

  if (( input_count <= 0 )); then
    echo "No failures remain before nucleus attempt ${attempt}; skipping remaining attempts."
    break
  fi

  temp_index=$(((attempt - 1) % ${#NUCLEUS_TEMPS[@]}))
  temperature="${NUCLEUS_TEMPS[$temp_index]}"
  temp_label="${temperature//./p}"
  output_stem="${OUTPUT_PREFIX}_nucleus_${attempt}_t${temp_label}"
  summary_path="data_testing/outputs/${output_stem}_summary.csv"
  detail_path="data_testing/outputs/${output_stem}_nucleus_results.csv"
  next_csv="${TMP_DIR}/after_nucleus_${attempt}_failures.csv"

  echo "--- nucleus attempt=${attempt} | beam=${NUCLEUS_BEAM_SIZE} | temp=${temperature} | input rows=${input_count} ---"
  run_stage "nucleus" "$NUCLEUS_BEAM_SIZE" "$temperature" "$output_stem" "${data_args[@]}"

  read -r stage_total stage_successes < <(read_summary "$summary_path")
  if [[ -z "$total_examples" ]]; then
    total_examples="$stage_total"
  fi
  cumulative_successes=$((cumulative_successes + stage_successes))
  remaining_failures="$(write_failures_csv "$detail_path" "$next_csv")"
  nucleus_successes+=("${attempt}@${temperature}:${stage_successes}")
  current_csv="$next_csv"
  first_stage=0

  accuracy="$(accuracy_pct "$cumulative_successes" "$total_examples")"
  printf '%s,nucleus_%s,nucleus,%s,%s,%s,%s,%s,%s,%s\n' \
    "$RUN_ID" "$attempt" "$NUCLEUS_BEAM_SIZE" "$temperature" "$stage_total" \
    "$stage_successes" "$cumulative_successes" "$remaining_failures" "$accuracy" >> "$STAGE_REPORT"
done

if [[ -z "$total_examples" ]]; then
  total_examples=0
fi
final_accuracy="$(accuracy_pct "$cumulative_successes" "$total_examples")"

python3 - "$FINAL_REPORT" \
  "$RUN_ID" \
  "$DATA_SOURCE" \
  "${INPUT_CSV:-}" \
  "$total_examples" \
  "${BEAMS[*]}" \
  "${beam_successes[*]:-}" \
  "$NUCLEUS_ATTEMPTS" \
  "${nucleus_successes[*]:-}" \
  "$cumulative_successes" \
  "$remaining_failures" \
  "$final_accuracy" <<'PY'
import csv
import sys

path = sys.argv[1]
row = sys.argv[2:]

with open(path, "a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(row)
PY

echo "Staged evaluation complete: ${cumulative_successes}/${total_examples} correct (${final_accuracy}%)."
echo "Wrote stage report to $STAGE_REPORT"
echo "Wrote final report to $FINAL_REPORT"
