#!/usr/bin/env bash
set -euo pipefail

# Space-separated run names to compare
RUN_NAMES="${RUN_NAMES:-unit_100k unit_200k unit_500k}"
# Space-separated test CSV paths
TEST_CSVS="${TEST_CSVS:-data/sqed_w0110_mM00M_den6_maxD2_20000_phys_oneshot.csv}"
# Max samples to evaluate per dataset (use "all" for no cap)
SAMPLES="${SAMPLES:-1000}"
# Beam size for test CSV evaluation
BEAM_SIZE="${BEAM_SIZE:-10}"
# Beam size for generated data evaluation (defaults to BEAM_SIZE)
BEAM_SIZE_GENERATED="${BEAM_SIZE_GENERATED:-10}"
# Set to 0 to skip freshly generated data evaluation
EVAL_GENERATED="${EVAL_GENERATED:-1}"
# Set to 0 to use the original verbose per-metric summary
SIMPLE_SUMMARY="${SIMPLE_SUMMARY:-1}"

# Fixed settings
SEED="${SEED:-123}"
DEVICE="${DEVICE:-auto}"
N_PARTICLES="${N_PARTICLES:-4}"
MAX_STEPS="${MAX_STEPS:-8}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TOKENIZER_MAX_PARTICLES="${TOKENIZER_MAX_PARTICLES:-8}"

read -r -a RUNS <<< "$RUN_NAMES"
read -r -a CSVS <<< "$TEST_CSVS"

mkdir -p results

run_eval() {
  local run="$1"; shift
  local output_stem="$1"; shift
  local summary_flag="--simple-summary"
  [[ "$SIMPLE_SUMMARY" == "1" ]] || summary_flag="--no-simple-summary"
  python3 data_testing/evaluate_model_on_generated_data.py \
    --model-path       "models/${run}/best_model.pt" \
    --device           "$DEVICE" \
    --n-particles      "$N_PARTICLES" \
    --num-samples      "$SAMPLES" \
    --seed             "$SEED" \
    --max-tokens       "$MAX_TOKENS" \
    --tokenizer-max-particles "$TOKENIZER_MAX_PARTICLES" \
    --max-steps        "$MAX_STEPS" \
    --decoding-method  beam \
    --beam-size        "$BEAM_SIZE" \
    --rerank-numerical \
    --output-stem      "$output_stem" \
    "$summary_flag" \
    "$@"
}

for run in "${RUNS[@]}"; do
  model="models/${run}/best_model.pt"
  if [[ ! -f "$model" ]]; then
    echo "WARNING: model not found: $model — skipping $run"
    continue
  fi

  if [[ "$EVAL_GENERATED" == "1" ]]; then
    echo "=== $run | generated | beam=$BEAM_SIZE_GENERATED | samples=$SAMPLES ==="
    run_eval "$run" "results/compare_${run}_generated_beam${BEAM_SIZE_GENERATED}" \
      --beam-size "$BEAM_SIZE_GENERATED" \
      --data-source generate
    echo ""
  fi

  for csv in "${CSVS[@]}"; do
    csv_name="$(basename "$csv" .csv)"
    echo "=== $run | $csv_name | beam=$BEAM_SIZE | samples=$SAMPLES ==="

    max_rows_args=()
    if [[ "$SAMPLES" != "all" && "$SAMPLES" != "ALL" ]]; then
      max_rows_args=(--existing-csv-max-rows "$SAMPLES")
    fi

    run_eval "$run" "results/compare_${run}_${csv_name}_beam${BEAM_SIZE}" \
      --data-source csv --existing-raw-csv "$csv" "${max_rows_args[@]}"
    echo ""
  done
done

echo "All evaluations complete."
