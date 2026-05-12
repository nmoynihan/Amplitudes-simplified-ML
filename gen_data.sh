#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${MODE:-train_eval}}"

N_PARTICLES="${N_PARTICLES:-4}"
SAMPLES="${SAMPLES:-5000}"
MIN_SCR="${MIN_SCR:-1}"
MAX_SCR="${MAX_SCR:-5}"
MIN_TERMS="${MIN_TERMS:-1}"
MAX_TERMS="${MAX_TERMS:-5}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TOKENIZER_MAX_PARTICLES="${TOKENIZER_MAX_PARTICLES:-8}"
DATASET_KIND="${DATASET_KIND:-oneshot}"

# New generator controls.
JOBS="${JOBS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2000}"
UNIT_PROBABILITY="${UNIT_PROBABILITY:-0.2}"
OLD_STYLE_PROBABILITY="${OLD_STYLE_PROBABILITY:-0.25}"
SPURIOUS_REPEAT_PROBABILITY="${SPURIOUS_REPEAT_PROBABILITY:-${DENOM_REPEAT_PROBABILITY:-0.35}}"
SCALAR_POWER_PROBABILITY="${SCALAR_POWER_PROBABILITY:-0.15}"

# Stratified broad dataset.  This keeps most examples broad while reserving a
# controlled slice for SQED-like coverage and a small slice for harder rewrites.
MIXED_PROFILE="${MIXED_PROFILE:-1}"
BROAD_PERCENT="${BROAD_PERCENT:-70}"
SQED_COVER_PERCENT="${SQED_COVER_PERCENT:-20}"
HARD_PERCENT="${HARD_PERCENT:-10}"
SQED_COVER_OLD_STYLE_PROBABILITY="${SQED_COVER_OLD_STYLE_PROBABILITY:-0.45}"
SQED_COVER_UNIT_PROBABILITY="${SQED_COVER_UNIT_PROBABILITY:-0.6}"
SQED_COVER_SPURIOUS_REPEAT_PROBABILITY="${SQED_COVER_SPURIOUS_REPEAT_PROBABILITY:-0.15}"
SQED_COVER_SCALAR_POWER_PROBABILITY="${SQED_COVER_SCALAR_POWER_PROBABILITY:-0.1}"
SQED_COVER_SCRAMBLES="${SQED_COVER_SCRAMBLES:-multiply_one ward momentum commute_dot ratio}"
BROAD_SCRAMBLES="${BROAD_SCRAMBLES:-all}"
HARD_SCRAMBLES="${HARD_SCRAMBLES:-all}"
HARD_MIN_TERMS="${HARD_MIN_TERMS:-3}"
HARD_MAX_TERMS="${HARD_MAX_TERMS:-$MAX_TERMS}"
HARD_MIN_SCR="${HARD_MIN_SCR:-3}"
HARD_MAX_SCR="${HARD_MAX_SCR:-$MAX_SCR}"

# Optional boolean switches. Accept 1/true/yes/on.
NO_PROGRESS="${NO_PROGRESS:-0}"
NO_TOKENISE="${NO_TOKENISE:-0}"
NO_VALIDATE="${NO_VALIDATE:-0}"
GROUPED_SCRAMBLED="${GROUPED_SCRAMBLED:-0}"

# Optional raw extra arguments passed through to data_gen/gen_data.py.
# Example: GEN_EXTRA_ARGS="--mass 2.0 --max-attempts-factor 20"
GEN_EXTRA_ARGS="${GEN_EXTRA_ARGS:-}"

RAW_OUT="${RAW_OUT:-data/gi_${N_PARTICLES}pt_os.csv}"
TOK_OUT="${TOK_OUT:-data/gi_${N_PARTICLES}pt_os_tok.csv}"
LOG_OUT="${LOG_OUT:-gen_data_${N_PARTICLES}pt_os.log}"

mkdir -p "$(dirname "$RAW_OUT")"
mkdir -p "$(dirname "$TOK_OUT")"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

EXTRA_ARGS=()

if truthy "$NO_PROGRESS"; then
  EXTRA_ARGS+=(--no-progress)
fi

if truthy "$NO_TOKENISE"; then
  EXTRA_ARGS+=(--no-tokenise)
fi

if truthy "$NO_VALIDATE"; then
  EXTRA_ARGS+=(--no-validate)
fi

if truthy "$GROUPED_SCRAMBLED"; then
  EXTRA_ARGS+=(--grouped-scrambled)
fi

if [[ -n "$GEN_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=($GEN_EXTRA_ARGS)
fi

run_generator() {
  local samples="$1"
  local seed="$2"
  local raw_out="$3"
  local tok_out="$4"
  local log_out="$5"
  shift 5

  python3 data_gen/gen_data.py "$N_PARTICLES" \
    --dataset-kind "$DATASET_KIND" \
    --samples "$samples" \
    --min-scr "$MIN_SCR" \
    --max-scr "$MAX_SCR" \
    --min-terms "$MIN_TERMS" \
    --max-terms "$MAX_TERMS" \
    --seed "$seed" \
    --jobs "$JOBS" \
    --batch-size "$BATCH_SIZE" \
    --unit-probability "$UNIT_PROBABILITY" \
    --old-style-probability "$OLD_STYLE_PROBABILITY" \
    --spurious-repeat-probability "$SPURIOUS_REPEAT_PROBABILITY" \
    --scalar-power-probability "$SCALAR_POWER_PROBABILITY" \
    --max-tokens "$MAX_TOKENS" \
    --tokenizer-max-particles "$TOKENIZER_MAX_PARTICLES" \
    --raw-out "$raw_out" \
    --tok-out "$tok_out" \
    --log-out "$log_out" \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

tokenise_final_csv() {
  python3 - "$RAW_OUT" "$TOK_OUT" "$TOKENIZER_MAX_PARTICLES" "$MAX_TOKENS" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("data_gen").resolve()))
import gen_data as gd

raw_out, tok_out, max_particles, max_tokens = sys.argv[1:5]
max_tokens_value = None if int(max_tokens) <= 0 else int(max_tokens)
gd.tokenise_csv(
    raw_out,
    tok_out,
    max_particles=int(max_particles),
    max_sequence_length=max_tokens_value,
)
PY
}

if truthy "$MIXED_PROFILE"; then
  PERCENT_TOTAL=$((BROAD_PERCENT + SQED_COVER_PERCENT + HARD_PERCENT))
  if [[ "$PERCENT_TOTAL" -le 0 ]]; then
    echo "BROAD_PERCENT + SQED_COVER_PERCENT + HARD_PERCENT must be positive."
    exit 1
  fi

  BROAD_SAMPLES=$((SAMPLES * BROAD_PERCENT / PERCENT_TOTAL))
  SQED_COVER_SAMPLES=$((SAMPLES * SQED_COVER_PERCENT / PERCENT_TOTAL))
  HARD_SAMPLES=$((SAMPLES - BROAD_SAMPLES - SQED_COVER_SAMPLES))
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT

  echo "# mixed gen_data.sh profile" > "$LOG_OUT"
  echo "# broad=$BROAD_SAMPLES sqed_cover=$SQED_COVER_SAMPLES hard=$HARD_SAMPLES total=$SAMPLES" >> "$LOG_OUT"
  printf "simple,scrambled\n" > "$RAW_OUT"

  if [[ "$BROAD_SAMPLES" -gt 0 ]]; then
    read -r -a BROAD_SCRAMBLE_NAMES <<< "$BROAD_SCRAMBLES"
    run_generator \
      "$BROAD_SAMPLES" "$SEED" \
      "$TMP_DIR/broad.csv" "$TMP_DIR/broad_tok.csv" "$TMP_DIR/broad.log" \
      --no-tokenise \
      --scrambles "${BROAD_SCRAMBLE_NAMES[@]}"
    tail -n +2 "$TMP_DIR/broad.csv" >> "$RAW_OUT"
    cat "$TMP_DIR/broad.log" >> "$LOG_OUT"
  fi

  if [[ "$SQED_COVER_SAMPLES" -gt 0 ]]; then
    read -r -a SQED_COVER_SCRAMBLE_NAMES <<< "$SQED_COVER_SCRAMBLES"
    run_generator \
      "$SQED_COVER_SAMPLES" "$((SEED + 1000003))" \
      "$TMP_DIR/sqed_cover.csv" "$TMP_DIR/sqed_cover_tok.csv" "$TMP_DIR/sqed_cover.log" \
      --no-tokenise \
      --min-terms 1 \
      --max-terms 2 \
      --old-style-probability "$SQED_COVER_OLD_STYLE_PROBABILITY" \
      --unit-probability "$SQED_COVER_UNIT_PROBABILITY" \
      --spurious-repeat-probability "$SQED_COVER_SPURIOUS_REPEAT_PROBABILITY" \
      --scalar-power-probability "$SQED_COVER_SCALAR_POWER_PROBABILITY" \
      --scrambles "${SQED_COVER_SCRAMBLE_NAMES[@]}"
    tail -n +2 "$TMP_DIR/sqed_cover.csv" >> "$RAW_OUT"
    cat "$TMP_DIR/sqed_cover.log" >> "$LOG_OUT"
  fi

  if [[ "$HARD_SAMPLES" -gt 0 ]]; then
    read -r -a HARD_SCRAMBLE_NAMES <<< "$HARD_SCRAMBLES"
    run_generator \
      "$HARD_SAMPLES" "$((SEED + 2000006))" \
      "$TMP_DIR/hard.csv" "$TMP_DIR/hard_tok.csv" "$TMP_DIR/hard.log" \
      --no-tokenise \
      --min-terms "$HARD_MIN_TERMS" \
      --max-terms "$HARD_MAX_TERMS" \
      --min-scr "$HARD_MIN_SCR" \
      --max-scr "$HARD_MAX_SCR" \
      --scrambles "${HARD_SCRAMBLE_NAMES[@]}"
    tail -n +2 "$TMP_DIR/hard.csv" >> "$RAW_OUT"
    cat "$TMP_DIR/hard.log" >> "$LOG_OUT"
  fi

  if ! truthy "$NO_TOKENISE"; then
    tokenise_final_csv
  fi
else
  run_generator "$SAMPLES" "$SEED" "$RAW_OUT" "$TOK_OUT" "$LOG_OUT"
fi


echo "Completed data generation. Full log written to $LOG_OUT"
echo "Raw output: $RAW_OUT"
if ! truthy "$NO_TOKENISE"; then
  echo "Tokenized output: $TOK_OUT"
fi
