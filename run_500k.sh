#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${MODE:-train_eval}}"

N_PARTICLES="${N_PARTICLES:-4}"
SAMPLES="${SAMPLES:-500000}"
MIN_SCR="${MIN_SCR:-1}"
MAX_SCR="${MAX_SCR:-4}"
MIN_TERMS="${MIN_TERMS:-1}"
MAX_TERMS="${MAX_TERMS:-3}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TOKENIZER_MAX_PARTICLES="${TOKENIZER_MAX_PARTICLES:-8}"
DATASET_KIND="${DATASET_KIND:-oneshot}"
SCRAMBLES="${SCRAMBLES:-}"
USER_SET_SKIP_DATA_GEN="${SKIP_DATA_GEN+x}"
SKIP_DATA_GEN="${SKIP_DATA_GEN:-0}"

# SQED-compatible broad default profile.  This keeps a broad slice while
# reducing nuisance coefficient variation that is absent from sqed_oneshot_150.
JOBS="${JOBS:-8}"
BATCH_SIZE_GEN="${BATCH_SIZE_GEN:-2000}"
UNIT_PROBABILITY="${UNIT_PROBABILITY:-0.9}"
OLD_STYLE_PROBABILITY="${OLD_STYLE_PROBABILITY:-0.4}"
SPURIOUS_REPEAT_PROBABILITY="${SPURIOUS_REPEAT_PROBABILITY:-${DENOM_REPEAT_PROBABILITY:-0.35}}"
SCALAR_POWER_PROBABILITY="${SCALAR_POWER_PROBABILITY:-0.15}"
MIXED_PROFILE="${MIXED_PROFILE:-1}"
BROAD_PERCENT="${BROAD_PERCENT:-60}"
SQED_COVER_PERCENT="${SQED_COVER_PERCENT:-30}"
HARD_PERCENT="${HARD_PERCENT:-10}"
SQED_COVER_OLD_STYLE_PROBABILITY="${SQED_COVER_OLD_STYLE_PROBABILITY:-0.65}"
SQED_COVER_UNIT_PROBABILITY="${SQED_COVER_UNIT_PROBABILITY:-1.0}"
SQED_COVER_SPURIOUS_REPEAT_PROBABILITY="${SQED_COVER_SPURIOUS_REPEAT_PROBABILITY:-0.15}"
SQED_COVER_SCALAR_POWER_PROBABILITY="${SQED_COVER_SCALAR_POWER_PROBABILITY:-0.1}"
SQED_COVER_SCRAMBLES="${SQED_COVER_SCRAMBLES:-${SCRAMBLES:-multiply_one ward momentum commute_dot ratio}}"
BROAD_SCRAMBLES="${BROAD_SCRAMBLES:-${SCRAMBLES:-all}}"
HARD_SCRAMBLES="${HARD_SCRAMBLES:-${SCRAMBLES:-all}}"
HARD_MIN_TERMS="${HARD_MIN_TERMS:-3}"
HARD_MAX_TERMS="${HARD_MAX_TERMS:-$MAX_TERMS}"
HARD_MIN_SCR="${HARD_MIN_SCR:-3}"
HARD_MAX_SCR="${HARD_MAX_SCR:-$MAX_SCR}"
NO_PROGRESS="${NO_PROGRESS:-0}"
NO_VALIDATE="${NO_VALIDATE:-0}"
GROUPED_SCRAMBLED="${GROUPED_SCRAMBLED:-0}"
GEN_EXTRA_ARGS="${GEN_EXTRA_ARGS:-}"

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

RUN_NAME="${RUN_NAME:-unit_500k}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.8}"
TRAIN_SAMPLE_SEED="${TRAIN_SAMPLE_SEED:-$SEED}"
TRAIN_TOK_SUBSET="${TRAIN_TOK_SUBSET:-data/${RUN_NAME}_train_${SAMPLES}_tok.csv}"
EMBEDDING_DIM="${EMBEDDING_DIM:-384}"
N_HEADS="${N_HEADS:-6}"
N_ENC_LAYERS="${N_ENC_LAYERS:-5}"
N_DEC_LAYERS="${N_DEC_LAYERS:-5}"
HEAD_FF_DIM="${HEAD_FF_DIM:-1024}"
DROPOUT="${DROPOUT:-0.05}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
DYNAMIC_PADDING="${DYNAMIC_PADDING:-1}"
BUCKETING="${BUCKETING:-1}"
BUCKET_SIZE_MULTIPLIER="${BUCKET_SIZE_MULTIPLIER:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
RESUME="${RESUME:-0}"
RESUME_FROM="${RESUME_FROM:-}"
if truthy "$RESUME" && [[ -z "$USER_SET_SKIP_DATA_GEN" ]]; then
  SKIP_DATA_GEN=1
fi

EVAL_SAMPLES="${EVAL_SAMPLES:-150}"
EVAL_SEED="${EVAL_SEED:-123}"
EVAL_MIN_SCR="${EVAL_MIN_SCR:-$MIN_SCR}"
EVAL_MAX_SCR="${EVAL_MAX_SCR:-$MAX_SCR}"
EVAL_MIN_TERMS="${EVAL_MIN_TERMS:-$MIN_TERMS}"
EVAL_MAX_TERMS="${EVAL_MAX_TERMS:-$MAX_TERMS}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-8}"
EVAL_BEAM_SIZE="${EVAL_BEAM_SIZE:-8}"
EVAL_DECODING_METHOD="${EVAL_DECODING_METHOD:-beam}"
EVAL_RERANK_NUMERICAL="${EVAL_RERANK_NUMERICAL:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_KIND="${EVAL_KIND:-oneshot}"
EVAL_OUTPUT_STEM="${EVAL_OUTPUT_STEM:-iterative_eval_${N_PARTICLES}pt_${RUN_NAME}}"
EVAL_GENERATED_OUTPUT_STEM="${EVAL_GENERATED_OUTPUT_STEM:-${EVAL_OUTPUT_STEM}_generated}"
EVAL_TEST_OUTPUT_STEM="${EVAL_TEST_OUTPUT_STEM:-${EVAL_OUTPUT_STEM}_testcsv}"
EVAL_TEST_CSV="${EVAL_TEST_CSV:-data/sqed_oneshot_150.csv}"
EVAL_TEST_CSV_MAX_ROWS="${EVAL_TEST_CSV_MAX_ROWS:-$EVAL_SAMPLES}"

RAW_OUT="${RAW_OUT:-data/gi_${N_PARTICLES}pt.csv}"
TOK_OUT="${TOK_OUT:-data/gi_${N_PARTICLES}pt_tok.csv}"
LOG_OUT="${LOG_OUT:-gen_data_${N_PARTICLES}pt.log}"
MODEL_PATH="${MODEL_PATH:-models/${RUN_NAME}/best_model.pt}"
RUN_LOG="${RUN_LOG:-logs/run_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$RAW_OUT")"
mkdir -p "$(dirname "$TOK_OUT")"
mkdir -p "$(dirname "$RUN_LOG")"

exec > >(tee -a "$RUN_LOG") 2>&1

case "$MODE" in
  train_eval|full)
    DO_TRAIN=1
    DO_EVAL=1
    ;;
  train-only|train_only|train)
    DO_TRAIN=1
    DO_EVAL=0
    ;;
  eval-only|eval_only|eval|--eval-only)
    DO_TRAIN=0
    DO_EVAL=1
    ;;
  step-eval|step_eval|step-eval-only|step_eval_only)
    DO_TRAIN=0
    DO_EVAL=1
    EVAL_KIND="step"
    if [[ "${EVAL_OUTPUT_STEM}" == iterative_eval_* ]]; then
      EVAL_OUTPUT_STEM="step_pair_eval_${N_PARTICLES}pt_${RUN_NAME}"
    fi
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use: train_eval, train-only, eval-only, or step-eval"
    exit 2
    ;;
esac

if [[ "$DO_TRAIN" -eq 1 ]] && truthy "$RESUME"; then
  if [[ -z "$RESUME_FROM" ]]; then
    RESUME_FROM="$(python3 - "$RUN_NAME" <<'PY'
import re
import sys
from pathlib import Path

run_name = sys.argv[1]
model_dir = Path("models") / run_name
candidates = []
for path in model_dir.glob("model_epoch_*.pt"):
    match = re.fullmatch(r"model_epoch_(\d+)\.pt", path.name)
    if match:
        candidates.append((int(match.group(1)), path))

if candidates:
    print(max(candidates, key=lambda item: item[0])[1])
else:
    best = model_dir / "best_model.pt"
    if best.exists():
        print(best)
    else:
        raise SystemExit(f"No checkpoint found in {model_dir}")
PY
)"
  fi

  if [[ ! -f "$RESUME_FROM" ]]; then
    echo "Resume checkpoint not found: $RESUME_FROM"
    exit 1
  fi
fi

echo "Run log: $RUN_LOG"
echo "Mode: $MODE"
echo "N_PARTICLES=$N_PARTICLES RUN_NAME=$RUN_NAME MODEL_PATH=$MODEL_PATH"
echo "SCRAMBLES=${SCRAMBLES:-all}"
echo "SKIP_DATA_GEN=$SKIP_DATA_GEN"
echo "MIXED_PROFILE=$MIXED_PROFILE BROAD/SQED/HARD=${BROAD_PERCENT}/${SQED_COVER_PERCENT}/${HARD_PERCENT}"
echo "Generator probabilities: UNIT=$UNIT_PROBABILITY OLD_STYLE=$OLD_STYLE_PROBABILITY SQED_UNIT=$SQED_COVER_UNIT_PROBABILITY SQED_OLD_STYLE=$SQED_COVER_OLD_STYLE_PROBABILITY"
echo "Model size: EMBEDDING_DIM=$EMBEDDING_DIM N_HEADS=$N_HEADS ENC/DEC=${N_ENC_LAYERS}/${N_DEC_LAYERS} HEAD_FF_DIM=$HEAD_FF_DIM DROPOUT=$DROPOUT"
echo "Training speed: BATCH_SIZE=$BATCH_SIZE TRAIN_SPLIT=$TRAIN_SPLIT AMP=$AMP AMP_DTYPE=$AMP_DTYPE DYNAMIC_PADDING=$DYNAMIC_PADDING BUCKETING=$BUCKETING MAX_TOKENS=$MAX_TOKENS"
echo "RESUME=$RESUME"
if [[ -n "$RESUME_FROM" ]]; then
  echo "RESUME_FROM=$RESUME_FROM"
fi
echo "Training cap: SAMPLES=$SAMPLES TRAIN_SAMPLE_SEED=$TRAIN_SAMPLE_SEED"
if [[ -n "$EVAL_TEST_CSV" ]]; then
  echo "EVAL_TEST_CSV=$EVAL_TEST_CSV"
  echo "EVAL_TEST_CSV_MAX_ROWS=$EVAL_TEST_CSV_MAX_ROWS"
fi
echo "EVAL_GENERATED_OUTPUT_STEM=$EVAL_GENERATED_OUTPUT_STEM"
echo "EVAL_RERANK_NUMERICAL=$EVAL_RERANK_NUMERICAL"
if [[ -n "$EVAL_TEST_CSV" ]]; then
  echo "EVAL_TEST_OUTPUT_STEM=$EVAL_TEST_OUTPUT_STEM"
fi

SCRAMBLE_ARGS=()
if [[ -n "$SCRAMBLES" ]]; then
  read -r -a SCRAMBLE_NAMES <<< "$SCRAMBLES"
  SCRAMBLE_ARGS=(--scrambles "${SCRAMBLE_NAMES[@]}")
fi

EVAL_RERANK_ARGS=()
if truthy "$EVAL_RERANK_NUMERICAL"; then
  EVAL_RERANK_ARGS=(--rerank-numerical)
else
  EVAL_RERANK_ARGS=(--no-rerank-numerical)
fi

EVAL_DATA_ARGS=()
if [[ -n "$EVAL_TEST_CSV" ]]; then
  EVAL_DATA_ARGS=(--data-source csv --existing-raw-csv "$EVAL_TEST_CSV")
  if [[ -n "$EVAL_TEST_CSV_MAX_ROWS" && "$EVAL_TEST_CSV_MAX_ROWS" != "all" && "$EVAL_TEST_CSV_MAX_ROWS" != "ALL" ]]; then
    EVAL_DATA_ARGS+=(--existing-csv-max-rows "$EVAL_TEST_CSV_MAX_ROWS")
  fi
fi

run_oneshot_eval() {
  local output_stem="$1"
  shift
  python3 data_testing/evaluate_model_on_generated_data.py \
    --model-path "$MODEL_PATH" \
    --device "$EVAL_DEVICE" \
    --n-particles "$N_PARTICLES" \
    --num-samples "$EVAL_SAMPLES" \
    --seed "$EVAL_SEED" \
    --min-scr "$EVAL_MIN_SCR" \
    --max-scr "$EVAL_MAX_SCR" \
    --min-terms "$EVAL_MIN_TERMS" \
    --max-terms "$EVAL_MAX_TERMS" \
    --max-tokens "$MAX_TOKENS" \
    --tokenizer-max-particles "$TOKENIZER_MAX_PARTICLES" \
    --max-steps "$EVAL_MAX_STEPS" \
    --decoding-method "$EVAL_DECODING_METHOD" \
    --beam-size "$EVAL_BEAM_SIZE" \
    "${EVAL_RERANK_ARGS[@]}" \
    --output-stem "$output_stem" \
    "$@" \
    "${SCRAMBLE_ARGS[@]}"
}

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$(pwd)/$1" ;;
  esac
}

prepare_training_csv() {
  local src="$1"
  local dst="$2"
  local samples="$3"
  local seed="$4"

  python3 - "$src" "$dst" "$samples" "$seed" <<'PY'
import csv
import random
import sys
from pathlib import Path

src, dst, samples, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

with open(src, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

if fieldnames is None:
    raise SystemExit(f"Training CSV has no header: {src}")

total = len(rows)
if samples <= 0 or total <= samples:
    print(src)
    print(f"Training data rows: {total}; no cap applied.", file=sys.stderr)
    raise SystemExit(0)

rng = random.Random(seed)
rng.shuffle(rows)
rows = rows[:samples]

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(dst)
print(
    f"Training data rows: capped {total} -> {samples} using seed {seed}; wrote {dst}",
    file=sys.stderr,
)
PY
}

if [[ "$DO_TRAIN" -eq 1 ]]; then
  if truthy "$SKIP_DATA_GEN"; then
    echo "Skipping data generation; using existing tokenized data: $TOK_OUT"
    if [[ ! -f "$TOK_OUT" ]]; then
      echo "Tokenized data file not found: $TOK_OUT"
      echo "Set TOK_OUT=/path/to/existing_tokenized.csv or disable SKIP_DATA_GEN."
      exit 1
    fi
  else
    echo "Generating training data with gen_data.sh profile..."
    env \
      N_PARTICLES="$N_PARTICLES" \
      SAMPLES="$SAMPLES" \
      MIN_SCR="$MIN_SCR" \
      MAX_SCR="$MAX_SCR" \
      MIN_TERMS="$MIN_TERMS" \
      MAX_TERMS="$MAX_TERMS" \
      SEED="$SEED" \
      MAX_TOKENS="$MAX_TOKENS" \
      TOKENIZER_MAX_PARTICLES="$TOKENIZER_MAX_PARTICLES" \
      DATASET_KIND="$DATASET_KIND" \
      JOBS="$JOBS" \
      BATCH_SIZE="$BATCH_SIZE_GEN" \
      UNIT_PROBABILITY="$UNIT_PROBABILITY" \
      OLD_STYLE_PROBABILITY="$OLD_STYLE_PROBABILITY" \
      SPURIOUS_REPEAT_PROBABILITY="$SPURIOUS_REPEAT_PROBABILITY" \
      SCALAR_POWER_PROBABILITY="$SCALAR_POWER_PROBABILITY" \
      MIXED_PROFILE="$MIXED_PROFILE" \
      BROAD_PERCENT="$BROAD_PERCENT" \
      SQED_COVER_PERCENT="$SQED_COVER_PERCENT" \
      HARD_PERCENT="$HARD_PERCENT" \
      BROAD_SCRAMBLES="$BROAD_SCRAMBLES" \
      SQED_COVER_SCRAMBLES="$SQED_COVER_SCRAMBLES" \
      HARD_SCRAMBLES="$HARD_SCRAMBLES" \
      SQED_COVER_OLD_STYLE_PROBABILITY="$SQED_COVER_OLD_STYLE_PROBABILITY" \
      SQED_COVER_UNIT_PROBABILITY="$SQED_COVER_UNIT_PROBABILITY" \
      SQED_COVER_SPURIOUS_REPEAT_PROBABILITY="$SQED_COVER_SPURIOUS_REPEAT_PROBABILITY" \
      SQED_COVER_SCALAR_POWER_PROBABILITY="$SQED_COVER_SCALAR_POWER_PROBABILITY" \
      HARD_MIN_TERMS="$HARD_MIN_TERMS" \
      HARD_MAX_TERMS="$HARD_MAX_TERMS" \
      HARD_MIN_SCR="$HARD_MIN_SCR" \
      HARD_MAX_SCR="$HARD_MAX_SCR" \
      NO_PROGRESS="$NO_PROGRESS" \
      NO_TOKENISE=0 \
      NO_VALIDATE="$NO_VALIDATE" \
      GROUPED_SCRAMBLED="$GROUPED_SCRAMBLED" \
      GEN_EXTRA_ARGS="$GEN_EXTRA_ARGS" \
      RAW_OUT="$RAW_OUT" \
      TOK_OUT="$TOK_OUT" \
      LOG_OUT="$LOG_OUT" \
      ./gen_data.sh
  fi

  TOK_OUT_ABS="$(abspath "$TOK_OUT")"
  TRAIN_TOK_SUBSET_ABS="$(abspath "$TRAIN_TOK_SUBSET")"
  TRAIN_TOK_OUT="$(prepare_training_csv "$TOK_OUT_ABS" "$TRAIN_TOK_SUBSET_ABS" "$SAMPLES" "$TRAIN_SAMPLE_SEED")"

  TRAIN_ARGS=(
    --run_name "$RUN_NAME"
    --data-files "$TRAIN_TOK_OUT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --train-split "$TRAIN_SPLIT"
    --embedding-dim "$EMBEDDING_DIM"
    --n-heads "$N_HEADS"
    --n-enc-layers "$N_ENC_LAYERS"
    --n-dec-layers "$N_DEC_LAYERS"
    --head-ff-dim "$HEAD_FF_DIM"
    --dropout "$DROPOUT"
    --amp-dtype "$AMP_DTYPE"
    --bucket-size-multiplier "$BUCKET_SIZE_MULTIPLIER"
    --num-workers "$NUM_WORKERS"
  )

  if truthy "$AMP"; then
    TRAIN_ARGS+=(--amp)
  else
    TRAIN_ARGS+=(--no-amp)
  fi

  if truthy "$DYNAMIC_PADDING"; then
    TRAIN_ARGS+=(--dynamic-padding)
  else
    TRAIN_ARGS+=(--no-dynamic-padding)
  fi

  if truthy "$BUCKETING"; then
    TRAIN_ARGS+=(--bucketing)
  else
    TRAIN_ARGS+=(--no-bucketing)
  fi

  if truthy "$PIN_MEMORY"; then
    TRAIN_ARGS+=(--pin-memory)
  else
    TRAIN_ARGS+=(--no-pin-memory)
  fi

  if [[ -n "$TRAIN_MAX_LENGTH" ]]; then
    TRAIN_ARGS+=(--max-length "$TRAIN_MAX_LENGTH")
  fi

  if [[ -n "$RESUME_FROM" ]]; then
    TRAIN_ARGS+=(--resume-from "$RESUME_FROM")
  fi

  python3 transformer/transformer_trainer.py \
    "${TRAIN_ARGS[@]}"
fi

if [[ "$DO_EVAL" -eq 1 ]]; then
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model checkpoint not found: $MODEL_PATH"
    echo "Set MODEL_PATH=/path/to/best_model.pt or run training first."
    exit 1
  fi

  if [[ "$EVAL_KIND" == "step" ]]; then
    python3 data_testing/evaluate_step_pairs.py \
      --model-path "$MODEL_PATH" \
      --device "$EVAL_DEVICE" \
      --n-particles "$N_PARTICLES" \
      --num-samples "$EVAL_SAMPLES" \
      --seed "$EVAL_SEED" \
      --min-scr "$EVAL_MIN_SCR" \
      --max-scr "$EVAL_MAX_SCR" \
      --min-terms "$EVAL_MIN_TERMS" \
      --max-terms "$EVAL_MAX_TERMS" \
      --max-tokens "$MAX_TOKENS" \
      --tokenizer-max-particles "$TOKENIZER_MAX_PARTICLES" \
      --decoding-method "$EVAL_DECODING_METHOD" \
      --beam-size "$EVAL_BEAM_SIZE" \
      --output-stem "$EVAL_OUTPUT_STEM" \
      "${SCRAMBLE_ARGS[@]}"
  else
    echo "Evaluating on generated held-out data..."
    run_oneshot_eval "$EVAL_GENERATED_OUTPUT_STEM" --data-source generate

    if [[ -n "$EVAL_TEST_CSV" ]]; then
      echo "Evaluating on external test CSV: $EVAL_TEST_CSV"
      run_oneshot_eval "$EVAL_TEST_OUTPUT_STEM" "${EVAL_DATA_ARGS[@]}"
    fi
  fi
fi

echo "Completed. Full log written to $RUN_LOG"
