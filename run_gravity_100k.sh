#!/usr/bin/env bash
set -euo pipefail

# One balanced model:
#   25k 3s2h one-shot + 25k 3s2h staged
#   25k 4s1h one-shot + 25k 4s1h staged
#
# Usage: ./run_gravity_100k.sh [data|train|eval|full|smoke]

MODE="${1:-full}"
PYTHON="${PYTHON:-python3}"
SAMPLES="${SAMPLES:-100000}"
SEED="${SEED:-42}"
JOBS="${JOBS:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
# The data cap counts expression tokens, matching the existing generators.
# Training adds BOS/EOS, so retain two extra positions rather than truncating
# an otherwise valid 4096-token expression.
MODEL_MAX_LENGTH="$((MAX_TOKENS + 2))"
RUN_NAME="${RUN_NAME:-gravity_5pt_mixed_100k}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-12}"
BEAM_SIZE="${BEAM_SIZE:-8}"

RAW_OUT="${RAW_OUT:-data/gravity/gravity_5pt_100k_raw.csv.gz}"
TOK_OUT="${TOK_OUT:-data/gravity/gravity_5pt_100k_tok.csv.gz}"
META_OUT="${META_OUT:-data/gravity/gravity_5pt_100k_metadata.csv.gz}"
BENCH_RAW="${BENCH_RAW:-data/gravity/benchmarks_raw.csv.gz}"
BENCH_TOK="${BENCH_TOK:-data/gravity/benchmarks_tok.csv.gz}"
BENCH_META="${BENCH_META:-data/gravity/benchmarks_metadata.csv.gz}"
MODEL_PATH="${MODEL_PATH:-models/${RUN_NAME}/best_model.pt}"

# Helps the CUDA allocator reuse fragmented segments during variable-length
# batches. This is especially useful for the few 4k-token gravity examples.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

do_data=0
do_train=0
do_eval=0
case "$MODE" in
  data) do_data=1 ;;
  train) do_train=1 ;;
  eval) do_eval=1 ;;
  full) do_data=1; do_train=1; do_eval=1 ;;
  smoke)
    do_data=1
    do_train=1
    SAMPLES="${SMOKE_SAMPLES:-16}"
    JOBS=1
    EPOCHS=1
    BATCH_SIZE=4
    RUN_NAME="${SMOKE_RUN_NAME:-gravity_smoke}"
    MODEL_PATH="models/${RUN_NAME}/best_model.pt"
    RAW_OUT="data/gravity/smoke_raw.csv.gz"
    TOK_OUT="data/gravity/smoke_tok.csv.gz"
    META_OUT="data/gravity/smoke_metadata.csv.gz"
    ;;
  *)
    echo "Usage: $0 [data|train|eval|full|smoke]"
    exit 2
    ;;
esac

mkdir -p data/gravity

if [[ "$do_data" -eq 1 ]]; then
  "$PYTHON" -m data_gen.data_gen_gravity.generate \
    --samples "$SAMPLES" \
    --process mixed \
    --kind mixed \
    --seed "$SEED" \
    --min-scr 1 \
    --max-scr 5 \
    --min-terms 1 \
    --max-terms 3 \
    --max-tokens "$MAX_TOKENS" \
    --jobs "$JOBS" \
    --raw-out "$RAW_OUT" \
    --tok-out "$TOK_OUT" \
    --metadata-out "$META_OUT"

  if [[ "$MODE" != "smoke" ]]; then
    "$PYTHON" -m data_gen.data_gen_gravity.generate \
      --benchmarks \
      --benchmark-samples 100 \
      --seed 240804720 \
      --max-tokens "$MAX_TOKENS" \
      --jobs "$JOBS" \
      --raw-out "$BENCH_RAW" \
      --tok-out "$BENCH_TOK" \
      --metadata-out "$BENCH_META"
  fi
fi

if [[ "$do_train" -eq 1 ]]; then
  train_args=(
    --run-name "$RUN_NAME"
    --data-files "$(pwd)/$TOK_OUT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --max-length "$MODEL_MAX_LENGTH"
    --train-split 0.9
    --embedding-dim "${EMBEDDING_DIM:-512}"
    --n-heads "${N_HEADS:-8}"
    --n-enc-layers "${N_ENC_LAYERS:-6}"
    --n-dec-layers "${N_DEC_LAYERS:-6}"
    --head-ff-dim "${HEAD_FF_DIM:-2048}"
    --dropout "${DROPOUT:-0.1}"
    --learning-rate "${LEARNING_RATE:-3e-4}"
    --grad-clip "${GRAD_CLIP:-1.0}"
    --label-smoothing "${LABEL_SMOOTHING:-0.1}"
    --dynamic-padding
    --bucketing
    --bucket-size-multiplier "${BUCKET_SIZE_MULTIPLIER:-100}"
    --num-workers "${NUM_WORKERS:-2}"
    --amp-dtype "${AMP_DTYPE:-bf16}"
  )
  if [[ "$MODE" == "smoke" ]]; then
    train_args=(
      --run-name "$RUN_NAME"
      --data-files "$(pwd)/$TOK_OUT"
      --epochs 1
      --batch-size "$BATCH_SIZE"
      --gradient-accumulation-steps 1
      --max-length "$MODEL_MAX_LENGTH"
      --train-split 0.75
      --embedding-dim 32
      --n-heads 4
      --n-enc-layers 1
      --n-dec-layers 1
      --head-ff-dim 64
      --dropout 0.0
      --dynamic-padding
      --bucketing
      --num-workers 0
      --no-amp
      --no-pin-memory
    )
  else
    train_args+=(--amp --pin-memory)
  fi
  "$PYTHON" transformer/transformer_trainer.py "${train_args[@]}"
fi

if [[ "$do_eval" -eq 1 ]]; then
  "$PYTHON" -m data_gen.data_gen_gravity.evaluate \
    --model-path "$MODEL_PATH" \
    --raw "$BENCH_RAW" \
    --tokenized "$BENCH_TOK" \
    --metadata "$BENCH_META" \
    --max-output-tokens 512 \
    --decoding-method beam \
    --beam-size "$BEAM_SIZE" \
    --output data/gravity/benchmark_evaluation.csv.gz \
    --summary data/gravity/benchmark_evaluation_summary.json
fi
