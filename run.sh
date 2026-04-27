#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-${MODE:-train_eval}}"

N_PARTICLES="${N_PARTICLES:-4}"
SAMPLES="${SAMPLES:-5000}"
MIN_SCR="${MIN_SCR:-1}"
MAX_SCR="${MAX_SCR:-5}"
MIN_TERMS="${MIN_TERMS:-1}"
MAX_TERMS="${MAX_TERMS:-4}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TOKENIZER_MAX_PARTICLES="${TOKENIZER_MAX_PARTICLES:-8}"
SCRAMBLES="${SCRAMBLES:-}"

RUN_NAME="${RUN_NAME:-step_model}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-}"

EVAL_SAMPLES="${EVAL_SAMPLES:-64}"
EVAL_SEED="${EVAL_SEED:-123}"
EVAL_MIN_SCR="${EVAL_MIN_SCR:-2}"
EVAL_MAX_SCR="${EVAL_MAX_SCR:-5}"
EVAL_MIN_TERMS="${EVAL_MIN_TERMS:-1}"
EVAL_MAX_TERMS="${EVAL_MAX_TERMS:-2}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-8}"
EVAL_BEAM_SIZE="${EVAL_BEAM_SIZE:-8}"
EVAL_DECODING_METHOD="${EVAL_DECODING_METHOD:-beam}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_KIND="${EVAL_KIND:-iterative}"
EVAL_OUTPUT_STEM="${EVAL_OUTPUT_STEM:-iterative_eval_${N_PARTICLES}pt_${RUN_NAME}}"

RAW_OUT="${RAW_OUT:-data/gi_${N_PARTICLES}pt_step.csv}"
TOK_OUT="${TOK_OUT:-data/gi_${N_PARTICLES}pt_step_tok.csv}"
LOG_OUT="${LOG_OUT:-gen_data_${N_PARTICLES}pt_step.log}"
MODEL_PATH="${MODEL_PATH:-models/${RUN_NAME}/best_model.pt}"
RUN_LOG="${RUN_LOG:-logs/run_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$RAW_OUT")"
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

echo "Run log: $RUN_LOG"
echo "Mode: $MODE"
echo "N_PARTICLES=$N_PARTICLES RUN_NAME=$RUN_NAME MODEL_PATH=$MODEL_PATH"
echo "SCRAMBLES=${SCRAMBLES:-all}"

SCRAMBLE_ARGS=()
if [[ -n "$SCRAMBLES" ]]; then
  read -r -a SCRAMBLE_NAMES <<< "$SCRAMBLES"
  SCRAMBLE_ARGS=(--scrambles "${SCRAMBLE_NAMES[@]}")
fi

if [[ "$DO_TRAIN" -eq 1 ]]; then
  python3 data_gen/gen_data.py "$N_PARTICLES" \
    --dataset-kind step \
    --samples "$SAMPLES" \
    --min-scr "$MIN_SCR" \
    --max-scr "$MAX_SCR" \
    --min-terms "$MIN_TERMS" \
    --max-terms "$MAX_TERMS" \
    --seed "$SEED" \
    --max-tokens "$MAX_TOKENS" \
    --tokenizer-max-particles "$TOKENIZER_MAX_PARTICLES" \
    --raw-out "$RAW_OUT" \
    --tok-out "$TOK_OUT" \
    --log-out "$LOG_OUT" \
    "${SCRAMBLE_ARGS[@]}"

  if [[ "$TOK_OUT" = /* ]]; then
    TRAIN_TOK_OUT="$TOK_OUT"
  else
    TRAIN_TOK_OUT="$(pwd)/$TOK_OUT"
  fi

  TRAIN_ARGS=(
    --run_name "$RUN_NAME"
    --data-files "$TRAIN_TOK_OUT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
  )

  if [[ -n "$TRAIN_MAX_LENGTH" ]]; then
    TRAIN_ARGS+=(--max-length "$TRAIN_MAX_LENGTH")
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
    python3 data_testing/evaluate_iterative_simplification.py \
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
      --output-stem "$EVAL_OUTPUT_STEM" \
      "${SCRAMBLE_ARGS[@]}"
  fi
fi

echo "Completed. Full log written to $RUN_LOG"
