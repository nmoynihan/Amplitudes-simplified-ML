#!/usr/bin/env bash
set -euo pipefail

# Stage 2: add only dot-product commutation.
export RUN_NAME="${RUN_NAME:-step_model_run2_commute}"
export SCRAMBLES="${SCRAMBLES:-commute_dot}"
export MIN_SCR="${MIN_SCR:-1}"
export MAX_SCR="${MAX_SCR:-2}"
export MIN_TERMS="${MIN_TERMS:-1}"
export MAX_TERMS="${MAX_TERMS:-1}"
export EVAL_MIN_SCR="${EVAL_MIN_SCR:-$MIN_SCR}"
export EVAL_MAX_SCR="${EVAL_MAX_SCR:-$MAX_SCR}"
export EVAL_MIN_TERMS="${EVAL_MIN_TERMS:-$MIN_TERMS}"
export EVAL_MAX_TERMS="${EVAL_MAX_TERMS:-$MAX_TERMS}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export EVAL_KIND="${EVAL_KIND:-step}"
export EVAL_OUTPUT_STEM="${EVAL_OUTPUT_STEM:-step_pair_eval_4pt_run2_commute}"
export RAW_OUT="${RAW_OUT:-data/gi_4pt_step_run2_commute.csv}"
export TOK_OUT="${TOK_OUT:-data/gi_4pt_step_run2_commute_tok.csv}"
export LOG_OUT="${LOG_OUT:-gen_data_4pt_step_run2_commute.log}"

exec "$(dirname "$0")/run.sh" "${1:-train_eval}"
