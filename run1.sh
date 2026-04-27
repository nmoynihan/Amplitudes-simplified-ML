#!/usr/bin/env bash
set -euo pipefail

# Stage 1: expanded -> simple only. No extra scrambles.
export RUN_NAME="${RUN_NAME:-step_model_run1_expand}"
export SCRAMBLES="${SCRAMBLES:-none}"
export MIN_SCR="${MIN_SCR:-0}"
export MAX_SCR="${MAX_SCR:-0}"
export MIN_TERMS="${MIN_TERMS:-1}"
export MAX_TERMS="${MAX_TERMS:-1}"
export EVAL_MIN_SCR="${EVAL_MIN_SCR:-$MIN_SCR}"
export EVAL_MAX_SCR="${EVAL_MAX_SCR:-$MAX_SCR}"
export EVAL_MIN_TERMS="${EVAL_MIN_TERMS:-$MIN_TERMS}"
export EVAL_MAX_TERMS="${EVAL_MAX_TERMS:-$MAX_TERMS}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export EVAL_KIND="${EVAL_KIND:-step}"
export EVAL_OUTPUT_STEM="${EVAL_OUTPUT_STEM:-step_pair_eval_4pt_run1_expand}"
export RAW_OUT="${RAW_OUT:-data/gi_4pt_step_run1_expand.csv}"
export TOK_OUT="${TOK_OUT:-data/gi_4pt_step_run1_expand_tok.csv}"
export LOG_OUT="${LOG_OUT:-gen_data_4pt_step_run1_expand.log}"

exec "$(dirname "$0")/run.sh" "${1:-train_eval}"
