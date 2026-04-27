#!/usr/bin/env bash
set -euo pipefail

# Stage 6: add mass-shell/momentum-conservation zero insertions.
export RUN_NAME="${RUN_NAME:-step_model_run6_mass_shell}"
export SCRAMBLES="${SCRAMBLES:-commute_dot multiply_one ward momentum ratio mass_shell_zero}"
export MIN_SCR="${MIN_SCR:-1}"
export MAX_SCR="${MAX_SCR:-4}"
export MIN_TERMS="${MIN_TERMS:-1}"
export MAX_TERMS="${MAX_TERMS:-2}"
export EVAL_MIN_SCR="${EVAL_MIN_SCR:-$MIN_SCR}"
export EVAL_MAX_SCR="${EVAL_MAX_SCR:-$MAX_SCR}"
export EVAL_MIN_TERMS="${EVAL_MIN_TERMS:-$MIN_TERMS}"
export EVAL_MAX_TERMS="${EVAL_MAX_TERMS:-$MAX_TERMS}"
export MAX_TOKENS="${MAX_TOKENS:-1536}"
export EVAL_KIND="${EVAL_KIND:-step}"
export EVAL_OUTPUT_STEM="${EVAL_OUTPUT_STEM:-step_pair_eval_4pt_run6_mass_shell}"
export RAW_OUT="${RAW_OUT:-data/gi_4pt_step_run6_mass_shell.csv}"
export TOK_OUT="${TOK_OUT:-data/gi_4pt_step_run6_mass_shell_tok.csv}"
export LOG_OUT="${LOG_OUT:-gen_data_4pt_step_run6_mass_shell.log}"

exec "$(dirname "$0")/run.sh" "${1:-train_eval}"
