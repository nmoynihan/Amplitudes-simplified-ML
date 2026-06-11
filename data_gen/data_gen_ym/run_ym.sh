#!/usr/bin/env bash
# run_ym.sh — generate colour-ordered all-gluon (simple, scrambled) data.
#
# Usage:
#   ./run_ym.sh [N] [extra generate flags...]
# Examples:
#   ./run_ym.sh 5 --jobs 2
#   SAMPLES=100 SEED=7 ./run_ym.sh 4 --no-tokenise
#   PYTHON=python3 ./run_ym.sh 5
#
# Resolves its own location and runs from data_gen/ so the `-m data_gen_ym.generate`
# package import and the (parent-dir) Tokenizer import both resolve, regardless of
# the caller's working directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../data_gen/data_gen_ym
DATA_GEN="$(dirname "$HERE")"                           # .../data_gen
PYTHON="${PYTHON:-$DATA_GEN/../venv/bin/python}"        # repo venv (override via $PYTHON)

N="${1:-5}"
shift || true

# Pin BLAS to 1 thread per worker. generate_kinematics' numpy ops are tiny 4-vector
# operations, so multi-threaded BLAS gives no benefit and, with N worker processes,
# causes thread oversubscription (N procs x ~8 BLAS threads fighting over the cores).
# 1 thread/worker keeps it 1:1 and is markedly faster. (Override by exporting these
# before calling, if you ever want otherwise.)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$DATA_GEN"
exec "$PYTHON" -m data_gen_ym.generate "$N" \
    --samples "${SAMPLES:-1000}" \
    --seed "${SEED:-42}" \
    "$@"
