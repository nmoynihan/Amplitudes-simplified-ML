#!/usr/bin/env bash
set -euo pipefail

# Generate a new validated 4PT SQED candidate pool, then publish an exact
# 499,800-row train set and a target-disjoint 200-row held-out test set.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/Users/kymani/miniforge3/envs/scattering/bin/python}"
GENERATION_SEED="${GENERATION_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-4020004}"
JOBS="${JOBS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2000}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONHASHSEED

# Ten per cent candidate headroom is intentional.  gen_data.sh deduplicates
# within each profile stratum, while the release builder also removes the
# rarer duplicates that occur across strata.
CANDIDATE_ROWS="${CANDIDATE_ROWS:-550000}"
RELEASE_ROWS="${RELEASE_ROWS:-500000}"
BROAD_PERCENT="${BROAD_PERCENT:-60}"
SQED_COVER_PERCENT="${SQED_COVER_PERCENT:-30}"
HARD_PERCENT="${HARD_PERCENT:-10}"

# Cross-profile input equivalences make the cover stratum slightly less unique
# than its raw row count suggests.  The 60.4/29.6/10.0 release mix is the
# closest stable allocation found by the full 550k source audit while keeping
# the held-out test composition at 60/30/10.
BROAD_RELEASE_PERMILLE="${BROAD_RELEASE_PERMILLE:-604}"
SQED_COVER_RELEASE_PERMILLE="${SQED_COVER_RELEASE_PERMILLE:-296}"
HARD_RELEASE_PERMILLE="${HARD_RELEASE_PERMILLE:-100}"

if ((BROAD_PERCENT + SQED_COVER_PERCENT + HARD_PERCENT != 100)); then
  echo "BROAD_PERCENT + SQED_COVER_PERCENT + HARD_PERCENT must equal 100." >&2
  exit 1
fi
if ((BROAD_RELEASE_PERMILLE + SQED_COVER_RELEASE_PERMILLE + HARD_RELEASE_PERMILLE != 1000)); then
  echo "Release permille values must sum to 1000." >&2
  exit 1
fi

BROAD_CANDIDATE_ROWS=$((CANDIDATE_ROWS * BROAD_PERCENT / 100))
SQED_COVER_CANDIDATE_ROWS=$((CANDIDATE_ROWS * SQED_COVER_PERCENT / 100))
HARD_CANDIDATE_ROWS=$((CANDIDATE_ROWS - BROAD_CANDIDATE_ROWS - SQED_COVER_CANDIDATE_ROWS))
BROAD_RELEASE_ROWS=$((RELEASE_ROWS * BROAD_RELEASE_PERMILLE / 1000))
SQED_COVER_RELEASE_ROWS=$((RELEASE_ROWS * SQED_COVER_RELEASE_PERMILLE / 1000))
HARD_RELEASE_ROWS=$((RELEASE_ROWS - BROAD_RELEASE_ROWS - SQED_COVER_RELEASE_ROWS))

BROAD_TEST_ROWS="${BROAD_TEST_ROWS:-120}"
SQED_COVER_TEST_ROWS="${SQED_COVER_TEST_ROWS:-60}"
HARD_TEST_ROWS="${HARD_TEST_ROWS:-20}"

STAGING_DIR="${STAGING_DIR:-data/sqed/sqed_4pt_500k_staging}"
OUTPUT_DIR="${OUTPUT_DIR:-data/sqed/sqed_4pt_500k}"
RAW_CANDIDATES="${RAW_CANDIDATES:-${STAGING_DIR}/sqed_4pt_oneshot_candidates_${CANDIDATE_ROWS}.csv}"
TOK_CANDIDATES="${TOK_CANDIDATES:-${STAGING_DIR}/sqed_4pt_oneshot_candidates_${CANDIDATE_ROWS}_tok.csv}"
GENERATION_LOG="${GENERATION_LOG:-${STAGING_DIR}/sqed_4pt_oneshot_candidates_${CANDIDATE_ROWS}.log}"
REUSE_CANDIDATES="${REUSE_CANDIDATES:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import numpy
import sympy
from data_gen.Tokenizer import ScatteringAmplitudeTokenizer

tokenizer = ScatteringAmplitudeTokenizer(max_particles=8)
if tokenizer.vocab_size != 57:
    raise SystemExit(f"unexpected tokenizer vocabulary size: {tokenizer.vocab_size}")
print(f"Using numpy {numpy.__version__}, sympy {sympy.__version__}")
PY

if [[ -f "$RAW_CANDIDATES" && -f "$TOK_CANDIDATES" ]]; then
  if [[ "$REUSE_CANDIDATES" != "1" ]]; then
    echo "Candidate files already exist; refusing implicit reuse." >&2
    echo "Set REUSE_CANDIDATES=1 only after verifying their generation settings." >&2
    exit 1
  fi
  echo "Reusing completed candidate files:"
  echo "  $RAW_CANDIDATES"
  echo "  $TOK_CANDIDATES"
elif [[ -e "$RAW_CANDIDATES" || -e "$TOK_CANDIDATES" ]]; then
  echo "Only one candidate file exists; refusing an ambiguous resume." >&2
  echo "Remove or relocate the incomplete staging artifact, then rerun." >&2
  exit 1
else
  echo "Generating ${CANDIDATE_ROWS} validated 4PT SQED candidates..."
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    GEN_EXTRA_ARGS="" \
    N_PARTICLES=4 \
    SAMPLES="$CANDIDATE_ROWS" \
    DATASET_KIND=oneshot \
    SEED="$GENERATION_SEED" \
    MIN_SCR=1 \
    MAX_SCR=4 \
    MIN_TERMS=1 \
    MAX_TERMS=3 \
    MAX_TOKENS="$MAX_TOKENS" \
    TOKENIZER_MAX_PARTICLES=8 \
    JOBS="$JOBS" \
    BATCH_SIZE="$BATCH_SIZE" \
    UNIT_PROBABILITY=0.9 \
    OLD_STYLE_PROBABILITY=0.4 \
    SPURIOUS_REPEAT_PROBABILITY=0.35 \
    SCALAR_POWER_PROBABILITY=0.15 \
    MIXED_PROFILE=1 \
    BROAD_PERCENT="$BROAD_PERCENT" \
    SQED_COVER_PERCENT="$SQED_COVER_PERCENT" \
    HARD_PERCENT="$HARD_PERCENT" \
    SQED_COVER_OLD_STYLE_PROBABILITY=0.65 \
    SQED_COVER_UNIT_PROBABILITY=1.0 \
    SQED_COVER_SPURIOUS_REPEAT_PROBABILITY=0.15 \
    SQED_COVER_SCALAR_POWER_PROBABILITY=0.1 \
    SQED_COVER_SCRAMBLES="multiply_one ward momentum commute_dot ratio" \
    BROAD_SCRAMBLES=all \
    HARD_SCRAMBLES=all \
    HARD_MIN_TERMS=3 \
    HARD_MAX_TERMS=3 \
    HARD_MIN_SCR=3 \
    HARD_MAX_SCR=4 \
    HARD_OLD_STYLE_PROBABILITY=0.0 \
    NO_VALIDATE=0 \
    NO_TOKENISE=0 \
    GROUPED_SCRAMBLED=0 \
    RAW_OUT="$RAW_CANDIDATES" \
    TOK_OUT="$TOK_CANDIDATES" \
    LOG_OUT="$GENERATION_LOG" \
    ./gen_data.sh
fi

RAW_ROWS=$(( $(wc -l < "$RAW_CANDIDATES") - 1 ))
TOK_ROWS=$(( $(wc -l < "$TOK_CANDIDATES") - 1 ))
if [[ "$RAW_ROWS" -ne "$CANDIDATE_ROWS" || "$TOK_ROWS" -ne "$CANDIDATE_ROWS" ]]; then
  echo "Candidate count mismatch: raw=$RAW_ROWS tokenized=$TOK_ROWS expected=$CANDIDATE_ROWS" >&2
  exit 1
fi

echo "Candidate pool complete; building the verified release..."
"$PYTHON_BIN" -m data_gen.split_sqed_4pt_release \
  --raw "$RAW_CANDIDATES" \
  --tokenised "$TOK_CANDIDATES" \
  --generation-log "$GENERATION_LOG" \
  --output-dir "$OUTPUT_DIR" \
  --generation-seed "$GENERATION_SEED" \
  --split-seed "$SPLIT_SEED" \
  --python-hash-seed "$PYTHONHASHSEED" \
  --jobs "$JOBS" \
  --batch-size "$BATCH_SIZE" \
  --max-tokens "$MAX_TOKENS" \
  --tokenizer-max-particles 8 \
  --broad-candidate-rows "$BROAD_CANDIDATE_ROWS" \
  --sqed-cover-candidate-rows "$SQED_COVER_CANDIDATE_ROWS" \
  --hard-candidate-rows "$HARD_CANDIDATE_ROWS" \
  --broad-release-rows "$BROAD_RELEASE_ROWS" \
  --sqed-cover-release-rows "$SQED_COVER_RELEASE_ROWS" \
  --hard-release-rows "$HARD_RELEASE_ROWS" \
  --broad-test-rows "$BROAD_TEST_ROWS" \
  --sqed-cover-test-rows "$SQED_COVER_TEST_ROWS" \
  --hard-test-rows "$HARD_TEST_ROWS"

echo "Completed 4PT SQED dataset release in $OUTPUT_DIR"
