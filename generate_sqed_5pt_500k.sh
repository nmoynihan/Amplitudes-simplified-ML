#!/usr/bin/env bash
set -euo pipefail

# Generate a new validated 5PT SQED candidate pool, then publish an exact
# 499,800-row train set and a target-disjoint 200-row held-out test set.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/Users/kymani/miniforge3/envs/scattering/bin/python}"
GENERATION_SEED="${GENERATION_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-5020005}"
JOBS="${JOBS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2000}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONHASHSEED
SQED_COVER_SEED_OFFSET="${SQED_COVER_SEED_OFFSET:-1000000007}"
HARD_SEED_OFFSET="${HARD_SEED_OFFSET:-2000000033}"

# Ten per cent candidate headroom is intentional.  gen_data.sh deduplicates
# within each profile stratum, while the release builder also removes the
# rarer duplicates that occur across strata.
CANDIDATE_ROWS="${CANDIDATE_ROWS:-550000}"
RELEASE_ROWS="${RELEASE_ROWS:-500000}"
BROAD_PERCENT="${BROAD_PERCENT:-60}"
SQED_COVER_PERCENT="${SQED_COVER_PERCENT:-30}"
HARD_PERCENT="${HARD_PERCENT:-10}"

# Preserve the requested 60/30/10 profile in both the candidate pool and the
# release.  Ten per cent source headroom is independently enforced per stratum.
BROAD_RELEASE_PERMILLE="${BROAD_RELEASE_PERMILLE:-600}"
SQED_COVER_RELEASE_PERMILLE="${SQED_COVER_RELEASE_PERMILLE:-300}"
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

STAGING_DIR="${STAGING_DIR:-data/sqed/sqed_5pt_500k_staging}"
OUTPUT_DIR="${OUTPUT_DIR:-data/sqed/sqed_5pt_500k}"
RAW_CANDIDATES="${RAW_CANDIDATES:-${STAGING_DIR}/sqed_5pt_oneshot_candidates_${CANDIDATE_ROWS}.csv}"
TOK_CANDIDATES="${TOK_CANDIDATES:-${STAGING_DIR}/sqed_5pt_oneshot_candidates_${CANDIDATE_ROWS}_tok.csv}"
GENERATION_LOG="${GENERATION_LOG:-${STAGING_DIR}/sqed_5pt_oneshot_candidates_${CANDIDATE_ROWS}.log}"
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
  if [[ ! -f "$GENERATION_LOG" ]]; then
    echo "Generation log is missing; refusing candidate reuse: $GENERATION_LOG" >&2
    exit 1
  fi
elif [[ -e "$RAW_CANDIDATES" || -e "$TOK_CANDIDATES" ]]; then
  echo "Only one candidate file exists; refusing an ambiguous resume." >&2
  echo "Remove or relocate the incomplete staging artifact, then rerun." >&2
  exit 1
else
  echo "Generating ${CANDIDATE_ROWS} validated 5PT SQED candidates..."
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    GEN_EXTRA_ARGS="" \
    N_PARTICLES=5 \
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
    OLD_STYLE_PROBABILITY=0.0 \
    SPURIOUS_REPEAT_PROBABILITY=0.35 \
    SCALAR_POWER_PROBABILITY=0.15 \
    MIXED_PROFILE=1 \
    BROAD_PERCENT="$BROAD_PERCENT" \
    SQED_COVER_PERCENT="$SQED_COVER_PERCENT" \
    HARD_PERCENT="$HARD_PERCENT" \
    SQED_COVER_MIN_TERMS=3 \
    SQED_COVER_MAX_TERMS=3 \
    SQED_COVER_MIN_SCR=1 \
    SQED_COVER_MAX_SCR=4 \
    SQED_COVER_SEED_OFFSET="$SQED_COVER_SEED_OFFSET" \
    HARD_SEED_OFFSET="$HARD_SEED_OFFSET" \
    SQED_COVER_OLD_STYLE_PROBABILITY=0.0 \
    SQED_COVER_UNIT_PROBABILITY=1.0 \
    SQED_COVER_SPURIOUS_REPEAT_PROBABILITY=0.0 \
    SQED_COVER_SCALAR_POWER_PROBABILITY=0.0 \
    SQED_COVER_SCRAMBLES="multiply_one ward momentum commute_dot ratio partial_fraction term_reorder" \
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

if [[ ! -f "$GENERATION_LOG" ]]; then
  echo "Generation log was not produced: $GENERATION_LOG" >&2
  exit 1
fi

RAW_ROWS=$(( $(wc -l < "$RAW_CANDIDATES") - 1 ))
TOK_ROWS=$(( $(wc -l < "$TOK_CANDIDATES") - 1 ))
if [[ "$RAW_ROWS" -ne "$CANDIDATE_ROWS" || "$TOK_ROWS" -ne "$CANDIDATE_ROWS" ]]; then
  echo "Candidate count mismatch: raw=$RAW_ROWS tokenized=$TOK_ROWS expected=$CANDIDATE_ROWS" >&2
  exit 1
fi

echo "Candidate pool complete; building the verified release..."
"$PYTHON_BIN" -m data_gen.split_sqed_5pt_release \
  --raw "$RAW_CANDIDATES" \
  --tokenised "$TOK_CANDIDATES" \
  --generation-log "$GENERATION_LOG" \
  --output-dir "$OUTPUT_DIR" \
  --generation-seed "$GENERATION_SEED" \
  --sqed-cover-seed-offset "$SQED_COVER_SEED_OFFSET" \
  --hard-seed-offset "$HARD_SEED_OFFSET" \
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

MANIFEST_PATH="${OUTPUT_DIR}/sqed_5pt_oneshot_${RELEASE_ROWS}_manifest.json"
VERIFICATION_REPORT_PATH="${OUTPUT_DIR}/sqed_5pt_oneshot_${RELEASE_ROWS}_verification.json"
echo "Running the independent full-release verifier..."
"$PYTHON_BIN" -m data_gen.verify_sqed_5pt_release \
  --manifest "$MANIFEST_PATH" \
  --candidate-raw "$RAW_CANDIDATES" \
  --candidate-tokenised "$TOK_CANDIDATES" \
  --release-dir "$OUTPUT_DIR" \
  --full-zero-audit \
  --report "$VERIFICATION_REPORT_PATH"

echo "Completed 5PT SQED dataset release in $OUTPUT_DIR"
