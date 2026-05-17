#!/usr/bin/env python3
"""
Evaluate one trained transformer on freshly generated amplitude data.

All configuration lives in this file. The script:

1. Generates new raw data on disk, or imports an existing raw CSV.
2. Tokenises that data on disk.
3. Loads one model checkpoint.
4. Runs one or more decoding modes (greedy by default, optional beam/nucleus).
5. Reports exact-match and numerical-equivalence metrics against both the
   target simple expressions and the input scrambled expressions.

Outputs are written under DATA_TESTING_DIR / OUTPUT_SUBDIR.
"""
from __future__ import annotations

import csv
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_TESTING_DIR = ROOT / "data_testing"
DATA_STORAGE_DIR = ROOT / "data"
OUTPUT_SUBDIR = "outputs"

#MODEL_PATH = ROOT / "models" / "best_model.pt"
# MODEL_PATH = ROOT / "models" / "fivek_newtest" / "best_model.pt"
MODEL_PATH = ROOT / "models" / "unit_500k" / "best_model.pt"
# "auto" chooses CUDA when available, otherwise CPU. Set explicitly for repeatable timing/debugging.
DEVICE = "auto"  # "auto", "cpu", "cuda"

# Generated evaluation set shape. N_PARTICLES=4 means two photon legs in this setup.
N_PARTICLES = 4
NUM_SAMPLES = 100
# Seed controls the generated evaluation examples, so changing it changes the test set.
GENERATION_SEED = 451
# Tokenizer vocabulary supports p_i/e_i/F_i/M_i up to this particle index.
TOKENIZER_MAX_PARTICLES = 8

# All generated raw/token/result CSVs use this stem under data_testing/outputs.
DATA_FILENAME_STEM = "generated_eval_4pt"
RAW_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}.csv"
TOK_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_tok.csv"
GEN_LOG_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}.log"
SUMMARY_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_summary.csv"
# Data source for this evaluation. Use "generate" for fresh synthetic data or
# "csv" for an existing raw CSV with columns named simple and scrambled.
DATA_SOURCE = "csv"  # "generate", "csv"
# Used only when DATA_SOURCE="csv". Relative paths are resolved from repo root.
# EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "sqed_w0110_mM00M_den6_maxD2_20000_phys_oneshot.csv" # Paolo's 20k dataset
# EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "gi_4pt_oneshot_5k_val.csv" # Nathan's 5k dataset
# EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "messified.csv" # Paolo's 20k dataset
# EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "sqed_oneshot_150.csv" # Paolo's 20k dataset
EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "sqed_4ptseed_oneshot.csv" # 100 scrambled amplitudes


# Optional row cap for imported CSVs. None evaluates every row in the file.
EXISTING_CSV_MAX_ROWS = 100 

# Optional exact-pair dedupe for imported CSVs before tokenization/evaluation.
EXISTING_CSV_DEDUPE = True

### Special dataset testing. Comment it all out to use one of the above real datasets instead.
#DATA_SOURCE = "csv"
#EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "sqed_w0110_mM00M_den6_maxD2_20000_phys_oneshot_gi_style_no_powers.csv"
#EXISTING_CSV_MAX_ROWS = 1000

# Number of additive terms in the canonical/simple expression before expansion/scrambling.
GEN_MIN_TERMS = 1
GEN_MAX_TERMS = 5
# Number of scramble passes applied to the expanded expression.
GEN_MIN_SCRAMBLES = 3
GEN_MAX_SCRAMBLES = 6
# Validate generated simple/expanded/scrambled expressions numerically before accepting rows.
GEN_VALIDATE = True
# Mass value used during generation-time validation.
GEN_MASS = 2.0

# Evaluation batching. Larger is faster if the selected device has enough memory.
BATCH_SIZE = 8
# None means pad/decode up to this generated dataset's longest source/target sequence.
MAX_SEQ_LENGTH_OVERRIDE = None
# Number of example rows printed per decode mode; full details are always written to CSV.
PRINT_EXAMPLES = 3
# When True, print a concise one-block summary instead of the verbose per-metric list.
SIMPLE_SUMMARY = True
# When True, write a compact human-readable CSV alongside the full detail CSV.
HUMAN_CSV = True
# When True, generate diagnostic plots alongside the evaluation outputs.
PLOTS = True

# Numeric equivalence check for model predictions. These samples are independent of generation.
NUMERIC_EQUIV_SAMPLES = 3
NUMERIC_EQUIV_SEED = 151
NUMERIC_EQUIV_MASS = 2.0
# Equivalence passes if either absolute or relative tolerance is met on every numeric sample.
NUMERIC_TOL_ABS = 1e-12
NUMERIC_TOL_REL = 1e-10
BEAM_SIZE = 50


@dataclass(frozen=True)
class DecodeConfig:
    name: str
    # Toggle individual modes without changing the rest of the evaluation setup.
    enabled: bool
    # Supported by decode_with_model: "greedy", "beam", or "nucleus".
    decoding_method: str
    # Per-mode override. None falls back to MAX_SEQ_LENGTH_OVERRIDE or dataset.max_length.
    max_length: int | None = None
    # Number of retained hypotheses for beam search and nucleus sampling.
    beam_size: int = BEAM_SIZE
    # Nucleus sampling cutoff and temperature; ignored by greedy/beam.
    p_nucleus: float = 0.95
    temperature_nucleus: float = 1.0
    # When true, score every returned hypothesis, not just the top-1 output.
    evaluate_beam_hypotheses: bool = False
    # Optional cap on checked hypotheses; None checks all returned hypotheses.
    max_beams_to_check: int | None = None
    # If true, choose the shortest decoded beam that is numerically equivalent
    # to the scrambled input as the reported top-1 prediction.
    rerank_numerical_equiv: bool = True


# Each enabled entry below produces its own detail CSV and one row in the summary CSV.
DECODE_RUNS: list[DecodeConfig] = [
    DecodeConfig(
        name="greedy",
        enabled=True,
        decoding_method="greedy",
        evaluate_beam_hypotheses=False,
    ),
    DecodeConfig(
        name="beam",
        enabled=True,
        decoding_method="beam",
        beam_size=BEAM_SIZE,
        evaluate_beam_hypotheses=True,
        max_beams_to_check=None,
    ),
    DecodeConfig(
        name="nucleus",
        enabled=True,
        decoding_method="nucleus",
        beam_size=BEAM_SIZE,
        p_nucleus=0.99, # Higher p means more tokens are considered at each step; 0.9 is a common default but may be too low for this task.
        temperature_nucleus=1.2, # Higher temperature means more random samples; 1.0 is the default and means no reweighting.
        evaluate_beam_hypotheses=True,
        max_beams_to_check=None,
    ),
]

CLI_SCRAMBLES: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained transformer on generated amplitude data.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-scr", type=int, default=None)
    parser.add_argument("--max-scr", type=int, default=None)
    parser.add_argument("--min-terms", type=int, default=None)
    parser.add_argument("--max-terms", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tokenizer-max-particles", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Accepted for compatibility with iterative evaluators; unused by this one-shot evaluator.",
    )
    parser.add_argument(
        "--decoding-method",
        choices=["greedy", "beam", "nucleus"],
        default=None,
    )
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--p-nucleus", type=float, default=None)
    parser.add_argument("--temperature-nucleus", type=float, default=None)
    parser.add_argument(
        "--rerank-numerical",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Rerank beam candidates by numerical equivalence to the scrambled "
            "input, selecting the shortest equivalent decoded candidate as top-1."
        ),
    )
    parser.add_argument("--output-stem", type=str, default=None)
    parser.add_argument(
        "--scrambles",
        nargs="*",
        default=None,
        choices=[
            "all",
            "none",
            "multiply_one",
            "ward",
            "momentum",
            "commute_dot",
            "ratio",
            "mass_shell_zero",
            "partial_fraction",
        ],
        help="Scramble labels to use when generating fresh evaluation data.",
    )
    parser.add_argument(
        "--data-source",
        choices=["generate", "csv"],
        default=None,
        help="Use generated synthetic data or import an existing raw CSV.",
    )
    parser.add_argument("--existing-raw-csv", type=str, default=None)
    parser.add_argument("--existing-csv-max-rows", type=int, default=None)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--simple-summary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print a concise summary block instead of the verbose per-metric list.",
    )
    parser.add_argument(
        "--human-csv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write a compact human-readable CSV with simple, scrambled, top_pred, correct columns.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate diagnostic plots alongside the evaluation outputs.",
    )
    return parser.parse_args()


def apply_cli_config(args: argparse.Namespace) -> None:
    global MODEL_PATH, DEVICE, N_PARTICLES, NUM_SAMPLES, GENERATION_SEED
    global TOKENIZER_MAX_PARTICLES, DATA_FILENAME_STEM, RAW_CSV_PATH, TOK_CSV_PATH
    global GEN_LOG_PATH, SUMMARY_CSV_PATH, DATA_SOURCE, EXISTING_RAW_CSV_PATH
    global EXISTING_CSV_MAX_ROWS, EXISTING_CSV_DEDUPE, GEN_MIN_TERMS, GEN_MAX_TERMS
    global GEN_MIN_SCRAMBLES, GEN_MAX_SCRAMBLES, BATCH_SIZE, MAX_SEQ_LENGTH_OVERRIDE
    global DECODE_RUNS, CLI_SCRAMBLES, SIMPLE_SUMMARY, HUMAN_CSV, PLOTS

    if args.model_path is not None:
        MODEL_PATH = resolve_input_path(args.model_path)
    if args.device is not None:
        DEVICE = args.device
    if args.n_particles is not None:
        N_PARTICLES = args.n_particles
    if args.num_samples is not None:
        NUM_SAMPLES = args.num_samples
    if args.seed is not None:
        GENERATION_SEED = args.seed
    if args.tokenizer_max_particles is not None:
        TOKENIZER_MAX_PARTICLES = args.tokenizer_max_particles
    if args.min_terms is not None:
        GEN_MIN_TERMS = args.min_terms
    if args.max_terms is not None:
        GEN_MAX_TERMS = args.max_terms
    if args.min_scr is not None:
        GEN_MIN_SCRAMBLES = args.min_scr
    if args.max_scr is not None:
        GEN_MAX_SCRAMBLES = args.max_scr
    if args.max_tokens is not None:
        MAX_SEQ_LENGTH_OVERRIDE = None if args.max_tokens <= 0 else args.max_tokens
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.existing_raw_csv is not None:
        EXISTING_RAW_CSV_PATH = resolve_input_path(args.existing_raw_csv)
    if args.existing_csv_max_rows is not None:
        EXISTING_CSV_MAX_ROWS = args.existing_csv_max_rows
    if args.no_dedupe:
        EXISTING_CSV_DEDUPE = False

    if args.data_source is not None:
        DATA_SOURCE = args.data_source
    elif any(
        value is not None
        for value in (
            args.num_samples,
            args.seed,
            args.min_scr,
            args.max_scr,
            args.min_terms,
            args.max_terms,
            args.scrambles,
        )
    ):
        DATA_SOURCE = "generate"

    if args.output_stem is not None:
        DATA_FILENAME_STEM = args.output_stem
        RAW_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}.csv"
        TOK_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_tok.csv"
        GEN_LOG_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}.log"
        SUMMARY_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_summary.csv"

    if args.decoding_method is not None:
        beam_size = args.beam_size if args.beam_size is not None else 1
        if args.decoding_method == "greedy":
            beam_size = 1
        default_cfg = next(
            (cfg for cfg in DECODE_RUNS if cfg.decoding_method == args.decoding_method),
            DECODE_RUNS[0],
        )
        DECODE_RUNS = [
            DecodeConfig(
                name=args.decoding_method,
                enabled=True,
                decoding_method=args.decoding_method,
                beam_size=beam_size,
                p_nucleus=args.p_nucleus if args.p_nucleus is not None else default_cfg.p_nucleus,
                temperature_nucleus=(
                    args.temperature_nucleus
                    if args.temperature_nucleus is not None
                    else default_cfg.temperature_nucleus
                ),
                evaluate_beam_hypotheses=args.decoding_method in {"beam", "nucleus"},
                max_beams_to_check=None,
                rerank_numerical_equiv=bool(args.rerank_numerical),
            )
        ]
    elif (
        args.beam_size is not None
        or args.rerank_numerical is not None
        or args.p_nucleus is not None
        or args.temperature_nucleus is not None
    ):
        DECODE_RUNS = [
            DecodeConfig(
                name=cfg.name,
                enabled=cfg.enabled,
                decoding_method=cfg.decoding_method,
                max_length=cfg.max_length,
                beam_size=args.beam_size if cfg.decoding_method in {"beam", "nucleus"} else cfg.beam_size,
                p_nucleus=args.p_nucleus if args.p_nucleus is not None else cfg.p_nucleus,
                temperature_nucleus=(
                    args.temperature_nucleus
                    if args.temperature_nucleus is not None
                    else cfg.temperature_nucleus
                ),
                evaluate_beam_hypotheses=cfg.evaluate_beam_hypotheses,
                max_beams_to_check=cfg.max_beams_to_check,
                rerank_numerical_equiv=(
                    bool(args.rerank_numerical)
                    if args.rerank_numerical is not None
                    else cfg.rerank_numerical_equiv
                ),
            )
            for cfg in DECODE_RUNS
        ]

    if args.max_steps is not None:
        print("--max-steps is ignored by evaluate_model_on_generated_data.py; this evaluator is one-shot.")

    if args.simple_summary is not None:
        SIMPLE_SUMMARY = args.simple_summary
    if args.human_csv is not None:
        HUMAN_CSV = args.human_csv
    if args.plots is not None:
        PLOTS = args.plots

    CLI_SCRAMBLES = args.scrambles


# ============================================================================
# Imports from repo modules
# ============================================================================

sys.path.insert(0, str(ROOT / "data_gen"))
sys.path.insert(0, str(ROOT / "transformer"))

import gen_data as gd
from Tokenizer import ScatteringAmplitudeTokenizer
from data_import import TransformerDataset
from kinematics import generate_kinematics
from transformer_functions import (
    TransformerRegressor,
    clean_seq,
    decode_with_model,
    load_transformer_model,
)


# ============================================================================
# Helpers
# ============================================================================

SPECIAL_TOKENS = {"pad": 0, "bos": 2, "eos": 3}


def resolve_device() -> str:
    if DEVICE == "cpu":
        return "cpu"
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE='cuda' but CUDA is not available")
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_output_dirs() -> None:
    (DATA_TESTING_DIR / OUTPUT_SUBDIR).mkdir(parents=True, exist_ok=True)
    for path in (RAW_CSV_PATH, TOK_CSV_PATH, GEN_LOG_PATH, SUMMARY_CSV_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)


def strip_special_tokens(seq: list[int], *, keep_eos: bool = False) -> list[int]:
    cleaned = clean_seq(seq, pad_token=SPECIAL_TOKENS["pad"], eos_token=SPECIAL_TOKENS["eos"])
    if cleaned and cleaned[0] == SPECIAL_TOKENS["bos"]:
        cleaned = cleaned[1:]
    if not keep_eos and cleaned and cleaned[-1] == SPECIAL_TOKENS["eos"]:
        cleaned = cleaned[:-1]
    return cleaned


def safe_decode_infix(tokenizer: ScatteringAmplitudeTokenizer, seq: list[int]) -> tuple[bool, str, str | None]:
    if not seq:
        return False, "", "empty prediction"
    try:
        return True, tokenizer.decode_infix(seq), None
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def load_raw_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_input_path(path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def read_raw_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"simple", "scrambled"}
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        pairs: list[tuple[str, str]] = []
        for row_idx, row in enumerate(reader, start=2):
            simple = (row.get("simple") or "").strip()
            scrambled = (row.get("scrambled") or "").strip()
            if not simple or not scrambled:
                raise ValueError(f"{path}:{row_idx} has an empty simple or scrambled value")
            pairs.append((simple, scrambled))

    if not pairs:
        raise ValueError(f"{path} contains no data rows")
    return pairs


def load_token_rows(path: Path) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "simple": json.loads(row["simple"]),
                    "scrambled": json.loads(row["scrambled"]),
                }
            )
    return rows


def precompute_kinematics() -> list[tuple[Any, Any]]:
    return [
        generate_kinematics(
            N_PARTICLES,
            M=NUMERIC_EQUIV_MASS,
            seed=NUMERIC_EQUIV_SEED + i,
        )
        for i in range(NUMERIC_EQUIV_SAMPLES)
    ]


def numerically_equivalent_exprs(
    expr_a: str,
    expr_b: str,
    cached_kinematics: list[tuple[Any, Any]],
) -> bool:
    try:
        for momenta, pols in cached_kinematics:
            val_a = gd.eval_infix_numeric(expr_a, momenta, pols)
            val_b = gd.eval_infix_numeric(expr_b, momenta, pols)
            if not (math.isfinite(val_a) and math.isfinite(val_b)):
                return False
            diff = abs(val_a - val_b)
            scale = max(abs(val_a), abs(val_b), 1.0)
            if not (diff <= NUMERIC_TOL_ABS or diff / scale <= NUMERIC_TOL_REL):
                return False
        return True
    except Exception:
        return False


def unique_sequences(seqs: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for seq in seqs:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            out.append(seq)
    return out


def generate_test_data() -> None:
    print(f"Generating {NUM_SAMPLES} fresh {N_PARTICLES}-point samples...")
    pairs = gd.build_dataset(
        N_PARTICLES,
        NUM_SAMPLES,
        max_scr=GEN_MAX_SCRAMBLES,
        min_scr=GEN_MIN_SCRAMBLES,
        seed=GENERATION_SEED,
        use_denominators=True,
        validate=GEN_VALIDATE,
        M=GEN_MASS,
        min_terms=GEN_MIN_TERMS,
        max_terms=GEN_MAX_TERMS,
        log_path=str(GEN_LOG_PATH),
        scramble_names=CLI_SCRAMBLES,
        max_tokens=MAX_SEQ_LENGTH_OVERRIDE,
        tokenizer_max_particles=TOKENIZER_MAX_PARTICLES,
    )
    pairs, removed = gd.dedupe_pairs(pairs)
    pairs = pairs[:NUM_SAMPLES]
    gd.write_csv(pairs, str(RAW_CSV_PATH))
    gd.tokenise_csv(
        str(RAW_CSV_PATH),
        str(TOK_CSV_PATH),
        max_particles=TOKENIZER_MAX_PARTICLES,
        max_sequence_length=MAX_SEQ_LENGTH_OVERRIDE,
    )
    print(f"Wrote raw data to {RAW_CSV_PATH}")
    print(f"Wrote tokenised data to {TOK_CSV_PATH}")
    print(f"Dedupe removed {removed} duplicate pairs")


def import_existing_test_data() -> None:
    source_path = resolve_input_path(EXISTING_RAW_CSV_PATH)
    print(f"Using existing raw data from {source_path}")
    pairs = read_raw_pairs(source_path)
    original_count = len(pairs)

    removed = 0
    if EXISTING_CSV_DEDUPE:
        pairs, removed = gd.dedupe_pairs(pairs)
    if EXISTING_CSV_MAX_ROWS is not None:
        pairs = pairs[: int(EXISTING_CSV_MAX_ROWS)]

    gd.write_csv(pairs, str(RAW_CSV_PATH))
    gd.tokenise_csv(
        str(RAW_CSV_PATH),
        str(TOK_CSV_PATH),
        max_particles=TOKENIZER_MAX_PARTICLES,
        max_sequence_length=MAX_SEQ_LENGTH_OVERRIDE,
    )

    with GEN_LOG_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"# imported raw CSV: {source_path}\n")
        handle.write(f"# rows_read={original_count} rows_used={len(pairs)} dedupe_removed={removed}\n")

    print(f"Imported {len(pairs)} / {original_count} rows")
    print(f"Wrote raw data to {RAW_CSV_PATH}")
    print(f"Wrote tokenised data to {TOK_CSV_PATH}")
    if EXISTING_CSV_DEDUPE:
        print(f"Dedupe removed {removed} duplicate pairs")


def prepare_test_data() -> None:
    if DATA_SOURCE == "generate":
        generate_test_data()
    elif DATA_SOURCE == "csv":
        import_existing_test_data()
    else:
        raise ValueError(f"DATA_SOURCE must be 'generate' or 'csv', got {DATA_SOURCE!r}")


def load_model(device: str):
    print(f"Loading model from {MODEL_PATH}")
    loaded = load_transformer_model(
        TransformerRegressor,
        str(MODEL_PATH),
        device=device,
    )
    model = loaded["model"]
    model.eval()
    return model


def get_max_decode_length(dataset: TransformerDataset, decode_cfg: DecodeConfig) -> int:
    if decode_cfg.max_length is not None:
        return decode_cfg.max_length
    if MAX_SEQ_LENGTH_OVERRIDE is not None:
        return MAX_SEQ_LENGTH_OVERRIDE
    return dataset.max_length


def summarize_mode(rows: list[dict[str, Any]], mode_name: str) -> dict[str, Any]:
    total = len(rows)
    summary = {
        "mode": mode_name,
        "total_examples": total,
        "top1_exact_token_matches": sum(int(r["top1_exact_token_match"]) for r in rows),
        "top1_exact_string_matches": sum(int(r["top1_exact_string_match"]) for r in rows),
        "top1_num_eq_simple": sum(int(r["top1_num_eq_simple"]) for r in rows),
        "top1_num_eq_scrambled": sum(int(r["top1_num_eq_scrambled"]) for r in rows),
        "any_beam_exact_token_matches": sum(int(r["any_beam_exact_token_match"]) for r in rows),
        "any_beam_exact_string_matches": sum(int(r["any_beam_exact_string_match"]) for r in rows),
        "any_beam_num_eq_simple": sum(int(r["any_beam_num_eq_simple"]) for r in rows),
        "any_beam_num_eq_scrambled": sum(int(r["any_beam_num_eq_scrambled"]) for r in rows),
        "avg_candidate_sequences_checked": (
            sum(float(r["candidate_sequences_checked"]) for r in rows) / total if total else 0.0
        ),
        "avg_valid_decoded_candidates": (
            sum(float(r["candidate_valid_decode_count"]) for r in rows) / total if total else 0.0
        ),
        "reranked_top1_replacements": sum(int(r.get("rerank_replaced_top1", 0)) for r in rows),
        "original_top1_num_eq_simple": sum(
            int(r.get("original_top1_num_eq_simple", r["top1_num_eq_simple"])) for r in rows
        ),
        "original_top1_num_eq_scrambled": sum(
            int(r.get("original_top1_num_eq_scrambled", r["top1_num_eq_scrambled"])) for r in rows
        ),
    }
    correct_rows = [r for r in rows if int(r["top1_num_eq_scrambled"])]
    summary["avg_pred_token_count_when_correct"] = (
        sum(int(r["top1_prediction_token_count"]) for r in correct_rows) / len(correct_rows)
        if correct_rows else 0.0
    )
    summary["avg_scrambled_token_count_when_correct"] = (
        sum(int(r["target_scrambled_token_count"]) for r in correct_rows) / len(correct_rows)
        if correct_rows else 0.0
    )
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    total = summary["total_examples"]
    print(f"\nMode: {summary['mode']}")
    print(f"  total examples              : {total}")
    print(
        f"  top-1 exact token matches   : {summary['top1_exact_token_matches']} / {total}"
    )
    print(
        f"  top-1 exact string matches  : {summary['top1_exact_string_matches']} / {total}"
    )
    print(
        f"  top-1 num-eq to simple      : {summary['top1_num_eq_simple']} / {total}"
    )
    print(
        f"  top-1 num-eq to scrambled   : {summary['top1_num_eq_scrambled']} / {total}"
    )
    print(
        f"  any-beam exact token match  : {summary['any_beam_exact_token_matches']} / {total}"
    )
    print(
        f"  any-beam exact string match : {summary['any_beam_exact_string_matches']} / {total}"
    )
    print(
        f"  any-beam num-eq to simple   : {summary['any_beam_num_eq_simple']} / {total}"
    )
    print(
        f"  any-beam num-eq to scrambled: {summary['any_beam_num_eq_scrambled']} / {total}"
    )
    print(
        f"  avg candidates checked      : {summary['avg_candidate_sequences_checked']:.2f}"
    )
    print(
        f"  avg valid decoded candidates: {summary['avg_valid_decoded_candidates']:.2f}"
    )
    if summary["reranked_top1_replacements"]:
        print(
            f"  reranked top-1 replacements : {summary['reranked_top1_replacements']} / {total}"
        )
        print(
            f"  original top-1 num-eq simple: {summary['original_top1_num_eq_simple']} / {total}"
        )


def print_simple_summary(summary: dict[str, Any]) -> None:
    total = summary["total_examples"]

    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%" if total else "n/a"

    top1 = summary["top1_num_eq_scrambled"]
    any_beam = summary["any_beam_num_eq_scrambled"]
    rerank = summary["reranked_top1_replacements"]
    orig_top1 = summary["original_top1_num_eq_scrambled"]
    avg_valid = summary["avg_valid_decoded_candidates"]
    avg_pred_tok = summary["avg_pred_token_count_when_correct"]
    avg_scr_tok = summary["avg_scrambled_token_count_when_correct"]

    print(f"\n[{summary['mode']}]  n={total}")
    print(f"  top-1 correct   : {top1:4d} / {total}  ({pct(top1)})")
    print(f"  any-beam correct: {any_beam:4d} / {total}  ({pct(any_beam)})")
    if rerank:
        gain = top1 - orig_top1
        print(f"  rerank gain     : {gain:+d}  ({pct(gain)})")
    print(f"  avg valid beams : {avg_valid:.1f}")
    if avg_pred_tok > 0:
        print(f"  avg tokens (correct):  pred {avg_pred_tok:.1f}  vs  scrambled {avg_scr_tok:.1f}")


def evaluate_mode(
    model,
    tokenizer: ScatteringAmplitudeTokenizer,
    dataset: TransformerDataset,
    raw_rows: list[dict[str, str]],
    token_rows: list[dict[str, list[int]]],
    cached_kinematics: list[tuple[Any, Any]],
    decode_cfg: DecodeConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    max_length = get_max_decode_length(dataset, decode_cfg)
    detail_rows: list[dict[str, Any]] = []
    row_idx = 0

    print(f"\nRunning decode mode: {decode_cfg.name}")
    print(f"  method     : {decode_cfg.decoding_method}")
    print(f"  max_length : {max_length}")
    if decode_cfg.decoding_method in {"beam", "nucleus"}:
        print(f"  beam_size  : {decode_cfg.beam_size}")
    if decode_cfg.decoding_method == "nucleus":
        print(f"  p_nucleus  : {decode_cfg.p_nucleus}")
        print(f"  temperature: {decode_cfg.temperature_nucleus}")

    for batch in loader:
        src = batch["input"]
        decoded, all_beams = decode_with_model(
            model,
            src,
            max_length=max_length,
            decoding_method=decode_cfg.decoding_method,
            beam_size=decode_cfg.beam_size,
            p_nucleus=decode_cfg.p_nucleus,
            temperature_nucleus=decode_cfg.temperature_nucleus,
            bos_token=SPECIAL_TOKENS["bos"],
            eos_token=SPECIAL_TOKENS["eos"],
            pad_token=SPECIAL_TOKENS["pad"],
        )

        batch_size = src.size(0)
        for i in range(batch_size):
            raw_row = raw_rows[row_idx]
            tok_row = token_rows[row_idx]
            target_simple_tokens = tok_row["simple"]
            target_scrambled_tokens = tok_row["scrambled"]
            target_simple_expr = raw_row["simple"]
            target_scrambled_expr = raw_row["scrambled"]

            top1_full = clean_seq(decoded[i].tolist(), pad_token=SPECIAL_TOKENS["pad"], eos_token=SPECIAL_TOKENS["eos"])
            top1_tokens = strip_special_tokens(top1_full)

            original_top1_decode_ok, original_pred_expr, original_pred_decode_error = safe_decode_infix(
                tokenizer,
                top1_tokens,
            )
            target_decode_ok, target_simple_decoded, _ = safe_decode_infix(tokenizer, target_simple_tokens)
            original_top1_num_eq_simple = (
                numerically_equivalent_exprs(original_pred_expr, target_simple_expr, cached_kinematics)
                if original_top1_decode_ok
                else False
            )
            original_top1_num_eq_scrambled = (
                numerically_equivalent_exprs(original_pred_expr, target_scrambled_expr, cached_kinematics)
                if original_top1_decode_ok
                else False
            )

            candidate_sequences = [top1_tokens]
            raw_beam_candidates = []
            if decode_cfg.evaluate_beam_hypotheses and all_beams is not None:
                raw_beam_candidates = [
                    strip_special_tokens(seq)
                    for seq in all_beams[i]
                ]
                if decode_cfg.max_beams_to_check is not None:
                    raw_beam_candidates = raw_beam_candidates[: decode_cfg.max_beams_to_check]
                candidate_sequences.extend(raw_beam_candidates)
            candidate_sequences = unique_sequences(candidate_sequences)

            candidate_valid_decode_count = 0
            exact_token_count = 0
            exact_string_count = 0
            num_eq_simple_count = 0
            num_eq_scrambled_count = 0
            candidate_exprs: list[str] = []
            candidate_records: list[dict[str, Any]] = []

            for candidate_index, candidate in enumerate(candidate_sequences):
                exact_token = candidate == target_simple_tokens
                if exact_token:
                    exact_token_count += 1

                cand_decode_ok, cand_expr, cand_decode_error = safe_decode_infix(tokenizer, candidate)
                cand_exact_string = False
                cand_num_eq_simple = False
                cand_num_eq_scrambled = False
                if cand_decode_ok:
                    candidate_valid_decode_count += 1
                    candidate_exprs.append(cand_expr)
                    if target_decode_ok and cand_expr == target_simple_decoded:
                        cand_exact_string = True
                        exact_string_count += 1
                    if numerically_equivalent_exprs(cand_expr, target_simple_expr, cached_kinematics):
                        cand_num_eq_simple = True
                        num_eq_simple_count += 1
                    if numerically_equivalent_exprs(cand_expr, target_scrambled_expr, cached_kinematics):
                        cand_num_eq_scrambled = True
                        num_eq_scrambled_count += 1
                candidate_records.append(
                    {
                        "index": candidate_index,
                        "tokens": candidate,
                        "decode_ok": cand_decode_ok,
                        "expr": cand_expr,
                        "decode_error": cand_decode_error or "",
                        "exact_token": exact_token,
                        "exact_string": cand_exact_string,
                        "num_eq_simple": cand_num_eq_simple,
                        "num_eq_scrambled": cand_num_eq_scrambled,
                    }
                )

            selected = candidate_records[0]
            if decode_cfg.rerank_numerical_equiv:
                equivalent_candidates = [
                    record
                    for record in candidate_records
                    if record["decode_ok"] and record["num_eq_scrambled"]
                ]
                if equivalent_candidates:
                    selected = min(
                        equivalent_candidates,
                        key=lambda record: (len(record["tokens"]), record["index"]),
                    )

            top1_tokens = selected["tokens"]
            top1_decode_ok = selected["decode_ok"]
            pred_expr = selected["expr"]
            pred_decode_error = selected["decode_error"]
            top1_exact_token_match = selected["exact_token"]
            top1_exact_string_match = selected["exact_string"]
            top1_num_eq_simple = selected["num_eq_simple"]
            top1_num_eq_scrambled = selected["num_eq_scrambled"]
            rerank_replaced_top1 = int(selected["index"] != 0)

            detail_rows.append(
                {
                    "row_id": row_idx,
                    "mode": decode_cfg.name,
                    "input_scrambled": target_scrambled_expr,
                    "target_simple": target_simple_expr,
                    "target_simple_token_count": len(target_simple_tokens),
                    "target_scrambled_token_count": len(target_scrambled_tokens),
                    "top1_prediction_expr": pred_expr,
                    "top1_prediction_tokens": json.dumps(top1_tokens),
                    "top1_prediction_token_count": len(top1_tokens),
                    "top1_decode_ok": int(top1_decode_ok),
                    "top1_decode_error": pred_decode_error or "",
                    "top1_exact_token_match": int(top1_exact_token_match),
                    "top1_exact_string_match": int(top1_exact_string_match),
                    "top1_num_eq_simple": int(top1_num_eq_simple),
                    "top1_num_eq_scrambled": int(top1_num_eq_scrambled),
                    "rerank_numerical_equiv": int(decode_cfg.rerank_numerical_equiv),
                    "rerank_replaced_top1": rerank_replaced_top1,
                    "rerank_selected_candidate_index": selected["index"],
                    "original_top1_prediction_expr": original_pred_expr,
                    "original_top1_decode_ok": int(original_top1_decode_ok),
                    "original_top1_decode_error": original_pred_decode_error or "",
                    "original_top1_num_eq_simple": int(original_top1_num_eq_simple),
                    "original_top1_num_eq_scrambled": int(original_top1_num_eq_scrambled),
                    "candidate_sequences_checked": len(candidate_sequences),
                    "candidate_valid_decode_count": candidate_valid_decode_count,
                    "candidate_exact_token_count": exact_token_count,
                    "candidate_exact_string_count": exact_string_count,
                    "candidate_num_eq_simple_count": num_eq_simple_count,
                    "candidate_num_eq_scrambled_count": num_eq_scrambled_count,
                    "any_beam_exact_token_match": int(exact_token_count > 0),
                    "any_beam_exact_string_match": int(exact_string_count > 0),
                    "any_beam_num_eq_simple": int(num_eq_simple_count > 0),
                    "any_beam_num_eq_scrambled": int(num_eq_scrambled_count > 0),
                    "candidate_exprs_preview": " || ".join(candidate_exprs[:3]),
                }
            )

            row_idx += 1

    summary = summarize_mode(detail_rows, decode_cfg.name)
    return detail_rows, summary


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_human_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact, human-readable CSV for quick inspection of results.

    Columns:
        simple                 - target canonical expression
        scrambled              - input expression given to the model
        top_pred               - model's top-1 prediction (reranked if enabled)
        correct                - yes/no: top_pred is numerically equivalent to scrambled
        any_correct            - yes/no: any candidate was numerically equivalent
        candidates_checked     - total candidate sequences evaluated
        valid_candidates       - candidates that decoded to a valid infix expression
        pred_tokens            - token count of top_pred
        scrambled_tokens       - token count of scrambled input
        decode_ok              - yes/no: top_pred decoded without error
    """
    if not rows:
        return
    fieldnames = [
        "simple",
        "scrambled",
        "top_pred",
        "correct",
        "any_correct",
        "candidates_checked",
        "valid_candidates",
        "pred_tokens",
        "scrambled_tokens",
        "decode_ok",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "simple": row["target_simple"],
                "scrambled": row["input_scrambled"],
                "top_pred": row["top1_prediction_expr"],
                "correct": "yes" if int(row["top1_num_eq_scrambled"]) else "no",
                "any_correct": "yes" if int(row["any_beam_num_eq_scrambled"]) else "no",
                "candidates_checked": row["candidate_sequences_checked"],
                "valid_candidates": row["candidate_valid_decode_count"],
                "pred_tokens": row["top1_prediction_token_count"],
                "scrambled_tokens": row["target_scrambled_token_count"],
                "decode_ok": "yes" if int(row["top1_decode_ok"]) else "no",
            })


def write_plots(stem: Path, rows: list[dict[str, Any]], mode_name: str) -> None:
    """Generate and save diagnostic plots for one decode run.

    Plots produced (all saved as <stem>_plots/<mode>_*.png):
        1. success_by_length   — success rate (top-1 correct) binned by scrambled token count
        2. token_length_dist   — histogram of scrambled / prediction token lengths, split
                                 by whether the prediction was correct
        3. candidate_quality   — distributions of total candidates checked and valid
                                 decoded candidates, split by whether any candidate was correct
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib/numpy not available — skipping plots")
        return

    plot_dir = stem.parent / f"{stem.name}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    scr_tok   = np.array([int(r["target_scrambled_token_count"]) for r in rows])
    pred_tok  = np.array([int(r["top1_prediction_token_count"])  for r in rows])
    correct   = np.array([int(r["top1_num_eq_scrambled"])        for r in rows], dtype=bool)
    any_corr  = np.array([int(r["any_beam_num_eq_scrambled"])    for r in rows], dtype=bool)
    n_cands   = np.array([int(r["candidate_sequences_checked"])  for r in rows])
    n_valid   = np.array([int(r["candidate_valid_decode_count"]) for r in rows])

    # ── 1. Success rate by scrambled token length ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    bin_edges = np.percentile(scr_tok, np.linspace(0, 100, 9))  # 8 equal-count bins
    bin_edges = np.unique(bin_edges.astype(int))
    bin_labels, bin_correct, bin_any, bin_counts = [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scr_tok >= lo) & (scr_tok < hi)
        if mask.sum() == 0:
            continue
        bin_labels.append(f"{lo}–{hi}")
        bin_correct.append(correct[mask].mean() * 100)
        bin_any.append(any_corr[mask].mean() * 100)
        bin_counts.append(mask.sum())
    # include last bin edge
    mask = scr_tok >= bin_edges[-1]
    if mask.sum() > 0:
        bin_labels.append(f"{bin_edges[-1]}+")
        bin_correct.append(correct[mask].mean() * 100)
        bin_any.append(any_corr[mask].mean() * 100)
        bin_counts.append(mask.sum())
    x = np.arange(len(bin_labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, bin_correct, w, label="top-1 correct", color="steelblue")
    bars2 = ax.bar(x + w / 2, bin_any,     w, label="any-beam correct", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Scrambled token count (bin)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"[{mode_name}] Success rate by input length  (n={len(rows)})")
    ax.legend()
    for bar, count in zip(bars1, bin_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"n={count}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    p = plot_dir / f"{mode_name}_success_by_length.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  plot: {p}")

    # ── 2. Token length distributions ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    bins = np.linspace(0, max(scr_tok.max(), pred_tok.max()) + 10, 30)
    for ax, vals, label in [
        (axes[0], scr_tok, "Scrambled"),
        (axes[1], pred_tok, "Prediction"),
    ]:
        ax.hist(vals[correct],  bins=bins, alpha=0.7, label="correct",   color="steelblue")
        ax.hist(vals[~correct], bins=bins, alpha=0.7, label="incorrect", color="tomato")
        ax.set_xlabel("Token count")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} token length")
        ax.legend()
    fig.suptitle(f"[{mode_name}] Token length distributions  (n={len(rows)})")
    fig.tight_layout()
    p = plot_dir / f"{mode_name}_token_length_dist.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  plot: {p}")

    # ── 3. Candidate quality ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    max_val = max(n_cands.max(), n_valid.max()) + 1
    cbins = np.arange(0, max_val + 2)
    for ax, vals, label in [
        (axes[0], n_cands, "Candidates checked"),
        (axes[1], n_valid, "Valid decoded candidates"),
    ]:
        ax.hist(vals[any_corr],  bins=cbins, alpha=0.7, label="any correct",    color="steelblue")
        ax.hist(vals[~any_corr], bins=cbins, alpha=0.7, label="none correct",   color="tomato")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend()
    fig.suptitle(f"[{mode_name}] Candidate quality  (n={len(rows)})")
    fig.tight_layout()
    p = plot_dir / f"{mode_name}_candidate_quality.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  plot: {p}")


def print_examples(detail_rows: list[dict[str, Any]], count: int) -> None:
    if not detail_rows:
        return
    print(f"\nExample predictions ({min(count, len(detail_rows))} rows):")
    for row in detail_rows[:count]:
        print(f"  row {row['row_id']} [{row['mode']}]")
        print(f"    target simple : {row['target_simple'][:180]}")
        print(f"    input scram   : {row['input_scrambled'][:180]}")
        print(f"    top1 pred     : {row['top1_prediction_expr'][:180]}")
        if int(row.get("rerank_replaced_top1", 0)):
            print(f"    original top1 : {row['original_top1_prediction_expr'][:180]}")
        print(
            f"    top1 metrics  : token={row['top1_exact_token_match']} "
            f"string={row['top1_exact_string_match']} "
            f"num_simple={row['top1_num_eq_simple']} "
            f"num_scr={row['top1_num_eq_scrambled']}"
        )
        print(
            f"    any-beam      : token={row['any_beam_exact_token_match']} "
            f"string={row['any_beam_exact_string_match']} "
            f"num_simple={row['any_beam_num_eq_simple']} "
            f"num_scr={row['any_beam_num_eq_scrambled']}"
        )


def main() -> None:
    args = parse_args()
    apply_cli_config(args)

    t0 = time.perf_counter()
    ensure_output_dirs()

    device = resolve_device()
    print(f"Using device: {device}")

    prepare_test_data()
    raw_rows = load_raw_rows(RAW_CSV_PATH)
    token_rows = load_token_rows(TOK_CSV_PATH)
    if len(raw_rows) != len(token_rows):
        raise ValueError("Raw and tokenised row counts do not match")

    tokenizer = ScatteringAmplitudeTokenizer(max_particles=TOKENIZER_MAX_PARTICLES)
    dataset = TransformerDataset(str(TOK_CSV_PATH), max_length=MAX_SEQ_LENGTH_OVERRIDE)
    cached_kinematics = precompute_kinematics()
    model = load_model(device)

    summary_rows: list[dict[str, Any]] = []
    for decode_cfg in DECODE_RUNS:
        if not decode_cfg.enabled:
            continue
        detail_rows, summary = evaluate_mode(
            model,
            tokenizer,
            dataset,
            raw_rows,
            token_rows,
            cached_kinematics,
            decode_cfg,
        )
        detail_path = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_{decode_cfg.name}_results.csv"
        write_detail_csv(detail_path, detail_rows)
        if HUMAN_CSV:
            human_path = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_{decode_cfg.name}_human.csv"
            write_human_csv(human_path, detail_rows)
            print(f"  wrote human CSV to {human_path}")
        if PLOTS:
            write_plots(DATA_TESTING_DIR / OUTPUT_SUBDIR / DATA_FILENAME_STEM, detail_rows, decode_cfg.name)
        if SIMPLE_SUMMARY:
            print_simple_summary(summary)
        else:
            print_summary(summary)
        print_examples(detail_rows, PRINT_EXAMPLES)
        print(f"  wrote detail results to {detail_path}")
        summary_rows.append(summary)

    write_summary_csv(SUMMARY_CSV_PATH, summary_rows)
    print(f"\nWrote summary to {SUMMARY_CSV_PATH}")
    print(f"Total wall time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
