#!/usr/bin/env python3
"""
Evaluate one trained transformer on freshly generated amplitude data.

All configuration lives in this file. The script:

1. Generates new raw data, imports a raw CSV, or imports a single-amplitude
   token CSV.
2. Tokenises raw expressions, or normalises the already-tokenised input.
3. Loads one model checkpoint.
4. Runs one or more decoding modes (greedy by default, optional beam/nucleus).
5. Reports exact-match and numerical-equivalence metrics against both the
   target simple expressions and the input scrambled expressions.
6. Mirrors terminal output to a timestamped, text-based PDF.

Outputs are written under DATA_TESTING_DIR / OUTPUT_SUBDIR.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

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
# "auto" chooses CUDA, then Apple MPS, then CPU. Set explicitly for repeatable timing/debugging.
DEVICE = "auto"  # "auto", "cpu", "cuda", "mps"

# Number of external legs. For scalar QED, N=4 means two photon legs; for
# Yang-Mills, all N legs are gluons.
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
# Data source for this evaluation. Use "generate" for fresh synthetic data,
# "csv" for an existing raw CSV with columns named simple and scrambled, or
# "single-amplitude" for one amplitude per row in token or Feynman format.
DATA_SOURCE = "csv"  # "generate", "csv", "single-amplitude"
# Used only when DATA_SOURCE="csv". Relative paths are resolved from repo root.
# EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "sqed_w0110_mM00M_den6_maxD2_20000_phys_oneshot.csv" # Paolo's 20k dataset
# EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "gi_4pt_oneshot_5k_val.csv" # Nathan's 5k dataset
# EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "messified.csv" # Paolo's 20k dataset
# EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "sqed_oneshot_150.csv" # Paolo's 20k dataset
EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "sqed_4ptseed_oneshot.csv" # 100 scrambled amplitudes

# Gravity numerical checks need the process assignment for every row because
# 3s2h and 4s1h carry different graviton legs. When this is None, the evaluator
# looks for a process column in EXISTING_RAW_CSV_PATH and then for a sibling
# file whose name replaces "_raw.csv[.gz]" with "_metadata.csv[.gz]".
GRAVITY_METADATA_CSV_PATH: Path | None = None
# Use this for a homogeneous gravity CSV that has no metadata file.
GRAVITY_PROCESS: str | None = None  # "3s2h", "4s1h", or None

# Used only when DATA_SOURCE="single-amplitude". The input may be .csv or
# .csv.gz. "tokens" expects a header and a JSON list of integer token IDs in
# SINGLE_AMPLITUDE_TOKENS_COLUMN. "feyn" expects headerless id,expression rows
# such as gluon5feyn12345.csv.gz. "auto" distinguishes those two layouts.
# In both cases the same amplitude is used as target and scrambled input.
SINGLE_AMPLITUDE_INPUT_CSV_PATH: Path | None = None
SINGLE_AMPLITUDE_INPUT_FORMAT = "auto"  # "auto", "tokens", "feyn"
SINGLE_AMPLITUDE_TOKENS_COLUMN = "tokens"
SINGLE_AMPLITUDE_EXPRESSION_COLUMN = 1


# Optional row cap for imported raw simple/scrambled CSVs.
# None evaluates every row in the file.
EXISTING_CSV_MAX_ROWS = 100 

# Optional exact-pair dedupe for imported raw CSVs before tokenization/evaluation.
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
# Maximum scrambled-input content tokens, excluding the BOS/EOS tokens added by
# TransformerDataset. None means the evaluator imposes no input cap; the loaded
# model's max_seq_len remains a hard positional-encoding limit.
INPUT_TOKEN_LIMIT: int | None = None
# Maximum generated output length, including BOS/EOS. None falls back to the
# prepared dataset's longest source/target sequence. This does not truncate input.
MAX_SEQ_LENGTH_OVERRIDE = None
# Number of example rows printed per decode mode; full details are always written to CSV.
PRINT_EXAMPLES = 3
# When True, print a concise one-block summary instead of the verbose per-metric list.
SIMPLE_SUMMARY = True
# When True, write a compact human-readable CSV alongside the full detail CSV.
HUMAN_CSV = True
# When True, generate diagnostic plots alongside the evaluation outputs.
PLOTS = True

# Numeric equivalence check for model predictions. "sqed" uses two massive
# scalar endpoint legs; "ym" uses all-massless gluon kinematics and e_1...e_N;
# "gravity" uses complex five-point spinor-helicity kinematics and the process
# assignment loaded from the gravity metadata CSV.
NUMERIC_BACKEND = "sqed"  # "sqed", "ym", "gravity"
NUMERIC_EQUIV_SAMPLES = 3
NUMERIC_EQUIV_SEED = 151
NUMERIC_EQUIV_MASS = 2.0
NUMERIC_EQUIV_ENERGY_SCALE = 2.0
# None chooses ("coulomb",) for sqed and ("coulomb", "covariant") for ym.
NUMERIC_EQUIV_POL_MODES: tuple[str, ...] | None = None
# Gravity checks rebuild polarisations at fixed momenta with these reference
# choices and, by default, also test an explicit gauge-shifted point.
GRAVITY_REFERENCE_MODES: tuple[str, ...] = ("first", "last", "random")
GRAVITY_GAUGE_SHIFT = True
# Equivalence passes if either absolute or relative tolerance is met on every
# numeric sample. None selects backend defaults: sqed=(1e-12, 1e-10) and
# ym=(1e-10, 1e-8), matching data_gen_ym.numerics._validate_pair, while
# gravity=(2e-9, 2e-8), matching data_gen_gravity.core.numerically_equivalent.
NUMERIC_TOL_ABS: float | None = None
NUMERIC_TOL_REL: float | None = None
BEAM_SIZE = 50


@dataclass(frozen=True)
class DecodeConfig:
    name: str
    # Toggle individual modes without changing the rest of the evaluation setup.
    enabled: bool
    # Supported by decode_with_model: "greedy", "beam", or "nucleus".
    decoding_method: str
    # Per-mode override. None falls back to MAX_SEQ_LENGTH_OVERRIDE or the
    # longest prepared source/target sequence.
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument(
        "--numeric-backend",
        choices=["sqed", "ym", "yang-mills", "gravity"],
        default=None,
        help="Kinematics and expression evaluator used for numerical equivalence.",
    )
    parser.add_argument(
        "--numeric-pol-modes",
        nargs="+",
        choices=["coulomb", "covariant"],
        default=None,
        help=(
            "Polarisation modes used for numerical checks. Defaults to coulomb "
            "for sqed and both coulomb/covariant for ym."
        ),
    )
    parser.add_argument(
        "--numeric-mass",
        type=float,
        default=None,
        help="Mass of the two scalar endpoint legs for sqed numerical checks.",
    )
    parser.add_argument(
        "--numeric-energy-scale",
        type=float,
        default=None,
        help="Overall energy scale for all-massless Yang-Mills numerical checks.",
    )
    parser.add_argument(
        "--gravity-reference-modes",
        nargs="+",
        choices=["first", "last", "random", "cyclic"],
        default=None,
        help=(
            "Spinor-reference choices used at each complex gravity phase-space "
            "point (default: first last random)."
        ),
    )
    parser.add_argument(
        "--gravity-gauge-shift",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also test an explicit graviton gauge shift (default: enabled).",
    )
    parser.add_argument(
        "--numeric-tol-abs",
        type=float,
        default=None,
        help="Absolute tolerance for numerical equivalence.",
    )
    parser.add_argument(
        "--numeric-tol-rel",
        type=float,
        default=None,
        help="Relative tolerance for numerical equivalence.",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-scr", type=int, default=None)
    parser.add_argument("--max-scr", type=int, default=None)
    parser.add_argument("--min-terms", type=int, default=None)
    parser.add_argument("--max-terms", type=int, default=None)
    parser.add_argument(
        "--input-token-limit",
        "--max-input-tokens",
        dest="input_token_limit",
        type=int,
        default=None,
        help=(
            "Maximum scrambled-input content tokens, excluding BOS/EOS. "
            "Use 0 or a negative value for no evaluator-imposed limit."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help=(
            "Legacy content-token cap: limits input content to N and output to "
            "N content tokens plus BOS/EOS, unless a dedicated option overrides it."
        ),
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=None,
        help=(
            "Maximum generated output length, including BOS/EOS. "
            "Use 0 or a negative value to infer it from the dataset."
        ),
    )
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
        choices=["generate", "csv", "single-amplitude"],
        default=None,
        help=(
            "Use generated synthetic data, import a raw simple/scrambled CSV, "
            "or import a token/Feynman-format single-amplitude CSV."
        ),
    )
    parser.add_argument("--existing-raw-csv", type=str, default=None)
    gravity_process_source = parser.add_mutually_exclusive_group()
    gravity_process_source.add_argument(
        "--gravity-metadata-csv",
        type=str,
        default=None,
        help=(
            "CSV or CSV.GZ aligned with the raw gravity data and containing a "
            "process column (3s2h or 4s1h). If omitted, a sibling *_metadata "
            "file is discovered automatically."
        ),
    )
    gravity_process_source.add_argument(
        "--gravity-process",
        choices=["3s2h", "4s1h"],
        default=None,
        help="Process to use for every row in a homogeneous gravity CSV.",
    )
    parser.add_argument(
        "--single-amplitude-input-csv",
        "--single-amplitude-csv",
        dest="single_amplitude_input_csv",
        type=str,
        default=None,
        help=(
            "Input .csv or .csv.gz containing either JSON token lists or "
            "headerless id,expression Feynman rows."
        ),
    )
    parser.add_argument(
        "--single-amplitude-output-csv",
        type=str,
        default=None,
        help=(
            "Optional path for the normalized simple/scrambled token CSV. "
            "Defaults to the token CSV derived from --output-stem."
        ),
    )
    parser.add_argument(
        "--single-amplitude-input-format",
        "--single-amplitude-format",
        choices=["auto", "tokens", "feyn"],
        default=None,
        help=(
            "Single-amplitude CSV layout. 'tokens' expects a named JSON-token "
            "column; 'feyn' expects headerless id,expression rows; 'auto' detects it."
        ),
    )
    parser.add_argument(
        "--tokens-column",
        type=str,
        default=None,
        help="Column containing token lists in a single-amplitude input (default: tokens).",
    )
    parser.add_argument(
        "--single-amplitude-expression-column",
        type=int,
        default=None,
        help="Zero-based expression column for a Feynman-format single-amplitude CSV.",
    )
    parser.add_argument(
        "--existing-csv-max-rows",
        type=int,
        default=None,
        help="Maximum imported raw rows; use 0 or a negative value for all rows.",
    )
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
    global GRAVITY_METADATA_CSV_PATH, GRAVITY_PROCESS
    global SINGLE_AMPLITUDE_INPUT_CSV_PATH, SINGLE_AMPLITUDE_INPUT_FORMAT
    global SINGLE_AMPLITUDE_TOKENS_COLUMN, SINGLE_AMPLITUDE_EXPRESSION_COLUMN
    global EXISTING_CSV_MAX_ROWS, EXISTING_CSV_DEDUPE, GEN_MIN_TERMS, GEN_MAX_TERMS
    global GEN_MIN_SCRAMBLES, GEN_MAX_SCRAMBLES, BATCH_SIZE, INPUT_TOKEN_LIMIT
    global MAX_SEQ_LENGTH_OVERRIDE, NUMERIC_BACKEND, NUMERIC_EQUIV_POL_MODES
    global NUMERIC_EQUIV_MASS, NUMERIC_EQUIV_ENERGY_SCALE
    global GRAVITY_REFERENCE_MODES, GRAVITY_GAUGE_SHIFT
    global NUMERIC_TOL_ABS, NUMERIC_TOL_REL
    global DECODE_RUNS, CLI_SCRAMBLES, SIMPLE_SUMMARY, HUMAN_CSV, PLOTS

    single_amplitude_args_used = any(
        value is not None
        for value in (
            args.single_amplitude_input_csv,
            args.single_amplitude_output_csv,
            args.single_amplitude_input_format,
            args.tokens_column,
            args.single_amplitude_expression_column,
        )
    )
    if (
        single_amplitude_args_used
        and args.data_source is not None
        and args.data_source != "single-amplitude"
    ):
        raise ValueError(
            "Single-amplitude CSV options require "
            "--data-source single-amplitude (or no explicit --data-source)"
        )

    if args.model_path is not None:
        MODEL_PATH = resolve_input_path(args.model_path)
    if args.device is not None:
        DEVICE = args.device
    if args.n_particles is not None:
        N_PARTICLES = args.n_particles
    if args.numeric_backend is not None:
        NUMERIC_BACKEND = (
            "ym" if args.numeric_backend == "yang-mills" else args.numeric_backend
        )
        if NUMERIC_BACKEND == "gravity" and args.n_particles is None:
            N_PARTICLES = 5
    if args.numeric_pol_modes is not None:
        NUMERIC_EQUIV_POL_MODES = tuple(dict.fromkeys(args.numeric_pol_modes))
    if args.numeric_mass is not None:
        NUMERIC_EQUIV_MASS = args.numeric_mass
    if args.numeric_energy_scale is not None:
        NUMERIC_EQUIV_ENERGY_SCALE = args.numeric_energy_scale
    if args.gravity_reference_modes is not None:
        GRAVITY_REFERENCE_MODES = tuple(
            dict.fromkeys(args.gravity_reference_modes)
        )
    if args.gravity_gauge_shift is not None:
        GRAVITY_GAUGE_SHIFT = args.gravity_gauge_shift
    if args.numeric_tol_abs is not None:
        NUMERIC_TOL_ABS = args.numeric_tol_abs
    if args.numeric_tol_rel is not None:
        NUMERIC_TOL_REL = args.numeric_tol_rel
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
        legacy_limit = None if args.max_tokens <= 0 else args.max_tokens
        MAX_SEQ_LENGTH_OVERRIDE = (
            None if legacy_limit is None else legacy_limit + 2
        )
        if args.input_token_limit is None:
            INPUT_TOKEN_LIMIT = legacy_limit
    if args.input_token_limit is not None:
        INPUT_TOKEN_LIMIT = (
            None if args.input_token_limit <= 0 else args.input_token_limit
        )
    if args.max_decode_tokens is not None:
        MAX_SEQ_LENGTH_OVERRIDE = (
            None if args.max_decode_tokens <= 0 else args.max_decode_tokens
        )
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.existing_raw_csv is not None:
        EXISTING_RAW_CSV_PATH = resolve_input_path(args.existing_raw_csv)
    if args.gravity_metadata_csv is not None:
        GRAVITY_METADATA_CSV_PATH = resolve_input_path(args.gravity_metadata_csv)
        GRAVITY_PROCESS = None
    if args.gravity_process is not None:
        GRAVITY_PROCESS = args.gravity_process
        GRAVITY_METADATA_CSV_PATH = None
    if args.single_amplitude_input_csv is not None:
        SINGLE_AMPLITUDE_INPUT_CSV_PATH = resolve_input_path(args.single_amplitude_input_csv)
    if args.single_amplitude_input_format is not None:
        SINGLE_AMPLITUDE_INPUT_FORMAT = args.single_amplitude_input_format
    if args.tokens_column is not None:
        if not args.tokens_column.strip():
            raise ValueError("--tokens-column cannot be empty")
        SINGLE_AMPLITUDE_TOKENS_COLUMN = args.tokens_column
    if args.single_amplitude_expression_column is not None:
        SINGLE_AMPLITUDE_EXPRESSION_COLUMN = args.single_amplitude_expression_column
    if args.existing_csv_max_rows is not None:
        EXISTING_CSV_MAX_ROWS = (
            None
            if args.existing_csv_max_rows <= 0
            else args.existing_csv_max_rows
        )
    if args.no_dedupe:
        EXISTING_CSV_DEDUPE = False

    if args.data_source is not None:
        DATA_SOURCE = args.data_source
    elif single_amplitude_args_used:
        DATA_SOURCE = "single-amplitude"
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
    if args.single_amplitude_output_csv is not None:
        TOK_CSV_PATH = resolve_input_path(args.single_amplitude_output_csv)

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
sys.path.insert(0, str(ROOT))

import gen_data as gd
from numeric_utils import numeric_values_close
from Tokenizer import ScatteringAmplitudeTokenizer
from data_gen.data_gen_gravity.core import (
    PROCESS_SPECS as GRAVITY_PROCESS_SPECS,
    eval_expression as eval_gravity_expression,
)
from data_gen.data_gen_gravity.kinematics import (
    generate_kinematics as generate_gravity_kinematics,
    with_references as gravity_with_references,
)
from data_import import TransformerDataset, dynamic_pad_collate
from data_gen_ym.kinematics import generate_kinematics as generate_ym_kinematics
from data_gen_ym.numerics import eval_infix_numeric as eval_ym_infix_numeric
from kinematics import generate_kinematics as generate_sqed_kinematics
from single_amplitude_test_set_to_simple_scrambled import (
    convert_file as convert_single_amplitude_file,
)
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

_ANSI_ESCAPE_RE = re.compile(
    r"(?:\x1B\][^\x07]*(?:\x07|\x1B\\)|"
    r"\x1B(?:\[[0-?]*[ -/]*[@-~]|[@-_]))"
)


class TeeTextIO:
    """Mirror one text stream to the terminal and a shared transcript."""

    def __init__(
        self,
        terminal_stream: Any,
        transcript: io.StringIO,
        lock: threading.RLock,
    ) -> None:
        self._terminal_stream = terminal_stream
        self._transcript = transcript
        self._lock = lock

    def write(self, text: str) -> int:
        with self._lock:
            self._terminal_stream.write(text)
            self._transcript.write(text)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._terminal_stream.flush()
            self._transcript.flush()

    def __getattr__(self, name: str) -> Any:
        # Preserve compatibility with code that inspects encoding, isatty(),
        # fileno(), and other properties of sys.stdout/sys.stderr.
        return getattr(self._terminal_stream, name)


@contextmanager
def capture_terminal_output() -> Iterator[io.StringIO]:
    """Yield a transcript while keeping stdout and stderr visible."""

    transcript = io.StringIO()
    lock = threading.RLock()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = TeeTextIO(original_stdout, transcript, lock)
    tee_stderr = TeeTextIO(original_stderr, transcript, lock)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    try:
        yield transcript
    finally:
        try:
            tee_stdout.flush()
            tee_stderr.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def evaluation_data_type_label() -> str:
    """Return the human-readable physics data type used in PDF filenames."""

    normalized = NUMERIC_BACKEND.strip().lower()
    if normalized in {"ym", "yang-mills"}:
        return "Yang-mills"
    if normalized == "sqed":
        return "SQED"
    if normalized == "gravity":
        return "Gravity"
    # validate_runtime_config normally makes this unreachable, but retaining a
    # safe label lets configuration errors still produce a transcript PDF.
    return re.sub(r"[^A-Za-z0-9-]+", "-", NUMERIC_BACKEND).strip("-") or "Unknown"


def evaluation_pdf_path(started_at: datetime) -> Path:
    """Build the requested timestamped evaluation-information PDF path."""

    timestamp = started_at.strftime("%Y-%m-%d_%H-%M-%S-%f")
    filename = (
        f"{evaluation_data_type_label()}_{N_PARTICLES}_"
        f"Evaluation_Information_{timestamp}.pdf"
    )
    return DATA_TESTING_DIR / OUTPUT_SUBDIR / filename


def require_reportlab() -> None:
    """Fail before model evaluation if the configured PDF dependency is absent."""

    try:
        __import__("reportlab.pdfgen.canvas")
    except ImportError as exc:
        raise RuntimeError(
            "Saving evaluation output as PDF requires ReportLab. "
            "Install environment/requirements.txt before running this script."
        ) from exc


def _pdf_monospace_font() -> tuple[str, bool]:
    """Register a Unicode text font, preferring an installed monospace face."""

    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_name = "EvaluationTerminalMono"
    if font_name in pdfmetrics.getRegisteredFontNames():
        return font_name, True

    candidates = (
        Path("/System/Library/Fonts/SFNSMono.ttf"),
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/System/Library/Fonts/Supplemental/Andale Mono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/cour.ttf"),
        Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf",
    )
    for font_path in candidates:
        if not font_path.is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            return font_name, True
        except Exception:
            continue
    return "Courier", False


def _wrapped_terminal_lines(
    terminal_text: str,
    max_characters: int,
) -> Iterator[str]:
    """Strip terminal controls and wrap text without losing long expressions."""

    normalized = _ANSI_ESCAPE_RE.sub("", terminal_text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\f", "\n")
    for raw_line in normalized.split("\n"):
        expanded = raw_line.expandtabs(4)
        printable = "".join(
            character if character >= " " else " "
            for character in expanded
        )
        if not printable:
            yield ""
            continue
        for start in range(0, len(printable), max_characters):
            yield printable[start : start + max_characters]


def write_terminal_output_pdf(
    path: Path,
    terminal_text: str,
    started_at: datetime,
) -> None:
    """Render terminal output as selectable, wrapped text over as many pages as needed."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = A4
    left_margin = 42.0
    right_margin = 42.0
    bottom_margin = 34.0
    text_top = page_height - 55.0
    font_size = 8.25
    line_leading = 10.25
    font_name, unicode_font = _pdf_monospace_font()
    character_width = pdfmetrics.stringWidth("M", font_name, font_size)
    max_characters = max(
        20,
        int((page_width - left_margin - right_margin) / character_width),
    )

    pdf_buffer = io.BytesIO()
    report = canvas.Canvas(pdf_buffer, pagesize=A4, pageCompression=1)
    report.setAuthor("evaluate_model.py")
    report.setCreator("evaluate_model.py")
    report.setSubject("Evaluation terminal output")
    report.setTitle(path.stem)

    header = (
        f"{evaluation_data_type_label()} | {N_PARTICLES} particles | "
        "Evaluation Information"
    )
    timestamp = started_at.strftime("%Y-%m-%d %H:%M:%S %Z")

    def start_page() -> Any:
        report.setFillColor(colors.HexColor("#1F2937"))
        report.setFont("Helvetica-Bold", 10)
        report.drawString(left_margin, page_height - 29.0, header)
        report.setFont("Helvetica", 7.5)
        report.drawRightString(page_width - right_margin, page_height - 29.0, timestamp)
        report.setStrokeColor(colors.HexColor("#D1D5DB"))
        report.line(
            left_margin,
            page_height - 37.0,
            page_width - right_margin,
            page_height - 37.0,
        )
        text_object = report.beginText(left_margin, text_top)
        text_object.setFont(font_name, font_size)
        text_object.setFillColor(colors.black)
        text_object.setLeading(line_leading)
        return text_object

    def finish_page(text_object: Any, page_number: int, *, last: bool) -> None:
        report.drawText(text_object)
        report.setFillColor(colors.HexColor("#6B7280"))
        report.setFont("Helvetica", 7)
        report.drawRightString(
            page_width - right_margin,
            20.0,
            f"Page {page_number}",
        )
        if not last:
            report.showPage()

    page_number = 1
    text_object = start_page()
    lines = list(_wrapped_terminal_lines(terminal_text, max_characters))
    if not lines:
        lines = ["(No terminal output was produced.)"]

    for line in lines:
        if text_object.getY() < bottom_margin + line_leading:
            finish_page(text_object, page_number, last=False)
            page_number += 1
            text_object = start_page()
        if not unicode_font:
            line = line.encode("cp1252", errors="replace").decode("cp1252")
        text_object.textLine(line)

    finish_page(text_object, page_number, last=True)
    report.save()

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(pdf_buffer.getvalue())
    try:
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def resolve_device() -> str:
    if DEVICE == "cpu":
        return "cpu"
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE='cuda' but CUDA is not available")
        return "cuda"
    if DEVICE == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("DEVICE='mps' but Apple Metal/MPS is not available")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
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


def select_candidate_for_reporting(
    candidate_records: list[dict[str, Any]],
    *,
    rerank_numerical_equiv: bool,
) -> tuple[dict[str, Any], str]:
    """Choose a generated candidate that can always be reported to the user.

    Numerical equivalence remains the strongest reranking signal. If no
    equivalent candidate exists and the model's original top-1 output is
    malformed, fall back to the first generated candidate that decodes into a
    valid expression. When every candidate is malformed, retain the original
    top-1 so its raw prefix tokens can still be displayed.
    """

    if not candidate_records:
        raise ValueError("Cannot select a prediction from an empty candidate list")

    selected = candidate_records[0]
    selection_reason = "model_top1"

    if rerank_numerical_equiv:
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
            selection_reason = "numerically_equivalent_rerank"

    if not selected["decode_ok"]:
        valid_fallback = next(
            (record for record in candidate_records if record["decode_ok"]),
            None,
        )
        if valid_fallback is not None:
            selected = valid_fallback
            selection_reason = "valid_decode_fallback"

    return selected, selection_reason


def prediction_text_for_display(
    tokenizer: ScatteringAmplitudeTokenizer,
    candidate: dict[str, Any],
) -> str:
    """Return a non-empty, human-readable representation of model output."""

    expression = str(candidate.get("expr") or "").strip()
    if candidate.get("decode_ok") and expression:
        return expression

    tokens = list(candidate.get("tokens") or [])
    if tokens:
        try:
            raw_prefix = tokenizer.decode_prefix(tokens)
        except Exception:
            raw_prefix = json.dumps(tokens)
        if raw_prefix:
            return f"[malformed prefix tokens] {raw_prefix}"

    return "[empty token prediction]"


def open_csv_text(path: Path, mode: str):
    """Open a plain or gzip-compressed CSV as UTF-8 text."""
    text_mode = mode if "t" in mode else f"{mode}t"
    if path.suffix == ".gz":
        return gzip.open(path, text_mode, newline="", encoding="utf-8")
    return path.open(text_mode, newline="", encoding="utf-8")


def load_raw_rows(path: Path) -> list[dict[str, str]]:
    with open_csv_text(path, "r") as handle:
        return list(csv.DictReader(handle))


def resolve_input_path(path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def read_raw_pairs(path: Path) -> list[tuple[str, str]]:
    with open_csv_text(path, "r") as handle:
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


def csv_fieldnames(path: Path) -> list[str]:
    with open_csv_text(path, "r") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def infer_gravity_metadata_path(raw_path: Path) -> Path | None:
    """Find the metadata file emitted beside a standard gravity raw CSV."""

    if "process" in csv_fieldnames(raw_path):
        return raw_path

    replacements = (
        ("_raw.csv.gz", "_metadata.csv.gz"),
        ("_raw.csv", "_metadata.csv"),
    )
    for raw_suffix, metadata_suffix in replacements:
        if raw_path.name.endswith(raw_suffix):
            candidate = raw_path.with_name(
                raw_path.name[: -len(raw_suffix)] + metadata_suffix
            )
            if candidate.is_file():
                return candidate
    return None


def resolve_gravity_processes(
    raw_rows: list[dict[str, str]],
) -> list[str | None]:
    """Return one process label per prepared row.

    Gravity metadata is aligned to the source raw CSV before applying the same
    first-occurrence deduplication and row cap as import_existing_test_data.
    This keeps process labels correct even when the source contains duplicates.
    """

    if NUMERIC_BACKEND != "gravity":
        return [None] * len(raw_rows)

    if GRAVITY_PROCESS is not None:
        return [GRAVITY_PROCESS] * len(raw_rows)
    if DATA_SOURCE != "csv":
        raise ValueError(
            "Gravity evaluation outside DATA_SOURCE='csv' requires "
            "--gravity-process 3s2h or --gravity-process 4s1h"
        )

    source_path = resolve_input_path(EXISTING_RAW_CSV_PATH)
    metadata_path = (
        resolve_input_path(GRAVITY_METADATA_CSV_PATH)
        if GRAVITY_METADATA_CSV_PATH is not None
        else infer_gravity_metadata_path(source_path)
    )
    if metadata_path is None:
        raise ValueError(
            f"Could not infer gravity metadata for {source_path}. Pass "
            "--gravity-metadata-csv or --gravity-process."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Gravity metadata CSV does not exist: {metadata_path}")

    source_pairs = read_raw_pairs(source_path)
    metadata_rows = load_raw_rows(metadata_path)
    if len(metadata_rows) != len(source_pairs):
        raise ValueError(
            f"Gravity metadata row count ({len(metadata_rows)}) does not match "
            f"the raw CSV row count ({len(source_pairs)})"
        )

    annotated: list[tuple[tuple[str, str], str]] = []
    process_by_pair: dict[tuple[str, str], str] = {}
    for row_index, (pair, metadata) in enumerate(
        zip(source_pairs, metadata_rows, strict=True),
        start=2,
    ):
        process = (metadata.get("process") or "").strip()
        if process not in GRAVITY_PROCESS_SPECS:
            raise ValueError(
                f"{metadata_path}:{row_index} has unsupported gravity process "
                f"{process!r}; expected one of {sorted(GRAVITY_PROCESS_SPECS)}"
            )

        metadata_simple = (metadata.get("simple") or "").strip()
        metadata_scrambled = (metadata.get("scrambled") or "").strip()
        if metadata_simple or metadata_scrambled:
            if (metadata_simple, metadata_scrambled) != pair:
                raise ValueError(
                    f"{metadata_path}:{row_index} is not aligned with "
                    f"{source_path}:{row_index}"
                )
        previous_process = process_by_pair.get(pair)
        if previous_process is not None and previous_process != process:
            raise ValueError(
                f"Gravity pair at {source_path}:{row_index} is assigned to both "
                f"{previous_process!r} and {process!r}"
            )
        process_by_pair[pair] = process
        annotated.append((pair, process))

    if EXISTING_CSV_DEDUPE:
        seen: set[tuple[str, str]] = set()
        deduped: list[tuple[tuple[str, str], str]] = []
        for pair, process in annotated:
            if pair not in seen:
                seen.add(pair)
                deduped.append((pair, process))
        annotated = deduped
    if EXISTING_CSV_MAX_ROWS is not None:
        annotated = annotated[: int(EXISTING_CSV_MAX_ROWS)]

    prepared_pairs = [
        (
            (row.get("simple") or "").strip(),
            (row.get("scrambled") or "").strip(),
        )
        for row in raw_rows
    ]
    annotated_pairs = [pair for pair, _ in annotated]
    if annotated_pairs != prepared_pairs:
        raise ValueError(
            "Prepared gravity rows no longer align with the source metadata. "
            "Check the raw/metadata paths and deduplication settings."
        )

    print(f"Gravity metadata: {metadata_path}")
    return [process for _, process in annotated]


def load_token_rows(path: Path) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    with open_csv_text(path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "simple": json.loads(row["simple"]),
                    "scrambled": json.loads(row["scrambled"]),
                }
            )
    return rows


def validate_input_token_rows(rows: list[dict[str, list[int]]]) -> int:
    if not rows:
        raise ValueError("The tokenised evaluation dataset contains no rows")

    max_content_tokens = 0
    for row_number, row in enumerate(rows, start=2):
        tokens = row.get("scrambled")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError(
                f"{TOK_CSV_PATH}:{row_number} scrambled input must be a JSON list of integers"
            )
        token_count = len(tokens)
        max_content_tokens = max(max_content_tokens, token_count)
        if INPUT_TOKEN_LIMIT is not None and token_count > INPUT_TOKEN_LIMIT:
            raise ValueError(
                f"{TOK_CSV_PATH}:{row_number} has {token_count} input content tokens "
                f"({token_count + 2} with BOS/EOS), exceeding "
                f"INPUT_TOKEN_LIMIT={INPUT_TOKEN_LIMIT}"
            )
    return max_content_tokens


def resolve_numeric_pol_modes() -> tuple[str, ...]:
    if NUMERIC_BACKEND == "gravity":
        if NUMERIC_EQUIV_POL_MODES is not None:
            raise ValueError(
                "NUMERIC_EQUIV_POL_MODES applies only to SQED/Yang-Mills; "
                "use GRAVITY_REFERENCE_MODES for gravity"
            )
        return ()

    modes = NUMERIC_EQUIV_POL_MODES
    if modes is None:
        modes = ("coulomb", "covariant") if NUMERIC_BACKEND == "ym" else ("coulomb",)
    if not modes:
        raise ValueError("NUMERIC_EQUIV_POL_MODES must contain at least one mode")
    invalid = sorted(set(modes) - {"coulomb", "covariant"})
    if invalid:
        raise ValueError(f"Unsupported numerical polarisation modes: {invalid}")
    return tuple(dict.fromkeys(modes))


def resolve_gravity_reference_modes() -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(GRAVITY_REFERENCE_MODES))
    invalid = sorted(set(modes) - {"first", "last", "random", "cyclic"})
    if invalid:
        raise ValueError(f"Unsupported gravity reference modes: {invalid}")
    if not modes and not GRAVITY_GAUGE_SHIFT:
        raise ValueError(
            "Gravity numerical checks need at least one reference mode or an "
            "enabled gauge-shift check"
        )
    return modes


def resolve_numeric_tolerances() -> tuple[float, float]:
    if NUMERIC_BACKEND == "gravity":
        default_abs, default_rel = 2e-9, 2e-8
    elif NUMERIC_BACKEND == "ym":
        default_abs, default_rel = 1e-10, 1e-8
    else:
        default_abs, default_rel = 1e-12, 1e-10
    tol_abs = default_abs if NUMERIC_TOL_ABS is None else NUMERIC_TOL_ABS
    tol_rel = default_rel if NUMERIC_TOL_REL is None else NUMERIC_TOL_REL
    if not math.isfinite(tol_abs) or tol_abs < 0:
        raise ValueError("NUMERIC_TOL_ABS must be finite and non-negative")
    if not math.isfinite(tol_rel) or tol_rel < 0:
        raise ValueError("NUMERIC_TOL_REL must be finite and non-negative")
    if tol_abs == 0 and tol_rel == 0:
        raise ValueError("At least one numerical tolerance must be positive")
    return tol_abs, tol_rel


def validate_runtime_config() -> None:
    global INPUT_TOKEN_LIMIT, MAX_SEQ_LENGTH_OVERRIDE, NUMERIC_BACKEND

    if NUMERIC_BACKEND == "yang-mills":
        NUMERIC_BACKEND = "ym"
    if NUMERIC_BACKEND not in {"sqed", "ym", "gravity"}:
        raise ValueError(
            "NUMERIC_BACKEND must be 'sqed', 'ym', or 'gravity', "
            f"got {NUMERIC_BACKEND!r}"
        )
    if N_PARTICLES < 4:
        raise ValueError("N_PARTICLES must be at least 4 for the configured kinematics generators")
    if NUMERIC_EQUIV_SAMPLES < 1:
        raise ValueError("NUMERIC_EQUIV_SAMPLES must be at least 1")
    if NUMERIC_BACKEND == "sqed":
        if not math.isfinite(NUMERIC_EQUIV_MASS) or NUMERIC_EQUIV_MASS <= 0:
            raise ValueError("NUMERIC_EQUIV_MASS must be finite and positive")
    elif NUMERIC_BACKEND == "ym":
        if (
            not math.isfinite(NUMERIC_EQUIV_ENERGY_SCALE)
            or NUMERIC_EQUIV_ENERGY_SCALE <= 0
        ):
            raise ValueError("NUMERIC_EQUIV_ENERGY_SCALE must be finite and positive")
        if DATA_SOURCE == "generate":
            raise ValueError(
                "DATA_SOURCE='generate' currently creates scalar-QED data; "
                "use a CSV source when NUMERIC_BACKEND='ym'"
            )
    else:
        if N_PARTICLES != 5:
            raise ValueError(
                "The gravity backend supports the repository's five-point "
                f"3s2h/4s1h amplitudes only; got N_PARTICLES={N_PARTICLES}"
            )
        if DATA_SOURCE == "generate":
            raise ValueError(
                "DATA_SOURCE='generate' currently creates scalar-QED data; "
                "use a gravity CSV source"
            )
        if (
            GRAVITY_METADATA_CSV_PATH is not None
            and GRAVITY_PROCESS is not None
        ):
            raise ValueError(
                "Set only one of GRAVITY_METADATA_CSV_PATH and GRAVITY_PROCESS"
            )
        if (
            GRAVITY_PROCESS is not None
            and GRAVITY_PROCESS not in GRAVITY_PROCESS_SPECS
        ):
            raise ValueError(
                f"Unsupported GRAVITY_PROCESS={GRAVITY_PROCESS!r}; expected "
                f"one of {sorted(GRAVITY_PROCESS_SPECS)}"
            )
        resolve_gravity_reference_modes()
    resolve_numeric_pol_modes()
    resolve_numeric_tolerances()

    if SINGLE_AMPLITUDE_INPUT_FORMAT not in {"auto", "tokens", "feyn"}:
        raise ValueError(
            "SINGLE_AMPLITUDE_INPUT_FORMAT must be 'auto', 'tokens', or 'feyn'"
        )
    if SINGLE_AMPLITUDE_EXPRESSION_COLUMN < 0:
        raise ValueError("SINGLE_AMPLITUDE_EXPRESSION_COLUMN must be non-negative")

    if INPUT_TOKEN_LIMIT is not None:
        if not isinstance(INPUT_TOKEN_LIMIT, int):
            raise ValueError("INPUT_TOKEN_LIMIT must be an integer or None")
        if INPUT_TOKEN_LIMIT <= 0:
            INPUT_TOKEN_LIMIT = None
    if MAX_SEQ_LENGTH_OVERRIDE is not None:
        if not isinstance(MAX_SEQ_LENGTH_OVERRIDE, int):
            raise ValueError("MAX_SEQ_LENGTH_OVERRIDE must be an integer or None")
        if MAX_SEQ_LENGTH_OVERRIDE <= 0:
            MAX_SEQ_LENGTH_OVERRIDE = None


def precompute_gravity_kinematics() -> dict[str, list[Any]]:
    """Build reusable complex points matching the dedicated gravity oracle."""

    reference_modes = resolve_gravity_reference_modes()
    cached: dict[str, list[Any]] = {}
    for process, spec in GRAVITY_PROCESS_SPECS.items():
        points: list[Any] = []
        for sample_idx in range(NUMERIC_EQUIV_SAMPLES):
            seed = NUMERIC_EQUIV_SEED + sample_idx
            base = generate_gravity_kinematics(
                seed=seed,
                graviton_legs=spec.graviton_legs,
                reference_mode="cyclic",
            )
            points.extend(
                gravity_with_references(
                    base,
                    spec.graviton_legs,
                    reference_mode=reference_mode,
                    seed=seed + 11,
                )
                for reference_mode in reference_modes
            )
            if GRAVITY_GAUGE_SHIFT:
                shifts = {
                    leg: complex(0.19 * (leg + 1), -0.07 * leg)
                    for leg in spec.graviton_legs
                }
                points.append(
                    gravity_with_references(
                        base,
                        spec.graviton_legs,
                        reference_mode="cyclic",
                        gauge_shifts=shifts,
                    )
                )
        cached[process] = points
    return cached


def precompute_kinematics() -> (
    list[tuple[Any, Any]] | dict[str, list[Any]]
):
    if NUMERIC_BACKEND == "gravity":
        return precompute_gravity_kinematics()

    modes = resolve_numeric_pol_modes()
    cached: list[tuple[Any, Any]] = []
    for pol_mode in modes:
        for sample_idx in range(NUMERIC_EQUIV_SAMPLES):
            seed = NUMERIC_EQUIV_SEED + sample_idx
            if NUMERIC_BACKEND == "ym":
                kinematics = generate_ym_kinematics(
                    N_PARTICLES,
                    E_scale=NUMERIC_EQUIV_ENERGY_SCALE,
                    pol_mode=pol_mode,
                    seed=seed,
                )
            elif NUMERIC_BACKEND == "sqed":
                kinematics = generate_sqed_kinematics(
                    N_PARTICLES,
                    M=NUMERIC_EQUIV_MASS,
                    pol_mode=pol_mode,
                    seed=seed,
                )
            else:
                raise ValueError(
                    f"NUMERIC_BACKEND must be 'sqed' or 'ym', got {NUMERIC_BACKEND!r}"
                )
            cached.append(kinematics)
    return cached


def validate_gravity_expression(expr: str, process: str) -> None:
    """Strictly validate a scalar expression for one gravity process.

    The shared gravity evaluator intentionally retains a permissive historical
    parser. Model predictions need a stricter boundary so unsupported symbols
    and free vectors cannot silently evaluate to numeric zero.
    """

    if process not in GRAVITY_PROCESS_SPECS:
        raise ValueError(f"Unknown gravity process: {process!r}")
    spec = GRAVITY_PROCESS_SPECS[process]
    graviton_legs = set(spec.graviton_legs)
    tokens = gd._strict_tokenize(expr)
    gd._validate_strict_token_sequence(tokens)

    depth = 0
    for token in tokens:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("gravity evaluator: unmatched closing parenthesis")
    if depth:
        raise ValueError("gravity evaluator: unmatched opening parenthesis")

    parser = gd._Parser(tokens)
    tree = parser.parse()
    if parser.i != len(tokens):
        raise ValueError(
            "gravity evaluator: expression was not fully consumed "
            f"({parser.i}/{len(tokens)} tokens)"
        )

    def validate_vector(vector: Any) -> None:
        if vector.tag == "p":
            if 1 <= vector.idx <= 5:
                return
            raise KeyError(f"gravity evaluator: unknown momentum p_{vector.idx}")
        if vector.tag in {"e", "F"}:
            if vector.idx in graviton_legs:
                return
            raise KeyError(
                f"gravity evaluator: {vector.tag}_{vector.idx} is not a "
                f"graviton leg for process {process}"
            )
        raise ValueError(
            f"gravity evaluator: unsupported vector tag {vector.tag!r}"
        )

    def visit(node: Any) -> None:
        if isinstance(node, gd._Num):
            return
        if isinstance(node, gd._UnaryOp):
            if node.op != "-":
                raise ValueError(
                    f"gravity evaluator: unsupported unary operator {node.op!r}"
                )
            visit(node.operand)
            return
        if isinstance(node, gd._BinOp):
            if node.op not in {"+", "-", "*", "/", "**"}:
                raise ValueError(
                    f"gravity evaluator: unsupported operator {node.op!r}"
                )
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, gd._Vec):
            validate_vector(node)
            raise ValueError(
                f"gravity evaluator: free vector {node.tag}_{node.idx} is not a scalar"
            )
        if isinstance(node, gd._DotChain):
            parts = node.parts
            is_trace = bool(parts) and parts[-1] is gd._DotChain._TR
            vectors = parts[:-1] if is_trace else parts
            if not vectors or any(
                not isinstance(vector, gd._Vec) for vector in vectors
            ):
                raise ValueError("gravity evaluator: malformed vector chain")
            for vector in vectors:
                validate_vector(vector)
            if is_trace:
                if len(vectors) < 2 or any(
                    vector.tag != "F" for vector in vectors
                ):
                    raise ValueError(
                        "gravity evaluator: Tr requires at least two F_i factors"
                    )
                return
            if any(vector.tag == "F" for vector in vectors):
                valid_f_chain = (
                    len(vectors) >= 3
                    and vectors[0].tag == "p"
                    and vectors[-1].tag == "p"
                    and all(vector.tag == "F" for vector in vectors[1:-1])
                )
                if not valid_f_chain:
                    raise ValueError(
                        "gravity evaluator: F_i is only valid inside "
                        "p·F...·p or Tr(F...)"
                    )
                return
            if len(vectors) != 2 or any(
                vector.tag not in {"p", "e"} for vector in vectors
            ):
                raise ValueError(
                    "gravity evaluator: a p/e dot product must contain "
                    "exactly two vectors"
                )
            return
        raise ValueError(
            "gravity evaluator: unsupported expression node "
            f"{type(node).__name__}"
        )

    visit(tree)


def eval_numeric_expr(
    expr: str,
    momenta: Any,
    pols: Any,
    *,
    gravity_process: str | None = None,
) -> float | complex:
    if NUMERIC_BACKEND == "gravity":
        if gravity_process is None:
            raise ValueError("Gravity numerical evaluation requires a process")
        validate_gravity_expression(expr, gravity_process)
        return eval_gravity_expression(expr, momenta)
    if NUMERIC_BACKEND == "ym":
        # Strict mode prevents unknown/free model symbols from silently becoming
        # numeric zero and producing false-positive equivalence or reranking.
        return eval_ym_infix_numeric(expr, momenta, pols, strict=True)
    if NUMERIC_BACKEND == "sqed":
        return gd.eval_infix_numeric(expr, momenta, pols, strict=True)
    raise ValueError(f"Unknown numerical backend: {NUMERIC_BACKEND!r}")


def numerically_equivalent_exprs(
    expr_a: str,
    expr_b: str,
    cached_kinematics: list[tuple[Any, Any]] | dict[str, list[Any]],
    *,
    gravity_process: str | None = None,
) -> bool:
    tol_abs, tol_rel = resolve_numeric_tolerances()
    try:
        if NUMERIC_BACKEND == "gravity":
            if gravity_process is None:
                return False
            if not isinstance(cached_kinematics, dict):
                raise TypeError("Gravity kinematics cache must be process-keyed")
            points = cached_kinematics.get(gravity_process)
            if not points:
                raise ValueError(
                    f"No cached gravity kinematics for {gravity_process!r}"
                )
            for kinematics in points:
                val_a = eval_numeric_expr(
                    expr_a,
                    kinematics,
                    None,
                    gravity_process=gravity_process,
                )
                val_b = eval_numeric_expr(
                    expr_b,
                    kinematics,
                    None,
                    gravity_process=gravity_process,
                )
                if not (
                    math.isfinite(abs(val_a))
                    and math.isfinite(abs(val_b))
                ):
                    return False
                if not numeric_values_close(
                    val_a,
                    val_b,
                    tol_abs=tol_abs,
                    tol_rel=tol_rel,
                ):
                    return False
            return True

        if not isinstance(cached_kinematics, list):
            raise TypeError("SQED/Yang-Mills kinematics cache must be a list")
        for momenta, pols in cached_kinematics:
            val_a = eval_numeric_expr(expr_a, momenta, pols)
            val_b = eval_numeric_expr(expr_b, momenta, pols)
            if not (
                math.isfinite(abs(val_a))
                and math.isfinite(abs(val_b))
            ):
                return False
            if not numeric_values_close(
                val_a,
                val_b,
                tol_abs=tol_abs,
                tol_rel=tol_rel,
            ):
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
        # The evaluator's input cap applies only to scrambled/source tokens.
        # build_dataset's max_tokens also caps the simple target, so filter the
        # source explicitly below instead of using that combined budget.
        max_tokens=None,
        tokenizer_max_particles=TOKENIZER_MAX_PARTICLES,
    )
    pairs, removed = gd.dedupe_pairs(pairs)
    input_limit_removed = 0
    if INPUT_TOKEN_LIMIT is not None:
        tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=TOKENIZER_MAX_PARTICLES,
            max_sequence_length=None,
        )
        limited_pairs = [
            pair
            for pair in pairs
            if len(tokenizer.encode_infix(pair[1])) <= INPUT_TOKEN_LIMIT
        ]
        input_limit_removed = len(pairs) - len(limited_pairs)
        pairs = limited_pairs
        if not pairs:
            raise ValueError(
                "No generated examples fit "
                f"INPUT_TOKEN_LIMIT={INPUT_TOKEN_LIMIT}; raise the limit or generate again"
            )
        if input_limit_removed and len(pairs) < NUM_SAMPLES:
            raise ValueError(
                f"Only {len(pairs)} / {NUM_SAMPLES} generated examples fit "
                f"INPUT_TOKEN_LIMIT={INPUT_TOKEN_LIMIT}; raise the limit rather than "
                "evaluating a silently reduced test set"
            )
    pairs = pairs[:NUM_SAMPLES]
    gd.write_csv(pairs, str(RAW_CSV_PATH))
    gd.tokenise_csv(
        str(RAW_CSV_PATH),
        str(TOK_CSV_PATH),
        max_particles=TOKENIZER_MAX_PARTICLES,
        max_sequence_length=None,
    )
    print(f"Wrote raw data to {RAW_CSV_PATH}")
    print(f"Wrote tokenised data to {TOK_CSV_PATH}")
    print(f"Dedupe removed {removed} duplicate pairs")
    if INPUT_TOKEN_LIMIT is not None:
        print(f"Input-token limit removed {input_limit_removed} generated pairs")


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
        max_sequence_length=None,
    )

    with GEN_LOG_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"# imported raw CSV: {source_path}\n")
        handle.write(f"# rows_read={original_count} rows_used={len(pairs)} dedupe_removed={removed}\n")

    print(f"Imported {len(pairs)} / {original_count} rows")
    print(f"Wrote raw data to {RAW_CSV_PATH}")
    print(f"Wrote tokenised data to {TOK_CSV_PATH}")
    if EXISTING_CSV_DEDUPE:
        print(f"Dedupe removed {removed} duplicate pairs")


def detect_single_amplitude_input_format(source_path: Path) -> str:
    if SINGLE_AMPLITUDE_INPUT_FORMAT != "auto":
        return SINGLE_AMPLITUDE_INPUT_FORMAT

    with open_csv_text(source_path, "r") as handle:
        first_row = next(csv.reader(handle), None)
    if first_row is None:
        raise ValueError(f"{source_path} is empty")
    return "tokens" if SINGLE_AMPLITUDE_TOKENS_COLUMN in first_row else "feyn"


def write_token_rows(path: Path, rows: list[dict[str, list[int]]]) -> None:
    with open_csv_text(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["simple", "scrambled"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "simple": json.dumps(row["simple"]),
                    "scrambled": json.dumps(row["scrambled"]),
                }
            )


def tokenise_feyn_single_amplitudes(
    source_path: Path,
) -> tuple[list[dict[str, list[int]]], list[str]]:
    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=TOKENIZER_MAX_PARTICLES,
        max_sequence_length=None,
    )
    token_rows: list[dict[str, list[int]]] = []
    source_exprs: list[str] = []

    with open_csv_text(source_path, "r") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if SINGLE_AMPLITUDE_EXPRESSION_COLUMN >= len(row):
                raise ValueError(
                    f"{source_path}:{row_number} has no expression at zero-based "
                    f"column {SINGLE_AMPLITUDE_EXPRESSION_COLUMN}"
                )
            expr = row[SINGLE_AMPLITUDE_EXPRESSION_COLUMN].strip()
            if not expr:
                raise ValueError(f"{source_path}:{row_number} has an empty expression")
            tokens = tokenizer.encode_infix(expr)
            if not tokens:
                raise ValueError(
                    f"{source_path}:{row_number} did not contain a tokenisable expression"
                )
            if INPUT_TOKEN_LIMIT is not None and len(tokens) > INPUT_TOKEN_LIMIT:
                raise ValueError(
                    f"{source_path}:{row_number} has {len(tokens)} input content tokens, "
                    f"exceeding --input-token-limit={INPUT_TOKEN_LIMIT}"
                )
            token_rows.append({"simple": tokens, "scrambled": tokens.copy()})
            source_exprs.append(expr)

    if not token_rows:
        raise ValueError(f"{source_path} contains no amplitude rows")
    write_token_rows(TOK_CSV_PATH, token_rows)
    return token_rows, source_exprs


def import_single_amplitude_test_data() -> None:
    if SINGLE_AMPLITUDE_INPUT_CSV_PATH is None:
        raise ValueError(
            "DATA_SOURCE='single-amplitude' requires "
            "--single-amplitude-input-csv (or SINGLE_AMPLITUDE_INPUT_CSV_PATH)"
        )

    source_path = resolve_input_path(SINGLE_AMPLITUDE_INPUT_CSV_PATH)
    source_resolved = source_path.resolve()
    preparation_outputs = {
        RAW_CSV_PATH.resolve(),
        TOK_CSV_PATH.resolve(),
        GEN_LOG_PATH.resolve(),
        SUMMARY_CSV_PATH.resolve(),
    }
    if source_resolved in preparation_outputs:
        raise ValueError(
            "The single-amplitude input must be different from every evaluation output file"
        )
    if TOK_CSV_PATH.resolve() in {
        RAW_CSV_PATH.resolve(),
        GEN_LOG_PATH.resolve(),
        SUMMARY_CSV_PATH.resolve(),
    }:
        raise ValueError(
            "The normalized single-amplitude token output conflicts with another "
            "evaluation output path"
        )

    input_format = detect_single_amplitude_input_format(source_path)
    print(f"Using single-amplitude {input_format} data from {source_path}")
    source_exprs: list[str] | None = None
    if input_format == "tokens":
        converted_count = convert_single_amplitude_file(
            source_path,
            TOK_CSV_PATH,
            tokens_column=SINGLE_AMPLITUDE_TOKENS_COLUMN,
        )
        token_rows = load_token_rows(TOK_CSV_PATH)
    else:
        token_rows, source_exprs = tokenise_feyn_single_amplitudes(source_path)
        converted_count = len(token_rows)

    if not token_rows:
        raise ValueError(f"{source_path} contains no data rows")
    if len(token_rows) != converted_count:
        raise ValueError(
            "Converted single-amplitude row count does not match the normalized token CSV"
        )

    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=TOKENIZER_MAX_PARTICLES,
        max_sequence_length=None,
    )
    raw_pairs: list[tuple[str, str]] = []
    special_ids = set(SPECIAL_TOKENS.values()) | {1}
    valid_ids = set(tokenizer.id_to_token)

    for row_index, row in enumerate(token_rows):
        row_number = row_index + (1 if input_format == "feyn" else 2)
        tokens = row["simple"]
        if not tokens:
            raise ValueError(f"{source_path}:{row_number} has an empty token list")
        if any(type(token) is not int for token in tokens):
            raise ValueError(
                f"{source_path}:{row_number} token list must contain integers, not booleans"
            )

        invalid_ids = sorted(set(tokens) - valid_ids)
        if invalid_ids:
            raise ValueError(
                f"{source_path}:{row_number} contains token IDs outside the configured "
                f"vocabulary: {invalid_ids}. Check --tokenizer-max-particles."
            )

        embedded_special_ids = sorted(set(tokens) & special_ids)
        if embedded_special_ids:
            raise ValueError(
                f"{source_path}:{row_number} contains special token IDs "
                f"{embedded_special_ids}; source rows must exclude PAD/UNK/BOS/EOS"
            )

        if INPUT_TOKEN_LIMIT is not None and len(tokens) > INPUT_TOKEN_LIMIT:
            raise ValueError(
                f"{source_path}:{row_number} has {len(tokens)} input content tokens, "
                f"exceeding --input-token-limit={INPUT_TOKEN_LIMIT}"
            )

        decode_ok, expr, decode_error = safe_decode_infix(tokenizer, tokens)
        if not decode_ok:
            raise ValueError(
                f"{source_path}:{row_number} is not a valid prefix expression: {decode_error}"
            )
        source_expr = source_exprs[row_index] if source_exprs is not None else expr
        raw_pairs.append((source_expr, source_expr))

    # Both input layouts are now normalized to the exact simple=scrambled token
    # structure expected by TransformerDataset. The raw copy supplies infix
    # expressions for numerical metrics and readable output.
    gd.write_csv(raw_pairs, str(RAW_CSV_PATH))
    with GEN_LOG_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"# imported single-amplitude CSV: {source_path}\n")
        handle.write(f"# input_format={input_format}\n")
        handle.write(f"# tokens_column={SINGLE_AMPLITUDE_TOKENS_COLUMN}\n")
        handle.write(
            f"# expression_column={SINGLE_AMPLITUDE_EXPRESSION_COLUMN}\n"
        )
        handle.write(f"# rows_used={len(token_rows)}\n")

    print(f"Imported {len(token_rows)} token rows")
    print(f"Wrote raw expression data to {RAW_CSV_PATH}")
    print(f"Wrote normalized token data to {TOK_CSV_PATH}")


def prepare_test_data() -> None:
    if DATA_SOURCE == "generate":
        generate_test_data()
    elif DATA_SOURCE == "csv":
        import_existing_test_data()
    elif DATA_SOURCE == "single-amplitude":
        import_single_amplitude_test_data()
    else:
        raise ValueError(
            "DATA_SOURCE must be 'generate', 'csv', or 'single-amplitude', "
            f"got {DATA_SOURCE!r}"
        )


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
    return max(
        max(len(seq) for seq in dataset.simple_sequences),
        max(len(seq) for seq in dataset.scrambled_sequences),
    )


def positional_encoding_capacity(model: Any, attribute: str) -> int:
    encoding = getattr(model, attribute, None)
    pe = getattr(encoding, "pe", None)
    if pe is not None:
        if hasattr(pe, "num_embeddings"):
            return int(pe.num_embeddings)
        shape = getattr(pe, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
    capacity = getattr(model, "max_seq_len", None)
    if capacity is None:
        raise ValueError(
            f"Cannot determine positional capacity from model.{attribute} or model.max_seq_len"
        )
    return int(capacity)


def validate_model_sequence_capacity(
    model: Any,
    dataset: TransformerDataset,
) -> None:
    source_capacity = positional_encoding_capacity(model, "src_pos_encoding")
    target_capacity = positional_encoding_capacity(model, "tgt_pos_encoding")
    longest_source_length = max(len(seq) for seq in dataset.scrambled_sequences)

    if longest_source_length > source_capacity:
        raise ValueError(
            f"The longest source sequence is {longest_source_length} tokens after "
            f"BOS/EOS, but the checkpoint supports only {source_capacity}. "
            "Unlimited input removes the evaluator cap but cannot exceed the model's "
            "positional-encoding capacity."
        )

    for decode_cfg in DECODE_RUNS:
        if not decode_cfg.enabled:
            continue
        max_decode_length = get_max_decode_length(dataset, decode_cfg)
        if max_decode_length < 2:
            raise ValueError(
                f"Decode mode {decode_cfg.name!r} needs max_length >= 2, "
                f"got {max_decode_length}"
            )
        if max_decode_length > target_capacity:
            raise ValueError(
                f"Decode mode {decode_cfg.name!r} requests {max_decode_length} tokens, "
                f"but the checkpoint target positional capacity is {target_capacity}"
            )


def validate_numeric_reference_rows(
    raw_rows: list[dict[str, str]],
    cached_kinematics: list[tuple[Any, Any]] | dict[str, list[Any]],
    gravity_processes: list[str | None] | None = None,
) -> None:
    """Fail fast when the chosen backend/N cannot evaluate the reference data."""
    if not cached_kinematics:
        raise ValueError("No numerical kinematics were prepared")
    if gravity_processes is None:
        gravity_processes = [None] * len(raw_rows)
    if len(gravity_processes) != len(raw_rows):
        raise ValueError(
            "Gravity process labels and reference rows have different lengths"
        )

    if NUMERIC_BACKEND != "gravity":
        if not isinstance(cached_kinematics, list):
            raise TypeError("SQED/Yang-Mills kinematics cache must be a list")
        momenta, pols = cached_kinematics[0]

    seen: set[str] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        gravity_process = gravity_processes[row_number - 2]
        if NUMERIC_BACKEND == "gravity":
            if gravity_process is None:
                raise ValueError(
                    f"Missing gravity process for reference row {row_number}"
                )
            if not isinstance(cached_kinematics, dict):
                raise TypeError("Gravity kinematics cache must be process-keyed")
            process_points = cached_kinematics.get(gravity_process)
            if not process_points:
                raise ValueError(
                    f"No cached gravity kinematics for {gravity_process!r}"
                )
            momenta, pols = process_points[0], None

        for column in ("simple", "scrambled"):
            expr = (row.get(column) or "").strip()
            seen_key = (
                f"{gravity_process}:{expr}"
                if NUMERIC_BACKEND == "gravity"
                else expr
            )
            if seen_key in seen:
                continue
            seen.add(seen_key)
            if NUMERIC_BACKEND == "sqed":
                endpoint_gluon = re.search(
                    rf"\b(?:e|F)_(?:1|{N_PARTICLES})\b",
                    expr,
                )
                if endpoint_gluon:
                    raise ValueError(
                        f"Reference {column} expression at "
                        f"{RAW_CSV_PATH}:{row_number} uses "
                        f"{endpoint_gluon.group(0)}, but SQED treats legs 1 and "
                        f"{N_PARTICLES} as scalar endpoints. This appears to be "
                        "all-gluon data; use --numeric-backend ym."
                    )
            try:
                value = eval_numeric_expr(
                    expr,
                    momenta,
                    pols,
                    gravity_process=gravity_process,
                )
            except Exception as exc:
                raise ValueError(
                    f"Reference {column} expression at {RAW_CSV_PATH}:{row_number} "
                    f"cannot be evaluated with numeric_backend={NUMERIC_BACKEND!r}, "
                    f"N_PARTICLES={N_PARTICLES}: {type(exc).__name__}: {exc}. "
                    "Check --numeric-backend and --n-particles."
                ) from exc
            if not math.isfinite(abs(value)):
                raise ValueError(
                    f"Reference {column} expression at {RAW_CSV_PATH}:{row_number} "
                    "evaluated to a non-finite value on the validation kinematics"
                )


def summarize_mode(rows: list[dict[str, Any]], mode_name: str) -> dict[str, Any]:
    total = len(rows)
    tol_abs, tol_rel = resolve_numeric_tolerances()
    summary = {
        "mode": mode_name,
        "numeric_backend": NUMERIC_BACKEND,
        "numeric_pol_modes": ",".join(resolve_numeric_pol_modes()),
        "gravity_reference_modes": (
            ",".join(resolve_gravity_reference_modes())
            if NUMERIC_BACKEND == "gravity"
            else ""
        ),
        "gravity_gauge_shift": (
            int(GRAVITY_GAUGE_SHIFT)
            if NUMERIC_BACKEND == "gravity"
            else ""
        ),
        "process_counts": json.dumps(
            {
                process: sum(
                    int(row.get("process") == process) for row in rows
                )
                for process in sorted(
                    {
                        str(row.get("process"))
                        for row in rows
                        if row.get("process")
                    }
                )
            },
            sort_keys=True,
        ),
        "numeric_tol_abs": tol_abs,
        "numeric_tol_rel": tol_rel,
        "input_token_limit": (
            "unlimited" if INPUT_TOKEN_LIMIT is None else INPUT_TOKEN_LIMIT
        ),
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
        "valid_fallback_replacements": sum(
            int(r.get("valid_fallback_replaced_top1", 0)) for r in rows
        ),
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
    if summary["valid_fallback_replacements"]:
        print(
            f"  valid-expression fallbacks  : {summary['valid_fallback_replacements']} / {total}"
        )


def print_simple_summary(summary: dict[str, Any]) -> None:
    total = summary["total_examples"]

    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%" if total else "n/a"

    top1 = summary["top1_num_eq_scrambled"]
    any_beam = summary["any_beam_num_eq_scrambled"]
    rerank = summary["reranked_top1_replacements"]
    valid_fallbacks = summary["valid_fallback_replacements"]
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
    if valid_fallbacks:
        print(f"  valid fallbacks : {valid_fallbacks:4d} / {total}  ({pct(valid_fallbacks)})")
    print(f"  avg valid beams : {avg_valid:.1f}")
    if avg_pred_tok > 0:
        print(f"  avg tokens (correct):  pred {avg_pred_tok:.1f}  vs  scrambled {avg_scr_tok:.1f}")


def evaluate_mode(
    model,
    tokenizer: ScatteringAmplitudeTokenizer,
    dataset: TransformerDataset,
    raw_rows: list[dict[str, str]],
    token_rows: list[dict[str, list[int]]],
    cached_kinematics: list[tuple[Any, Any]] | dict[str, list[Any]],
    gravity_processes: list[str | None],
    decode_cfg: DecodeConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(gravity_processes) != len(raw_rows):
        raise ValueError(
            "Gravity process labels and evaluation rows have different lengths"
        )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dynamic_pad_collate,
    )
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
            gravity_process = gravity_processes[row_idx]

            top1_full = clean_seq(decoded[i].tolist(), pad_token=SPECIAL_TOKENS["pad"], eos_token=SPECIAL_TOKENS["eos"])
            top1_tokens = strip_special_tokens(top1_full)

            original_top1_decode_ok, original_pred_expr, original_pred_decode_error = safe_decode_infix(
                tokenizer,
                top1_tokens,
            )
            target_decode_ok, target_simple_decoded, _ = safe_decode_infix(tokenizer, target_simple_tokens)
            original_top1_num_eq_simple = (
                numerically_equivalent_exprs(
                    original_pred_expr,
                    target_simple_expr,
                    cached_kinematics,
                    gravity_process=gravity_process,
                )
                if original_top1_decode_ok
                else False
            )
            original_top1_num_eq_scrambled = (
                numerically_equivalent_exprs(
                    original_pred_expr,
                    target_scrambled_expr,
                    cached_kinematics,
                    gravity_process=gravity_process,
                )
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
                    if numerically_equivalent_exprs(
                        cand_expr,
                        target_simple_expr,
                        cached_kinematics,
                        gravity_process=gravity_process,
                    ):
                        cand_num_eq_simple = True
                        num_eq_simple_count += 1
                    if numerically_equivalent_exprs(
                        cand_expr,
                        target_scrambled_expr,
                        cached_kinematics,
                        gravity_process=gravity_process,
                    ):
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

            selected, selection_reason = select_candidate_for_reporting(
                candidate_records,
                rerank_numerical_equiv=decode_cfg.rerank_numerical_equiv,
            )

            top1_tokens = selected["tokens"]
            top1_decode_ok = selected["decode_ok"]
            pred_expr = selected["expr"]
            pred_decode_error = selected["decode_error"]
            pred_display = prediction_text_for_display(tokenizer, selected)
            top1_exact_token_match = selected["exact_token"]
            top1_exact_string_match = selected["exact_string"]
            top1_num_eq_simple = selected["num_eq_simple"]
            top1_num_eq_scrambled = selected["num_eq_scrambled"]
            selection_replaced_top1 = int(selected["index"] != 0)
            rerank_replaced_top1 = int(
                selection_replaced_top1
                and selection_reason == "numerically_equivalent_rerank"
            )
            valid_fallback_replaced_top1 = int(
                selection_replaced_top1
                and selection_reason == "valid_decode_fallback"
            )
            original_top1_display = prediction_text_for_display(
                tokenizer,
                candidate_records[0],
            )

            detail_rows.append(
                {
                    "row_id": row_idx,
                    "mode": decode_cfg.name,
                    "numeric_backend": NUMERIC_BACKEND,
                    "process": gravity_process or "",
                    "input_scrambled": target_scrambled_expr,
                    "target_simple": target_simple_expr,
                    "target_simple_token_count": len(target_simple_tokens),
                    "target_scrambled_token_count": len(target_scrambled_tokens),
                    "top1_prediction_expr": pred_expr,
                    "top1_prediction_display": pred_display,
                    "top1_prediction_tokens": json.dumps(top1_tokens),
                    "top1_prediction_token_count": len(top1_tokens),
                    "top1_decode_ok": int(top1_decode_ok),
                    "top1_decode_error": pred_decode_error or "",
                    "top1_exact_token_match": int(top1_exact_token_match),
                    "top1_exact_string_match": int(top1_exact_string_match),
                    "top1_num_eq_simple": int(top1_num_eq_simple),
                    "top1_num_eq_scrambled": int(top1_num_eq_scrambled),
                    "selection_reason": selection_reason,
                    "selection_replaced_top1": selection_replaced_top1,
                    "rerank_numerical_equiv": int(decode_cfg.rerank_numerical_equiv),
                    "rerank_replaced_top1": rerank_replaced_top1,
                    "valid_fallback_replaced_top1": valid_fallback_replaced_top1,
                    "rerank_selected_candidate_index": selected["index"],
                    "original_top1_prediction_expr": original_pred_expr,
                    "original_top1_prediction_display": original_top1_display,
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
        prediction_source      - model top-1, numerical rerank, or valid fallback
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
        "prediction_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "simple": row["target_simple"],
                "scrambled": row["input_scrambled"],
                "top_pred": row.get(
                    "top1_prediction_display",
                    row["top1_prediction_expr"],
                ),
                "correct": "yes" if int(row["top1_num_eq_scrambled"]) else "no",
                "any_correct": "yes" if int(row["any_beam_num_eq_scrambled"]) else "no",
                "candidates_checked": row["candidate_sequences_checked"],
                "valid_candidates": row["candidate_valid_decode_count"],
                "pred_tokens": row["top1_prediction_token_count"],
                "scrambled_tokens": row["target_scrambled_token_count"],
                "decode_ok": "yes" if int(row["top1_decode_ok"]) else "no",
                "prediction_source": row.get("selection_reason", "model_top1"),
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
        top1_display = row.get(
            "top1_prediction_display",
            row["top1_prediction_expr"],
        )
        print(f"    top1 pred     : {top1_display[:180]}")
        selection_reason = row.get("selection_reason", "model_top1")
        if selection_reason != "model_top1":
            print(f"    selection     : {selection_reason}")
        if int(row.get("selection_replaced_top1", row.get("rerank_replaced_top1", 0))):
            original_display = row.get(
                "original_top1_prediction_display",
                row["original_top1_prediction_expr"],
            )
            print(f"    original top1 : {original_display[:180]}")
        if not int(row["top1_decode_ok"]):
            print(f"    decode error  : {row['top1_decode_error'][:180]}")
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


def _run_evaluation() -> None:
    args = parse_args()
    apply_cli_config(args)
    validate_runtime_config()

    t0 = time.perf_counter()
    ensure_output_dirs()

    device = resolve_device()
    print(f"Using device: {device}")

    prepare_test_data()
    raw_rows = load_raw_rows(RAW_CSV_PATH)
    token_rows = load_token_rows(TOK_CSV_PATH)
    if len(raw_rows) != len(token_rows):
        raise ValueError("Raw and tokenised row counts do not match")
    gravity_processes = resolve_gravity_processes(raw_rows)
    max_input_tokens = validate_input_token_rows(token_rows)

    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=TOKENIZER_MAX_PARTICLES,
        max_sequence_length=None,
    )
    # Input limits are validated explicitly above. Dynamic padding keeps source
    # and target lengths separate and avoids silently truncating either sequence.
    dataset = TransformerDataset(
        str(TOK_CSV_PATH),
        max_length=None,
        dynamic_padding=True,
    )

    pol_modes = resolve_numeric_pol_modes()
    tol_abs, tol_rel = resolve_numeric_tolerances()
    input_limit_text = "unlimited" if INPUT_TOKEN_LIMIT is None else str(INPUT_TOKEN_LIMIT)
    print(
        f"Input token limit: {input_limit_text}; longest input: "
        f"{max_input_tokens} content / {max_input_tokens + 2} with BOS/EOS"
    )
    if NUMERIC_BACKEND == "gravity":
        process_counts = {
            process: gravity_processes.count(process)
            for process in sorted(GRAVITY_PROCESS_SPECS)
            if process in gravity_processes
        }
        print(
            "Numerical evaluator: gravity "
            f"(processes={process_counts}; "
            f"reference_modes={','.join(resolve_gravity_reference_modes())}; "
            f"gauge_shift={GRAVITY_GAUGE_SHIFT}; "
            f"tol_abs={tol_abs:g}; tol_rel={tol_rel:g})"
        )
    else:
        scale_name = "mass" if NUMERIC_BACKEND == "sqed" else "energy_scale"
        scale_value = (
            NUMERIC_EQUIV_MASS
            if NUMERIC_BACKEND == "sqed"
            else NUMERIC_EQUIV_ENERGY_SCALE
        )
        print(
            f"Numerical evaluator: {NUMERIC_BACKEND} "
            f"({scale_name}={scale_value}; pol_modes={','.join(pol_modes)}; "
            f"tol_abs={tol_abs:g}; tol_rel={tol_rel:g})"
        )
    cached_kinematics = precompute_kinematics()
    validate_numeric_reference_rows(
        raw_rows,
        cached_kinematics,
        gravity_processes,
    )
    model = load_model(device)
    validate_model_sequence_capacity(model, dataset)

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
            gravity_processes,
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


def main() -> int:
    """Run the evaluator while mirroring all Python terminal output to a PDF."""

    started_at = datetime.now().astimezone()
    exit_code = 0
    pdf_available = False

    with capture_terminal_output() as transcript:
        try:
            require_reportlab()
            pdf_available = True
            _run_evaluation()
        except SystemExit as exc:
            if exc.code is None:
                exit_code = 0
            elif isinstance(exc.code, int):
                exit_code = exc.code
            else:
                print(str(exc.code), file=sys.stderr)
                exit_code = 1
        except KeyboardInterrupt:
            traceback.print_exc()
            exit_code = 130
        except BaseException:
            traceback.print_exc()
            exit_code = 1

        if pdf_available:
            pdf_path = evaluation_pdf_path(started_at)
            print(f"\nTerminal output PDF: {pdf_path}")
            try:
                write_terminal_output_pdf(
                    pdf_path,
                    transcript.getvalue(),
                    started_at,
                )
            except BaseException:
                print(
                    f"Failed to write terminal output PDF to {pdf_path}:",
                    file=sys.stderr,
                )
                traceback.print_exc()
                if exit_code == 0:
                    exit_code = 1
        else:
            print(
                "Terminal output PDF was not created because ReportLab is unavailable.",
                file=sys.stderr,
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
