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

MODEL_PATH = ROOT / "models" / "twofiddytwo" / "best_model.pt"
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
EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "sqed_w0110_mM00M_den6_maxD2_20000_phys_oneshot.csv" # Paolo's 20k dataset
#EXISTING_RAW_CSV_PATH = DATA_STORAGE_DIR / "gi_4pt_oneshot_5k_val.csv" # Nathan's 5k dataset

# Optional row cap for imported CSVs. None evaluates every row in the file.
EXISTING_CSV_MAX_ROWS = 1000

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

# Numeric equivalence check for model predictions. These samples are independent of generation.
NUMERIC_EQUIV_SAMPLES = 3
NUMERIC_EQUIV_SEED = 151
NUMERIC_EQUIV_MASS = 2.0
# Equivalence passes if either absolute or relative tolerance is met on every numeric sample.
NUMERIC_TOL_ABS = 1e-12
NUMERIC_TOL_REL = 1e-10


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
    beam_size: int = 4
    # Nucleus sampling cutoff and temperature; ignored by greedy/beam.
    p_nucleus: float = 0.95
    temperature_nucleus: float = 1.0
    # When true, score every returned hypothesis, not just the top-1 output.
    evaluate_beam_hypotheses: bool = False
    # Optional cap on checked hypotheses; None checks all returned hypotheses.
    max_beams_to_check: int | None = None


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
        beam_size=4,
        evaluate_beam_hypotheses=True,
        max_beams_to_check=None,
    ),
    DecodeConfig(
        name="nucleus",
        enabled=False,
        decoding_method="nucleus",
        beam_size=4,
        p_nucleus=0.99,
        temperature_nucleus=1.0,
        evaluate_beam_hypotheses=True,
        max_beams_to_check=None,
    ),
]


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
    )
    pairs, removed = gd.dedupe_pairs(pairs)
    pairs = pairs[:NUM_SAMPLES]
    gd.write_csv(pairs, str(RAW_CSV_PATH))
    gd.tokenise_csv(str(RAW_CSV_PATH), str(TOK_CSV_PATH), max_particles=TOKENIZER_MAX_PARTICLES)
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
    gd.tokenise_csv(str(RAW_CSV_PATH), str(TOK_CSV_PATH), max_particles=TOKENIZER_MAX_PARTICLES)

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
    }
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
            top1_exact_token_match = top1_tokens == target_simple_tokens

            top1_decode_ok, pred_expr, pred_decode_error = safe_decode_infix(tokenizer, top1_tokens)
            target_decode_ok, target_simple_decoded, _ = safe_decode_infix(tokenizer, target_simple_tokens)
            top1_exact_string_match = (
                top1_decode_ok and target_decode_ok and pred_expr == target_simple_decoded
            )
            top1_num_eq_simple = (
                numerically_equivalent_exprs(pred_expr, target_simple_expr, cached_kinematics)
                if top1_decode_ok
                else False
            )
            top1_num_eq_scrambled = (
                numerically_equivalent_exprs(pred_expr, target_scrambled_expr, cached_kinematics)
                if top1_decode_ok
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

            for candidate in candidate_sequences:
                exact_token = candidate == target_simple_tokens
                if exact_token:
                    exact_token_count += 1

                cand_decode_ok, cand_expr, _ = safe_decode_infix(tokenizer, candidate)
                if cand_decode_ok:
                    candidate_valid_decode_count += 1
                    candidate_exprs.append(cand_expr)
                    if target_decode_ok and cand_expr == target_simple_decoded:
                        exact_string_count += 1
                    if numerically_equivalent_exprs(cand_expr, target_simple_expr, cached_kinematics):
                        num_eq_simple_count += 1
                    if numerically_equivalent_exprs(cand_expr, target_scrambled_expr, cached_kinematics):
                        num_eq_scrambled_count += 1

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


def print_examples(detail_rows: list[dict[str, Any]], count: int) -> None:
    if not detail_rows:
        return
    print(f"\nExample predictions ({min(count, len(detail_rows))} rows):")
    for row in detail_rows[:count]:
        print(f"  row {row['row_id']} [{row['mode']}]")
        print(f"    target simple : {row['target_simple'][:180]}")
        print(f"    input scram   : {row['input_scrambled'][:180]}")
        print(f"    top1 pred     : {row['top1_prediction_expr'][:180]}")
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
        print_summary(summary)
        print_examples(detail_rows, PRINT_EXAMPLES)
        print(f"  wrote detail results to {detail_path}")
        summary_rows.append(summary)

    write_summary_csv(SUMMARY_CSV_PATH, summary_rows)
    print(f"\nWrote summary to {SUMMARY_CSV_PATH}")
    print(f"Total wall time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
