#!/usr/bin/env python3
"""
Compare multiple trained transformer checkpoints on one shared freshly
generated dataset.

All configuration lives in this file. The script:

1. Generates one fresh dataset on disk.
2. Tokenises that dataset on disk.
3. Loads each configured model checkpoint in turn.
4. Evaluates each configured decoding mode for each model.
5. Writes per-model per-mode detail CSVs and one collated comparison CSV.

This script intentionally reuses the single-model evaluator module so the
metrics stay identical between "one model" and "many models" runs.
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_model_on_generated_data as base


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_TESTING_DIR = ROOT / "data_testing"
OUTPUT_SUBDIR = "outputs"

COMPARE_NAME = "model_compare_4pt"
COMPARE_SUMMARY_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{COMPARE_NAME}_comparison_summary.csv"

DATA_FILENAME_STEM = f"{COMPARE_NAME}_shared_data"
N_PARTICLES = 4
NUM_SAMPLES = 16
GENERATION_SEED = 123
TOKENIZER_MAX_PARTICLES = 8

GEN_MIN_TERMS = 1
GEN_MAX_TERMS = 2
GEN_MIN_SCRAMBLES = 1
GEN_MAX_SCRAMBLES = 3
GEN_VALIDATE = True
GEN_MASS = 2.0

BATCH_SIZE = 8
MAX_SEQ_LENGTH_OVERRIDE = None
PRINT_EXAMPLES_PER_MODE = 1

NUMERIC_EQUIV_SAMPLES = 3
NUMERIC_EQUIV_SEED = 42
NUMERIC_EQUIV_MASS = 2.0
NUMERIC_TOL_ABS = 1e-12
NUMERIC_TOL_REL = 1e-10

DEVICE = "auto"  # "auto", "cpu", "cuda"


@dataclass(frozen=True)
class ModelRun:
    name: str
    model_path: Path
    enabled: bool = True


MODEL_RUNS: list[ModelRun] = [
    ModelRun(
        name="best_model_iter",
        model_path=ROOT / "models" / "best_model_iter.pt",
        enabled=True,
    ),
    ModelRun(
        name="best_model_oneshot",
        model_path=ROOT / "models" / "best_model_oneshot.pt",
        enabled=True,
    ),
]


DECODE_RUNS: list[base.DecodeConfig] = [
    base.DecodeConfig(
        name="greedy",
        enabled=True,
        decoding_method="greedy",
        evaluate_beam_hypotheses=False,
    ),
    base.DecodeConfig(
        name="beam",
        enabled=True,
        decoding_method="beam",
        beam_size=4,
        evaluate_beam_hypotheses=True,
        max_beams_to_check=None,
    ),
    base.DecodeConfig(
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
# Config sync
# ============================================================================


def sync_base_config() -> None:
    base.DATA_TESTING_DIR = DATA_TESTING_DIR
    base.OUTPUT_SUBDIR = OUTPUT_SUBDIR
    base.DEVICE = DEVICE

    base.N_PARTICLES = N_PARTICLES
    base.NUM_SAMPLES = NUM_SAMPLES
    base.GENERATION_SEED = GENERATION_SEED
    base.TOKENIZER_MAX_PARTICLES = TOKENIZER_MAX_PARTICLES

    base.DATA_FILENAME_STEM = DATA_FILENAME_STEM
    base.RAW_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}.csv"
    base.TOK_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_tok.csv"
    base.GEN_LOG_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}.log"
    base.SUMMARY_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_unused_summary.csv"

    base.GEN_MIN_TERMS = GEN_MIN_TERMS
    base.GEN_MAX_TERMS = GEN_MAX_TERMS
    base.GEN_MIN_SCRAMBLES = GEN_MIN_SCRAMBLES
    base.GEN_MAX_SCRAMBLES = GEN_MAX_SCRAMBLES
    base.GEN_VALIDATE = GEN_VALIDATE
    base.GEN_MASS = GEN_MASS

    base.BATCH_SIZE = BATCH_SIZE
    base.MAX_SEQ_LENGTH_OVERRIDE = MAX_SEQ_LENGTH_OVERRIDE
    base.PRINT_EXAMPLES = PRINT_EXAMPLES_PER_MODE

    base.NUMERIC_EQUIV_SAMPLES = NUMERIC_EQUIV_SAMPLES
    base.NUMERIC_EQUIV_SEED = NUMERIC_EQUIV_SEED
    base.NUMERIC_EQUIV_MASS = NUMERIC_EQUIV_MASS
    base.NUMERIC_TOL_ABS = NUMERIC_TOL_ABS
    base.NUMERIC_TOL_REL = NUMERIC_TOL_REL

    base.DECODE_RUNS = DECODE_RUNS


# ============================================================================
# Output helpers
# ============================================================================


def write_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def augment_summary(
    summary: dict[str, object],
    *,
    model_name: str,
    model_path: Path,
) -> dict[str, object]:
    total = int(summary["total_examples"])
    out = {
        "model_name": model_name,
        "model_path": str(model_path),
        **summary,
    }
    if total > 0:
        out["top1_exact_token_rate"] = float(summary["top1_exact_token_matches"]) / total
        out["top1_exact_string_rate"] = float(summary["top1_exact_string_matches"]) / total
        out["top1_num_eq_simple_rate"] = float(summary["top1_num_eq_simple"]) / total
        out["top1_num_eq_scrambled_rate"] = float(summary["top1_num_eq_scrambled"]) / total
        out["any_beam_exact_token_rate"] = float(summary["any_beam_exact_token_matches"]) / total
        out["any_beam_exact_string_rate"] = float(summary["any_beam_exact_string_matches"]) / total
        out["any_beam_num_eq_simple_rate"] = float(summary["any_beam_num_eq_simple"]) / total
        out["any_beam_num_eq_scrambled_rate"] = float(summary["any_beam_num_eq_scrambled"]) / total
    else:
        out["top1_exact_token_rate"] = 0.0
        out["top1_exact_string_rate"] = 0.0
        out["top1_num_eq_simple_rate"] = 0.0
        out["top1_num_eq_scrambled_rate"] = 0.0
        out["any_beam_exact_token_rate"] = 0.0
        out["any_beam_exact_string_rate"] = 0.0
        out["any_beam_num_eq_simple_rate"] = 0.0
        out["any_beam_num_eq_scrambled_rate"] = 0.0
    return out


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    t0 = time.perf_counter()
    sync_base_config()
    base.ensure_output_dirs()

    device = base.resolve_device()
    print(f"Using device: {device}")
    print("Generating one shared dataset for all model comparisons...")
    base.generate_test_data()

    raw_rows = base.load_raw_rows(base.RAW_CSV_PATH)
    token_rows = base.load_token_rows(base.TOK_CSV_PATH)
    if len(raw_rows) != len(token_rows):
        raise ValueError("Raw and tokenised row counts do not match")

    tokenizer = base.ScatteringAmplitudeTokenizer(max_particles=TOKENIZER_MAX_PARTICLES)
    dataset = base.TransformerDataset(str(base.TOK_CSV_PATH), max_length=MAX_SEQ_LENGTH_OVERRIDE)
    cached_kinematics = base.precompute_kinematics()

    comparison_rows: list[dict[str, object]] = []

    for model_run in MODEL_RUNS:
        if not model_run.enabled:
            continue

        print(f"\n{'=' * 72}")
        print(f"Evaluating model: {model_run.name}")
        print(f"Checkpoint      : {model_run.model_path}")
        base.MODEL_PATH = model_run.model_path
        model = base.load_model(device)

        for decode_cfg in DECODE_RUNS:
            if not decode_cfg.enabled:
                continue

            detail_rows, summary = base.evaluate_mode(
                model,
                tokenizer,
                dataset,
                raw_rows,
                token_rows,
                cached_kinematics,
                decode_cfg,
            )

            detail_path = (
                DATA_TESTING_DIR
                / OUTPUT_SUBDIR
                / f"{COMPARE_NAME}_{model_run.name}_{decode_cfg.name}_results.csv"
            )
            base.write_detail_csv(detail_path, detail_rows)
            base.print_summary(summary)
            base.print_examples(detail_rows, PRINT_EXAMPLES_PER_MODE)
            print(f"  wrote detail results to {detail_path}")

            comparison_rows.append(
                augment_summary(
                    summary,
                    model_name=model_run.name,
                    model_path=model_run.model_path,
                )
            )

    write_comparison_csv(COMPARE_SUMMARY_CSV_PATH, comparison_rows)
    print(f"\nWrote collated comparison summary to {COMPARE_SUMMARY_CSV_PATH}")
    print(f"Total wall time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
