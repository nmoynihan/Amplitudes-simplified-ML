#!/usr/bin/env python3
"""Search for a numerical match by repeatedly sampling the ORIGINAL amplitude.

One attempt is one ``decode_with_model(..., decoding_method="nucleus")`` call.
Check top-1 first, then all returned hypotheses in their existing list order
(the decoder does not return scores). Stop at the first numerical match, even
if it is longer than the input. No greedy pre-check or input replacement occurs.

CSV/CSV.GZ layouts: raw simple/scrambled pairs, JSON token simple/scrambled
pairs, a tokens column, an expression/amplitude column, or headerless Feynman
id,expression records. Token inputs contain content IDs only, without special
tokens. Pair searches always use scrambled as input AND numerical reference.
Rows, including duplicates and invalid records, retain their source order.

Example (from the repository root; substitute an available checkpoint):
    python data_testing/evaluate_nucleus_search.py \\
        --checkpoint models/best_model.pt --test-data data/test.csv.gz \\
        --input-format raw --max-rows 10 --max-attempts 20 --beam-size 4 \\
        --p-nucleus 0.95 --temperature-nucleus 1.0 --max-length 512 \\
        --sampling-seed 42 --numeric-seed 151 --device cpu

Each run creates a unique directory with results.csv (flushed after each row),
config.json, summary.json, and summary.txt (the final console report). Token
lengths count content only: valid original inputs and the first matching
prediction, excluding BOS/EOS/PAD. candidates_checked counts distinct normalized
token candidates, including malformed candidates; candidates_encountered also
counts repeats, and cache_hits reports skipped repeated checks. No repeat or
decoding failure refunds an attempt. Sampling is seeded ONCE after setup;
reproduction requires the same dataset order, config, device and software.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_testing import evaluate_model as evaluator
from data_testing import evaluate_single_amplitude as single


# ============================================================================
# Configuration: edit these defaults or override with the documented CLI flags.
# Relative paths resolve from ROOT. None max_rows means all records, no dedupe.
# max_attempts is independent of beam_size (hypotheses retained per decode call).
# max_length includes BOS/EOS and must fit the checkpoint's target capacity.
# Numeric tolerances None select evaluate_model's SQED/YM/gravity defaults.
# SQED uses mass; YM uses energy_scale and defaults to both polarization modes.
# Gravity requires five particles and process labels (column, metadata or flag).
# ============================================================================
@dataclass(frozen=True)
class RunConfig:
    checkpoint: Path = ROOT / "models" / "best_model.pt"
    test_data: Path = ROOT / "data" / "sqed" / "sqed_4ptseed_oneshot.csv.gz"
    input_format: str = "auto"
    max_rows: int | None = None
    tokens_column: str = "tokens"
    expression_column: str | None = None  # Header name or zero-based column index.
    id_column: str | None = None  # Default: id, row_id, index, then record ordinal.
    max_attempts: int = 100
    beam_size: int = 4
    p_nucleus: float = 0.95
    temperature_nucleus: float = 1.0
    sampling_seed: int = 42
    numeric_seed: int = 151
    device: str = "auto"
    n_particles: int = 4
    tokenizer_max_particles: int = 8
    numeric_backend: str = "sqed"
    mass: float = 2.0
    energy_scale: float = 2.0
    numeric_samples: int = 3
    numeric_pol_modes: tuple[str, ...] | None = None
    gravity_metadata_csv: Path | None = None
    gravity_process: str | None = None
    gravity_reference_modes: tuple[str, ...] = ("first", "last", "random")
    gravity_gauge_shift: bool = True
    tol_abs: float | None = None
    tol_rel: float | None = None
    max_length: int = 512
    output_dir: Path = ROOT / "data_testing" / "outputs" / "nucleus_search"


DEFAULT_CONFIG = RunConfig()


@dataclass
class InputRow:
    row_index: int  # One-based source data-record ordinal, before any row cap.
    row_id: str
    input_expression: str
    input_tokens: list[int] | None = None
    simple_expression: str | None = None
    process: str | None = None
    error: str | None = None
    source_line: int | None = None
    input_value: str = ""  # Original cell, including malformed JSON when applicable.
    id_column: str | None = None  # Actual header supplying row_id, if present.


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "--model-path", type=Path)
    parser.add_argument("--test-data", "--input-csv", type=Path)
    parser.add_argument("--input-format", choices=["auto", "raw", "token-pair", "tokens", "expression", "feyn"])
    for name in ("max-rows", "max-attempts", "beam-size", "sampling-seed", "numeric-seed",
                 "n-particles", "tokenizer-max-particles", "numeric-samples", "max-length"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("p-nucleus", "temperature-nucleus", "mass", "energy-scale", "tol-abs", "tol-rel"):
        parser.add_argument(f"--{name}", type=float)
    for name in ("tokens-column", "expression-column", "id-column"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--numeric-backend", choices=["sqed", "ym", "yang-mills", "gravity"])
    parser.add_argument("--numeric-pol-modes", nargs="+", choices=["coulomb", "covariant"])
    parser.add_argument("--gravity-reference-modes", nargs="+", choices=["first", "last", "random", "cyclic"])
    parser.add_argument("--gravity-process", choices=sorted(evaluator.GRAVITY_PROCESS_SPECS))
    parser.add_argument("--gravity-metadata-csv", type=Path)
    parser.add_argument("--gravity-gauge-shift", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=Path)
    overrides = {key: value for key, value in vars(parser.parse_args(argv)).items() if value is not None}
    for key in ("numeric_pol_modes", "gravity_reference_modes"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    return replace(DEFAULT_CONFIG, **overrides)


@contextmanager
def evaluator_settings(config: RunConfig) -> Iterator[None]:
    """Reuse legacy helpers without leaving changes to evaluator module globals.

    This standalone runner is serial; the legacy configuration is not thread-safe.
    """
    settings = dict(
        MODEL_PATH=config.checkpoint, DEVICE=config.device, DATA_SOURCE="csv",
        RAW_CSV_PATH=config.test_data, TOK_CSV_PATH=config.test_data,
        N_PARTICLES=config.n_particles, TOKENIZER_MAX_PARTICLES=config.tokenizer_max_particles,
        NUMERIC_BACKEND="ym" if config.numeric_backend == "yang-mills" else config.numeric_backend,
        NUMERIC_EQUIV_SAMPLES=config.numeric_samples, NUMERIC_EQUIV_SEED=config.numeric_seed,
        NUMERIC_EQUIV_MASS=config.mass, NUMERIC_EQUIV_ENERGY_SCALE=config.energy_scale,
        NUMERIC_EQUIV_POL_MODES=config.numeric_pol_modes,
        NUMERIC_TOL_ABS=config.tol_abs, NUMERIC_TOL_REL=config.tol_rel,
        GRAVITY_PROCESS=config.gravity_process, GRAVITY_METADATA_CSV_PATH=config.gravity_metadata_csv,
        GRAVITY_REFERENCE_MODES=config.gravity_reference_modes, GRAVITY_GAUGE_SHIFT=config.gravity_gauge_shift,
        INPUT_TOKEN_LIMIT=None, MAX_SEQ_LENGTH_OVERRIDE=config.max_length,
        SINGLE_AMPLITUDE_INPUT_FORMAT="auto", SINGLE_AMPLITUDE_EXPRESSION_COLUMN=1,
    )
    previous = {name: getattr(evaluator, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(evaluator, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(evaluator, name, value)


def validate_config(config: RunConfig) -> None:
    for name in ("max_attempts", "beam_size", "numeric_samples", "n_particles", "tokenizer_max_particles"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.max_rows is not None and (type(config.max_rows) is not int or config.max_rows < 1):
        raise ValueError("max_rows must be a positive integer or None")
    if type(config.max_length) is not int or config.max_length < 2:
        raise ValueError("max_length must be an integer >= 2 (including BOS/EOS)")
    for name in ("sampling_seed", "numeric_seed"):
        value = getattr(config, name)
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2**32)")
    if not math.isfinite(config.p_nucleus) or not 0 < config.p_nucleus <= 1:
        raise ValueError("p_nucleus must be finite and in (0, 1]")
    if not math.isfinite(config.temperature_nucleus) or config.temperature_nucleus <= 0:
        raise ValueError("temperature_nucleus must be finite and positive")
    if config.n_particles > config.tokenizer_max_particles:
        raise ValueError("n_particles exceeds tokenizer_max_particles")
    if config.input_format not in {"auto", "raw", "token-pair", "tokens", "expression", "feyn"}:
        raise ValueError(f"Unsupported input_format: {config.input_format}")
    if config.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {config.device}")
    with evaluator_settings(config):
        evaluator.validate_runtime_config()


def decode_content(tokenizer: Any, tokens: Any) -> str:
    """Apply the existing input-token constraints without silently dropping IDs."""
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise ValueError("Input must be a nonempty JSON list of integer token IDs (not booleans)")
    invalid = set(tokens) - set(tokenizer.id_to_token)
    if invalid:
        raise ValueError(f"Token IDs outside configured vocabulary: {sorted(invalid)}")
    if set(tokens) & {0, 1, 2, 3}:
        raise ValueError("Content tokens must exclude PAD/UNK/BOS/EOS")
    evaluator.validate_input_token_rows([{"scrambled": tokens}])
    ok, expression, error = evaluator.safe_decode_infix(tokenizer, tokens)
    if not ok:
        raise ValueError(f"Malformed prefix expression: {error}")
    return expression


def load_inputs(config: RunConfig, tokenizer: Any) -> tuple[list[InputRow], int, str]:
    """Read original records using the evaluators' gzip and layout helpers.

    Bulk preparation helpers discard IDs and fail on the first invalid row, so
    use their CSV reader/format detection and tokenizer on each retained record.
    Empty physical lines are skipped; CSV records with empty cells are retained.
    Auto pair format recognizes JSON lists in the scrambled cell per row.
    """
    path = config.test_data
    if not path.name.endswith((".csv", ".csv.gz")):
        raise ValueError("test_data must be a .csv or .csv.gz file")
    rows: list[InputRow] = []
    total = 0
    with evaluator.open_csv_text(path, "r") as handle:
        reader = csv.reader(handle)
        first = next((record for record in reader if record), None)
        if first is None:
            raise ValueError(f"Empty dataset: {path}")
        detected = single.detect_input_format(first, requested_format=config.input_format,
                                             tokens_column=config.tokens_column,
                                             expression_column=config.expression_column)
        header = [] if detected == "feyn" else single._normalised_header(first)
        if detected in {"raw", "token-pair"}:
            expression_index = single._column_index(header, "scrambled")
            simple_index = single._column_index(header, "simple")
        elif detected == "tokens":
            expression_index = single._column_index(header, config.tokens_column)
            simple_index = None
        elif detected == "expression":
            expression_index = single._resolve_header_expression_index(header, config.expression_column)
            simple_index = None
        else:
            expression_index = single._resolve_feyn_expression_index(first, config.expression_column)
            simple_index = None
        id_index = None
        if config.id_column is not None:
            id_index = single._column_index(header, config.id_column) if header else int(config.id_column)
            if id_index < 0:
                raise ValueError("id_column index must be non-negative")
        elif header:
            id_index = next((header.index(name) for name in ("id", "row_id", "index") if name in header), None)
        elif expression_index != 0:
            id_index = 0

        def records():
            if detected == "feyn":
                yield reader.line_num, first
            for record in reader:
                if record:
                    yield reader.line_num, record

        for line_number, record in records():
            total += 1
            if config.max_rows is not None and total > config.max_rows:
                continue
            value = record[expression_index] if expression_index < len(record) else ""
            row_id = record[id_index] if id_index is not None and id_index < len(record) else str(total)
            simple_value = record[simple_index] if simple_index is not None and simple_index < len(record) else None
            row = InputRow(total, row_id, value, simple_expression=simple_value,
                           process=single._optional_process(header, record),
                           source_line=line_number, input_value=value,
                           id_column=header[id_index] if header and id_index is not None else None)
            is_tokens = detected in {"tokens", "token-pair"} or (
                config.input_format == "auto" and detected == "raw" and value.lstrip().startswith("["))
            try:
                if is_tokens:
                    row.input_expression = ""
                    row.input_tokens = json.loads(value)
                    row.input_expression = decode_content(tokenizer, row.input_tokens)
                    # The simple target is diagnostic only; do not reject a valid
                    # numerical reference merely because the target is malformed.
                    if simple_value:
                        try:
                            row.simple_expression = decode_content(tokenizer, json.loads(simple_value))
                        except (ValueError, TypeError):
                            pass
                elif not value.strip():
                    raise ValueError("Empty input expression")
            except (ValueError, TypeError) as exc:
                row.error = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    if not total:
        raise ValueError(f"Dataset contains no data rows: {path}")
    if config.numeric_backend == "gravity":
        attach_gravity_processes(config, rows, total)
    return rows, total, detected


def attach_gravity_processes(config: RunConfig, rows: list[InputRow], total: int) -> None:
    """Align metadata by original record ordinal, never by expression/dedupe."""
    if config.gravity_process is not None:
        for row in rows:
            if row.process and row.process != config.gravity_process:
                row.error = f"Inline gravity process {row.process!r} conflicts with --gravity-process {config.gravity_process!r}"
            row.process = config.gravity_process
        return
    metadata_path = config.gravity_metadata_csv
    if metadata_path is None and any(row.process is None for row in rows):
        metadata_path = evaluator.infer_gravity_metadata_path(config.test_data)
        if metadata_path == config.test_data:
            metadata_path = None  # Use the inline process, including missing values.
    if metadata_path is not None:
        metadata = evaluator.load_raw_rows(metadata_path)
        if len(metadata) != total:
            raise ValueError(f"Gravity metadata has {len(metadata)} rows; dataset has {total}")
        for row in rows:
            entry = metadata[row.row_index - 1]
            process = (entry.get("process") or "").strip()
            if row.id_column in entry and entry[row.id_column] != row.row_id:
                row.error = f"Gravity metadata ID mismatch at source row {row.row_index}"
            if entry.get("scrambled"):
                metadata_expression = entry["scrambled"].strip()
                aligned = metadata_expression == row.input_value.strip()
                if not aligned and row.input_tokens is not None:
                    # Standard gravity metadata contains raw expressions even
                    # when the chosen test set contains their tokenized form.
                    tokenizer = evaluator.ScatteringAmplitudeTokenizer(
                        max_particles=config.tokenizer_max_particles, max_sequence_length=None)
                    try:
                        aligned = tokenizer.encode_infix(metadata_expression.replace("**", "^")) == row.input_tokens
                    except (ValueError, TypeError):
                        pass
                if not aligned:
                    row.error = f"Gravity metadata scrambled expression mismatch at source row {row.row_index}"
            if row.process and row.process != process:
                row.error = f"Conflicting inline and metadata gravity processes at row {row.row_index}"
            row.process = process
    for row in rows:
        if row.process not in evaluator.GRAVITY_PROCESS_SPECS:
            row.error = (f"Missing or unsupported gravity process {row.process!r}; supply a process column, "
                         "--gravity-metadata-csv or --gravity-process")


def validate_reference(row: InputRow, tokenizer: Any, cached_kinematics: Any) -> None:
    if row.error:
        raise ValueError(row.error)
    expression = row.input_expression
    if not expression.strip():
        raise ValueError("Empty numerical reference")
    # Reuse strict syntax, backend compatibility, scalar-endpoint and cache checks.
    evaluator.validate_numeric_reference_rows(
        [{"simple": expression, "scrambled": expression}], cached_kinematics, [row.process])
    if row.input_tokens is None:
        row.input_tokens = tokenizer.encode_infix(expression.replace("**", "^"))
    model_expression = decode_content(tokenizer, row.input_tokens)
    tol_abs, tol_rel = evaluator.resolve_numeric_tolerances()
    if evaluator.NUMERIC_BACKEND == "gravity":
        points = [(point, None) for point in cached_kinematics[row.process]]
    else:
        points = cached_kinematics
    # The existing input validator checks only the first point. Reject poles or
    # evaluation failures on ANY configured sample/reference/polarization mode.
    for index, (momenta, pols) in enumerate(points, start=1):
        try:
            value = evaluator.eval_numeric_expr(expression, momenta, pols, gravity_process=row.process)
            if not math.isfinite(abs(value)):
                raise ValueError("non-finite reference value")
            model_value = evaluator.eval_numeric_expr(model_expression, momenta, pols, gravity_process=row.process)
            if not math.isfinite(abs(model_value)) or not evaluator.numeric_values_close(
                value, model_value, tol_abs=tol_abs, tol_rel=tol_rel
            ):
                raise ValueError("Tokenized input does not numerically reproduce the original reference")
        except Exception as exc:
            raise ValueError(f"Numerical reference failed at kinematics point {index}: {exc}") from exc


def normalize_candidate(sequence: Any) -> list[int]:
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
    if not isinstance(sequence, list) or any(type(token) is not int for token in sequence):
        raise ValueError("Generated candidate must be an integer token list")
    content = list(sequence)
    if content and content[0] == 2:
        content.pop(0)
    if 3 in content:
        end = content.index(3)
        if any(token != 0 for token in content[end + 1:]):
            raise ValueError("Non-padding tokens after generated EOS")
        content = content[:end]
    else:
        while content and content[-1] == 0:
            content.pop()
    if set(content) & {0, 1, 2, 3}:
        raise ValueError("Embedded PAD/UNK/BOS/EOS in generated content")
    return evaluator.strip_special_tokens(content)


@torch.inference_mode()
def search_amplitude(model: Any, tokenizer: Any, row: InputRow, config: RunConfig,
                     cached_kinematics: Any, *, decode_fn: Any = None) -> dict[str, Any]:
    """Search one row; caller scopes evaluator_settings and seeds once per run."""
    started = time.perf_counter()
    result = dict(row_index=row.row_index, row_id=row.row_id, source_line=row.source_line,
                  input_expression=row.input_expression, input_value=row.input_value,
                  simple_expression=row.simple_expression, gravity_process=row.process,
                  status="exhausted", attempts_used=0, first_successful_attempt=None,
                  candidates_checked=0, candidates_encountered=0, cache_hits=0,
                  input_token_length=None, matching_token_length=None,
                  matching_expression=None, matching_token_ids=None,
                  elapsed_seconds=0.0, error_diagnostics={})
    errors: Counter[str] = Counter()

    def diagnostic(message: str) -> None:
        # Bound log memory on very long searches while keeping counts useful.
        key = message[:500]
        errors[key if key in errors or len(errors) < 8 else "additional errors"] += 1

    try:
        validate_reference(row, tokenizer, cached_kinematics)
    except Exception as exc:
        result["status"] = "invalid_input"
        diagnostic(f"Invalid reference/input: {type(exc).__name__}: {exc}")
    else:
        # Validation guarantees these are the original content IDs, without
        # BOS/EOS/PAD; paired datasets supply scrambled, not the simple target.
        result["input_token_length"] = len(row.input_tokens)
        src = torch.tensor([[2, *row.input_tokens, 3]], dtype=torch.long,
                           device=getattr(model, "device", config.device))
        decoder = decode_fn if decode_fn is not None else evaluator.decode_with_model
        checked: set[Any] = set()
        for attempt in range(1, config.max_attempts + 1):
            result["attempts_used"] = attempt
            try:
                decoded, all_beams = decoder(
                    model, src, max_length=config.max_length, decoding_method="nucleus",
                    beam_size=config.beam_size, p_nucleus=config.p_nucleus,
                    temperature_nucleus=config.temperature_nucleus,
                    bos_token=2, eos_token=3, pad_token=0)
                candidates = [decoded[0]]
                if all_beams is not None:
                    candidates.extend(all_beams[0])
            except Exception as exc:
                diagnostic(f"Decode failure: {type(exc).__name__}: {exc}")
                continue
            for candidate in candidates:
                result["candidates_encountered"] += 1
                try:
                    tokens = normalize_candidate(candidate)
                    key = tuple(tokens)
                    normalization_error = None
                except Exception as exc:
                    key = ("malformed", repr(candidate))
                    normalization_error = str(exc)
                if key in checked:
                    result["cache_hits"] += 1
                    continue
                checked.add(key)
                result["candidates_checked"] += 1
                try:
                    if normalization_error is not None:
                        raise ValueError(normalization_error)
                    expression = decode_content(tokenizer, tokens)
                    # Equality and length never bypass strict numerical checking.
                    matched = evaluator.numerically_equivalent_exprs(
                        expression, row.input_expression, cached_kinematics, gravity_process=row.process)
                    if not matched:
                        diagnostic("Numerical mismatch or invalid/non-finite candidate evaluation")
                        continue
                except Exception as exc:
                    diagnostic(f"Malformed candidate: {type(exc).__name__}: {exc}")
                    continue
                result.update(status="matched", first_successful_attempt=attempt,
                              matching_expression=expression, matching_token_ids=tokens,
                              matching_token_length=len(tokens))
                break
            if result["status"] == "matched":
                break
    result["elapsed_seconds"] = time.perf_counter() - started
    result["error_diagnostics"] = dict(errors)
    return result


def seed_sampling(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Includes CUDA/MPS seeds in supported PyTorch releases.


def summarize(results: list[dict[str, Any]], dataset_rows: int, runtime: float) -> dict[str, Any]:
    counts = Counter(row["status"] for row in results)
    matched_rows = [row for row in results if row["status"] == "matched"]
    valid_rows = [row for row in results if row["status"] in {"matched", "exhausted"}]
    successful_attempts = [row["first_successful_attempt"] for row in matched_rows]
    valid = len(valid_rows)
    token_lengths = {
        name: dict(count=len(population),
                   mean=statistics.mean(row[field] for row in population) if population else None)
        for name, population, field in (
            ("valid_searched_inputs", valid_rows, "input_token_length"),
            ("matched_inputs", matched_rows, "input_token_length"),
            ("matched_predictions", matched_rows, "matching_token_length"),
        )
    }
    return dict(dataset_rows=dataset_rows, selected_rows=len(results),
                rows_excluded_by_limit=dataset_rows - len(results), matched=counts["matched"],
                exhausted=counts["exhausted"], invalid_input=counts["invalid_input"],
                success_rate=counts["matched"] / valid if valid else None,
                success_rate_denominator=valid, success_rate_denominator_description="valid searched inputs (matched + exhausted)",
                attempts_used_total=sum(row["attempts_used"] for row in results),
                attempts_to_success=dict(count=len(successful_attempts),
                    mean=statistics.mean(successful_attempts) if successful_attempts else None,
                    median=statistics.median(successful_attempts) if successful_attempts else None,
                    min=min(successful_attempts, default=None), max=max(successful_attempts, default=None)),
                token_lengths=token_lengths, total_runtime_seconds=runtime)


def format_summary(summary: dict[str, Any], run_dir: Path) -> str:
    """Build the complete final report shared by stdout and summary.txt."""
    rate = summary["success_rate"]
    rate_text = "N/A (no valid inputs)" if rate is None else f"{rate:.2%}"
    attempts_text = ", ".join(
        f"{key}={value if value is not None else 'N/A'}"
        for key, value in summary["attempts_to_success"].items()
    )
    lines = [
        f"Dataset: {summary['dataset_rows']} rows; selected: {summary['selected_rows']}; "
        f"excluded by row limit: {summary['rows_excluded_by_limit']}.",
        f"Matched {summary['matched']}; exhausted {summary['exhausted']}; invalid {summary['invalid_input']}. "
        f"Success rate {rate_text} ({summary['matched']}/{summary['success_rate_denominator']} valid searched inputs).",
        f"Attempts to success: {attempts_text}",
    ]
    for key, label in (
        ("valid_searched_inputs", "Mean original input length (valid searched rows: matched + exhausted"),
        ("matched_inputs", "Mean original input length (matched rows only"),
        ("matched_predictions", "Mean nucleus prediction length (same matched rows, first numerical match"),
    ):
        metric = summary["token_lengths"][key]
        mean_text = "N/A" if metric["mean"] is None else f"{metric['mean']:.2f}"
        lines.append(f"{label}, n={metric['count']}): {mean_text} content tokens")
    lines.extend([
        f"Total runtime: {summary['total_runtime_seconds']:.2f}s.",
        f"Results: {run_dir / 'results.csv'}",
        f"Configuration: {run_dir / 'config.json'}",
        f"Summary JSON: {run_dir / 'summary.json'}",
        f"Summary text: {run_dir / 'summary.txt'}",
    ])
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str, allow_nan=False)
        handle.write("\n")


def run(config: RunConfig) -> Path:
    started = time.perf_counter()
    config = replace(config, **{name: evaluator.resolve_input_path(getattr(config, name)).resolve()
                               for name in ("checkpoint", "test_data", "output_dir", "gravity_metadata_csv")
                               if getattr(config, name) is not None})
    validate_config(config)
    for path in (config.checkpoint, config.test_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    with evaluator_settings(config):
        tokenizer = evaluator.ScatteringAmplitudeTokenizer(max_particles=config.tokenizer_max_particles,
                                                          max_sequence_length=None)
        rows, total, detected = load_inputs(config, tokenizer)
        device = evaluator.resolve_device()
        model = evaluator.load_model(device)  # Exactly one checkpoint load; helper calls eval().
        model.device = device
        source_capacity = evaluator.positional_encoding_capacity(model, "src_pos_encoding")
        target_capacity = evaluator.positional_encoding_capacity(model, "tgt_pos_encoding")
        if config.max_length > target_capacity:
            raise ValueError(f"max_length={config.max_length} exceeds checkpoint target capacity {target_capacity}")
        if model.vocab_size < tokenizer.vocab_size:
            raise ValueError(f"Checkpoint vocabulary {model.vocab_size} is smaller than tokenizer vocabulary {tokenizer.vocab_size}; "
                             "check --tokenizer-max-particles")
        cached_kinematics = evaluator.precompute_kinematics()
        # Validate all selected inputs and source lengths BEFORE any inference.
        for row in rows:
            try:
                validate_reference(row, tokenizer, cached_kinematics)
                if len(row.input_tokens) + 2 > source_capacity:
                    raise ValueError(f"Source has {len(row.input_tokens) + 2} tokens including BOS/EOS; "
                                     f"checkpoint source capacity is {source_capacity}")
            except Exception as exc:
                row.error = f"{type(exc).__name__}: {exc}"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"),
                                        dir=config.output_dir))
        resolved = asdict(config)
        if (evaluator.NUMERIC_BACKEND == "gravity" and config.gravity_process is None
                and config.gravity_metadata_csv is None):
            resolved["gravity_metadata_csv"] = evaluator.infer_gravity_metadata_path(config.test_data)
        resolved.update(device=device, numeric_backend=evaluator.NUMERIC_BACKEND,
                        tol_abs=evaluator.resolve_numeric_tolerances()[0],
                        tol_rel=evaluator.resolve_numeric_tolerances()[1],
                        numeric_pol_modes=evaluator.resolve_numeric_pol_modes(),
                        gravity_reference_modes=(evaluator.resolve_gravity_reference_modes()
                                                 if evaluator.NUMERIC_BACKEND == "gravity" else config.gravity_reference_modes),
                        detected_input_format=detected, dataset_rows=total, selected_rows=len(rows),
                        source_capacity=source_capacity, target_capacity=target_capacity,
                        checkpoint_vocab_size=model.vocab_size, tokenizer_vocab_size=tokenizer.vocab_size,
                        candidate_order="top-1, then all_beams[0] in returned list order; cached repeats skipped",
                        torch_version=torch.__version__, numpy_version=np.__version__,
                        kinematics_points={key: len(points) for key, points in cached_kinematics.items()}
                            if isinstance(cached_kinematics, dict) else len(cached_kinematics))
        write_json(run_dir / "config.json", resolved)
        print(f"Dataset: {total} rows; selected: {len(rows)}; output: {run_dir}", flush=True)
        print(f"Nucleus: {config.max_attempts} calls/row, beam_size={config.beam_size}, "
              f"p={config.p_nucleus}, temperature={config.temperature_nucleus}; device={device}", flush=True)
        seed_sampling(config.sampling_seed)  # After checkpoint init and all numerical setup.
        results = []
        with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = None
            for index, row in enumerate(rows, start=1):
                result = search_amplitude(model, tokenizer, row, config, cached_kinematics)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(result))
                    writer.writeheader()
                serializable = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                                for key, value in result.items()}
                writer.writerow(serializable)
                handle.flush()
                results.append(result)
                print(f"[{index}/{len(rows)}] id={row.row_id!r} {result['status']} "
                      f"attempts={result['attempts_used']} checked={result['candidates_checked']} "
                      f"time={result['elapsed_seconds']:.2f}s", flush=True)
                if result["status"] == "invalid_input":
                    print(f"  {result['error_diagnostics']}", flush=True)
        summary = summarize(results, total, time.perf_counter() - started)
        write_json(run_dir / "summary.json", summary)
        report = format_summary(summary, run_dir)
        (run_dir / "summary.txt").write_text(report, encoding="utf-8")
        print(report, end="", flush=True)
        return run_dir


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
