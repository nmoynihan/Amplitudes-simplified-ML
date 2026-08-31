#!/usr/bin/env python3
"""Evaluate the first amplitude in a CSV with one trained model.

This is a focused front end for :mod:`data_testing.evaluate_model`.  It counts
the amplitude entries in the supplied CSV, keeps the first non-empty entry,
and delegates model inference and strict numerical checking to the main
evaluator.  The supported numerical backends are scalar QED, Yang--Mills, and
five-point gravity.

Supported CSV layouts are:

* ``simple,scrambled`` raw or tokenised evaluator data;
* a header containing a JSON integer-list ``tokens`` column;
* a header containing an ``expression`` or ``amplitude`` column; and
* headerless ``id,expression`` Feynman data.

Both plain ``.csv`` and gzip-compressed ``.csv.gz`` inputs are accepted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parent.parent
GRAVITY_PROCESSES = {"3s2h", "4s1h"}
EXPRESSION_HEADER_NAMES = ("expression", "amplitude", "feyn")


@dataclass(frozen=True)
class SelectedAmplitude:
    """The first amplitude and the information needed by ``evaluate_model``."""

    source_format: str
    evaluator_format: str
    total_entries: int
    simple: str | None = None
    scrambled: str | None = None
    simple_tokens: str | None = None
    tokens: str | None = None
    expression: str | None = None
    process: str | None = None


def open_csv_text(path: Path, mode: str) -> TextIO:
    """Open a plain or gzip-compressed CSV as UTF-8 text."""

    text_mode = mode if "t" in mode else f"{mode}t"
    if path.suffix == ".gz":
        return gzip.open(path, text_mode, newline="", encoding="utf-8")
    return path.open(text_mode, newline="", encoding="utf-8")


def resolve_repo_path(path: Path | str) -> Path:
    """Resolve relative inputs from the repository root, like evaluate_model."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def _nonempty_rows(reader: Iterator[list[str]]) -> Iterator[list[str]]:
    for row in reader:
        if any(cell.strip() for cell in row):
            yield row


def _normalised_header(row: list[str]) -> list[str]:
    return [cell.strip().lower() for cell in row]


def _column_index(header: list[str], name: str) -> int:
    normalised = _normalised_header(header)
    try:
        return normalised.index(name.strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"CSV header does not contain a {name!r} column; got {header}"
        ) from exc


def _cell(row: list[str], index: int, *, label: str) -> str:
    if index >= len(row):
        raise ValueError(f"First amplitude row has no {label} column at index {index}")
    value = row[index].strip()
    if not value:
        raise ValueError(f"First amplitude row has an empty {label} value")
    return value


def _maybe_json_token_list(value: str, *, label: str) -> list[int] | None:
    """Return a validated token list, or ``None`` when ``value`` is not one."""

    try:
        tokens = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(tokens, list):
        return None
    if not tokens:
        raise ValueError(f"First amplitude's {label!r} token list is empty")
    if any(type(token) is not int for token in tokens):
        raise ValueError(
            f"First amplitude's {label!r} token list must contain integers"
        )
    return tokens


def detect_input_format(
    first_row: list[str],
    *,
    requested_format: str,
    tokens_column: str,
    expression_column: str | None = None,
) -> str:
    """Detect whether the first non-empty CSV record is a header or data."""

    if requested_format != "auto":
        return requested_format

    header = set(_normalised_header(first_row))
    if {"simple", "scrambled"}.issubset(header):
        return "raw"
    if tokens_column.strip().lower() in header:
        return "tokens"
    if header.intersection(EXPRESSION_HEADER_NAMES):
        return "expression"
    if expression_column is not None:
        try:
            int(expression_column)
        except ValueError:
            if expression_column.strip().lower() in header:
                return "expression"
    return "feyn"


def _resolve_header_expression_index(
    header: list[str],
    expression_column: str | None,
) -> int:
    if expression_column is not None:
        try:
            index = int(expression_column)
        except ValueError:
            return _column_index(header, expression_column)
        if index < 0:
            raise ValueError("--expression-column must be non-negative")
        return index

    normalised = _normalised_header(header)
    for name in EXPRESSION_HEADER_NAMES:
        if name in normalised:
            return normalised.index(name)
    raise ValueError(
        "Expression CSV header must contain an 'expression', 'amplitude', or "
        "'feyn' column, or use --expression-column"
    )


def _resolve_feyn_expression_index(
    first_data_row: list[str],
    expression_column: str | None,
) -> int:
    if expression_column is None:
        return 1 if len(first_data_row) > 1 else 0
    try:
        index = int(expression_column)
    except ValueError as exc:
        raise ValueError(
            "A headerless Feynman CSV needs a zero-based integer "
            "--expression-column"
        ) from exc
    if index < 0:
        raise ValueError("--expression-column must be non-negative")
    return index


def _optional_process(header: list[str], row: list[str]) -> str | None:
    normalised = _normalised_header(header)
    if "process" not in normalised:
        return None
    index = normalised.index("process")
    if index >= len(row):
        return None
    return row[index].strip() or None


def select_first_amplitude(
    source_path: Path,
    *,
    input_format: str = "auto",
    tokens_column: str = "tokens",
    expression_column: str | None = None,
) -> SelectedAmplitude:
    """Count amplitudes in ``source_path`` and return only the first entry."""

    if not source_path.is_file():
        raise FileNotFoundError(f"Amplitude CSV does not exist: {source_path}")
    if not (source_path.name.endswith(".csv") or source_path.name.endswith(".csv.gz")):
        raise ValueError(f"Amplitude input must be a .csv or .csv.gz file: {source_path}")

    with open_csv_text(source_path, "r") as handle:
        rows = _nonempty_rows(csv.reader(handle))
        first_row = next(rows, None)
        if first_row is None:
            raise ValueError(f"Amplitude CSV is empty: {source_path}")

        detected = detect_input_format(
            first_row,
            requested_format=input_format,
            tokens_column=tokens_column,
            expression_column=expression_column,
        )

        if detected == "feyn":
            first_data_row = first_row
            total_entries = 1 + sum(1 for _ in rows)
            expression_index = _resolve_feyn_expression_index(
                first_data_row,
                expression_column,
            )
            expression = _cell(
                first_data_row,
                expression_index,
                label="expression",
            )
            return SelectedAmplitude(
                source_format="feyn",
                evaluator_format="feyn",
                total_entries=total_entries,
                expression=expression,
            )

        header = first_row
        first_data_row = next(rows, None)
        if first_data_row is None:
            raise ValueError(f"Amplitude CSV has a header but no data rows: {source_path}")
        total_entries = 1 + sum(1 for _ in rows)
        process = _optional_process(header, first_data_row)

        if detected in {"raw", "token-pair"}:
            simple = _cell(
                first_data_row,
                _column_index(header, "simple"),
                label="simple",
            )
            scrambled = _cell(
                first_data_row,
                _column_index(header, "scrambled"),
                label="scrambled",
            )
            simple_tokens = _maybe_json_token_list(simple, label="simple")
            scrambled_tokens = _maybe_json_token_list(scrambled, label="scrambled")
            if simple_tokens is not None or scrambled_tokens is not None:
                if simple_tokens is None or scrambled_tokens is None:
                    raise ValueError(
                        "A tokenised simple,scrambled row must contain JSON token "
                        "lists in both columns"
                    )
                return SelectedAmplitude(
                    source_format="token-pair",
                    evaluator_format="token-pair",
                    total_entries=total_entries,
                    simple_tokens=json.dumps(simple_tokens),
                    tokens=json.dumps(scrambled_tokens),
                    process=process,
                )
            if detected == "token-pair":
                raise ValueError(
                    "A token-pair CSV must contain JSON token lists in its "
                    "simple and scrambled columns"
                )
            return SelectedAmplitude(
                source_format="raw",
                evaluator_format="raw",
                total_entries=total_entries,
                simple=simple,
                scrambled=scrambled,
                process=process,
            )

        if detected == "tokens":
            token_text = _cell(
                first_data_row,
                _column_index(header, tokens_column),
                label=tokens_column,
            )
            tokens = _maybe_json_token_list(token_text, label=tokens_column)
            if tokens is None:
                raise ValueError(
                    f"First amplitude's {tokens_column!r} value must be a JSON token list"
                )
            return SelectedAmplitude(
                source_format="tokens",
                evaluator_format="tokens",
                total_entries=total_entries,
                tokens=json.dumps(tokens),
                process=process,
            )

        if detected == "expression":
            expression_index = _resolve_header_expression_index(
                header,
                expression_column,
            )
            expression = _cell(
                first_data_row,
                expression_index,
                label="expression",
            )
            return SelectedAmplitude(
                source_format="expression",
                evaluator_format="feyn",
                total_entries=total_entries,
                expression=expression,
                process=process,
            )

    raise ValueError(f"Unsupported input format: {detected!r}")


def write_selected_amplitude(path: Path, selected: SelectedAmplitude) -> None:
    """Write the selected entry in a layout accepted by evaluate_model."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if selected.evaluator_format == "raw":
            header = ["simple", "scrambled"]
            row = [selected.simple, selected.scrambled]
            if selected.process is not None:
                header.append("process")
                row.append(selected.process)
            writer.writerow(header)
            writer.writerow(row)
        elif selected.evaluator_format == "tokens":
            writer.writerow(["id", "tokens"])
            writer.writerow([1, selected.tokens])
        elif selected.evaluator_format == "token-pair":
            writer.writerow(["simple", "scrambled"])
            writer.writerow([selected.simple_tokens, selected.tokens])
        elif selected.evaluator_format == "feyn":
            writer.writerow([1, selected.expression])
        else:
            raise ValueError(
                f"Unsupported evaluator format: {selected.evaluator_format!r}"
            )


def read_first_gravity_process(
    metadata_path: Path,
    *,
    selected: SelectedAmplitude | None = None,
    tokenizer_max_particles: int = 8,
) -> str:
    """Read the first process and, when possible, prove row alignment."""

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Gravity metadata CSV does not exist: {metadata_path}")
    with open_csv_text(metadata_path, "r") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "process" not in reader.fieldnames:
            raise ValueError(
                f"Gravity metadata CSV must contain a 'process' column: {metadata_path}"
            )
        first_row: dict[str, str] | None = None
        total_entries = 0
        for row in reader:
            if not any((value or "").strip() for value in row.values()):
                continue
            total_entries += 1
            if first_row is None:
                first_row = row

    if first_row is None:
        raise ValueError(f"Gravity metadata CSV contains no data rows: {metadata_path}")
    process = (first_row.get("process") or "").strip()
    if process not in GRAVITY_PROCESSES:
        raise ValueError(
            f"First gravity metadata row has unsupported process {process!r}; "
            f"expected one of {sorted(GRAVITY_PROCESSES)}"
        )

    if selected is not None:
        if total_entries != selected.total_entries:
            raise ValueError(
                f"Gravity metadata has {total_entries} amplitude rows but the "
                f"input has {selected.total_entries}"
            )

        metadata_simple = (first_row.get("simple") or "").strip()
        metadata_scrambled = (first_row.get("scrambled") or "").strip()
        alignment_checked = False

        if selected.evaluator_format == "raw":
            if metadata_simple:
                alignment_checked = True
                if metadata_simple != selected.simple:
                    raise ValueError(
                        "First gravity metadata row is not aligned with the "
                        "selected amplitude's simple expression"
                    )
            if metadata_scrambled:
                alignment_checked = True
                if metadata_scrambled != selected.scrambled:
                    raise ValueError(
                        "First gravity metadata row is not aligned with the "
                        "selected amplitude's scrambled expression"
                    )
        elif selected.expression is not None:
            metadata_expressions = {
                value for value in (metadata_simple, metadata_scrambled) if value
            }
            if metadata_expressions:
                alignment_checked = True
                if selected.expression not in metadata_expressions:
                    raise ValueError(
                        "First gravity metadata row is not aligned with the "
                        "selected expression"
                    )
        elif selected.tokens is not None:
            if metadata_simple or metadata_scrambled:
                from data_gen.Tokenizer import ScatteringAmplitudeTokenizer

                tokenizer = ScatteringAmplitudeTokenizer(
                    max_particles=tokenizer_max_particles,
                    max_sequence_length=None,
                )
                encoded_simple = (
                    json.dumps(tokenizer.encode_infix(metadata_simple))
                    if metadata_simple
                    else None
                )
                encoded_scrambled = (
                    json.dumps(tokenizer.encode_infix(metadata_scrambled))
                    if metadata_scrambled
                    else None
                )
                if selected.simple_tokens is not None:
                    if (
                        encoded_simple is not None
                        and encoded_simple != selected.simple_tokens
                    ):
                        raise ValueError(
                            "First gravity metadata row is not aligned with the "
                            "selected simple token list"
                        )
                    if (
                        encoded_scrambled is not None
                        and encoded_scrambled != selected.tokens
                    ):
                        raise ValueError(
                            "First gravity metadata row is not aligned with the "
                            "selected scrambled token list"
                        )
                    alignment_checked = (
                        encoded_simple is not None or encoded_scrambled is not None
                    )
                else:
                    encoded_expressions = {
                        value
                        for value in (encoded_simple, encoded_scrambled)
                        if value is not None
                    }
                    if selected.tokens not in encoded_expressions:
                        raise ValueError(
                            "First gravity metadata row is not aligned with the "
                            "selected token list"
                        )
                    alignment_checked = True

            for key in ("tokens", "scrambled_tokens"):
                metadata_value = (first_row.get(key) or "").strip()
                if not metadata_value:
                    continue
                metadata_tokens = _maybe_json_token_list(
                    metadata_value,
                    label=f"metadata {key}",
                )
                if metadata_tokens is None:
                    continue
                alignment_checked = True
                if json.dumps(metadata_tokens) != selected.tokens:
                    raise ValueError(
                        "First gravity metadata row is not aligned with the "
                        "selected token list"
                    )
                break

        if not alignment_checked and selected.total_entries > 1:
            raise ValueError(
                "Gravity metadata does not contain expressions or token IDs "
                "that can verify alignment with the selected first amplitude; "
                "pass --gravity-process explicitly"
            )

    return process


def infer_gravity_metadata_path(raw_path: Path) -> Path | None:
    """Find the standard sibling metadata path for a raw gravity CSV."""

    replacements = (
        ("_raw.csv.gz", "_metadata.csv.gz"),
        ("_raw.csv", "_metadata.csv"),
        ("_tok.csv.gz", "_metadata.csv.gz"),
        ("_tok.csv", "_metadata.csv"),
    )
    for raw_suffix, metadata_suffix in replacements:
        if raw_path.name.endswith(raw_suffix):
            candidate = raw_path.with_name(
                raw_path.name[: -len(raw_suffix)] + metadata_suffix
            )
            if candidate.is_file():
                return candidate
    return None


def resolve_gravity_process(
    *,
    backend: str,
    selected: SelectedAmplitude,
    explicit_process: str | None,
    metadata_path: Path | None,
    source_path: Path,
    tokenizer_max_particles: int = 8,
) -> str | None:
    """Resolve and cross-check the process needed by the gravity oracle."""

    if backend != "gravity":
        if explicit_process is not None or metadata_path is not None:
            raise ValueError(
                "--gravity-process/--gravity-metadata-csv require "
                "--numeric-backend gravity"
            )
        return None

    metadata_process: str | None = None
    resolved_metadata = metadata_path
    if resolved_metadata is None and selected.process is None:
        resolved_metadata = infer_gravity_metadata_path(source_path)
    if resolved_metadata is not None:
        metadata_process = read_first_gravity_process(
            resolved_metadata,
            selected=selected,
            tokenizer_max_particles=tokenizer_max_particles,
        )

    candidates = {
        process
        for process in (explicit_process, selected.process, metadata_process)
        if process is not None
    }
    if len(candidates) > 1:
        raise ValueError(
            "Conflicting gravity process assignments for the first amplitude: "
            f"{sorted(candidates)}"
        )
    if not candidates:
        raise ValueError(
            "Gravity evaluation needs the first amplitude's process. Add a "
            "'process' column or pass --gravity-process 3s2h/4s1h."
        )
    process = next(iter(candidates))
    if process not in GRAVITY_PROCESSES:
        raise ValueError(
            f"Unsupported gravity process {process!r}; "
            f"expected one of {sorted(GRAVITY_PROCESSES)}"
        )
    return process


def default_output_stem(source_path: Path, backend: str) -> str:
    name = source_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".csv"):
        name = name[:-4]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-")
    return f"{safe_name or 'amplitude'}_first_{backend}_eval"


def build_evaluator_args(
    args: argparse.Namespace,
    *,
    selected_path: Path,
    selected: SelectedAmplitude,
    gravity_process: str | None,
) -> list[str]:
    """Translate the focused CLI into evaluate_model's CLI arguments."""

    backend = "ym" if args.numeric_backend == "yang-mills" else args.numeric_backend
    evaluator_args = [
        "--model-path",
        str(resolve_repo_path(args.model_path)),
        "--numeric-backend",
        backend,
        "--output-stem",
        args.output_stem or default_output_stem(resolve_repo_path(args.amplitude_csv), backend),
        "--decoding-method",
        args.decoding_method,
        "--rerank-numerical" if args.rerank_numerical else "--no-rerank-numerical",
        "--plots" if args.plots else "--no-plots",
        "--human-csv" if args.human_csv else "--no-human-csv",
        "--simple-summary" if args.simple_summary else "--no-simple-summary",
    ]

    if selected.evaluator_format == "raw":
        evaluator_args.extend(
            [
                "--data-source",
                "csv",
                "--existing-raw-csv",
                str(selected_path),
                "--existing-csv-max-rows",
                "1",
                "--no-dedupe",
            ]
        )
    else:
        evaluator_args.extend(
            [
                "--data-source",
                "single-amplitude",
                "--single-amplitude-input-csv",
                str(selected_path),
                "--single-amplitude-input-format",
                selected.evaluator_format,
            ]
        )
        if selected.evaluator_format == "tokens":
            evaluator_args.extend(["--tokens-column", "tokens"])
        elif selected.evaluator_format == "feyn":
            evaluator_args.extend(["--single-amplitude-expression-column", "1"])

    scalar_options = (
        ("--device", args.device),
        ("--n-particles", args.n_particles),
        ("--numeric-mass", args.numeric_mass),
        ("--numeric-energy-scale", args.numeric_energy_scale),
        ("--numeric-tol-abs", args.numeric_tol_abs),
        ("--numeric-tol-rel", args.numeric_tol_rel),
        ("--input-token-limit", args.input_token_limit),
        ("--max-decode-tokens", args.max_decode_tokens),
        ("--tokenizer-max-particles", args.tokenizer_max_particles),
    )
    for flag, value in scalar_options:
        if value is not None:
            evaluator_args.extend([flag, str(value)])

    if args.decoding_method in {"beam", "nucleus"}:
        evaluator_args.extend(["--beam-size", str(args.beam_size)])
    if args.p_nucleus is not None:
        evaluator_args.extend(["--p-nucleus", str(args.p_nucleus)])
    if args.temperature_nucleus is not None:
        evaluator_args.extend(
            ["--temperature-nucleus", str(args.temperature_nucleus)]
        )
    if args.numeric_pol_modes is not None:
        evaluator_args.append("--numeric-pol-modes")
        evaluator_args.extend(args.numeric_pol_modes)
    if args.gravity_reference_modes is not None:
        evaluator_args.append("--gravity-reference-modes")
        evaluator_args.extend(args.gravity_reference_modes)
    if args.gravity_gauge_shift is not None:
        evaluator_args.append(
            "--gravity-gauge-shift"
            if args.gravity_gauge_shift
            else "--no-gravity-gauge-shift"
        )
    if gravity_process is not None:
        evaluator_args.extend(["--gravity-process", gravity_process])

    return evaluator_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the first amplitude in a CSV with a trained model and "
            "the strict SQED, Yang--Mills, or gravity numerical oracle."
        ),
    )
    parser.add_argument("model_path", help="Trained model checkpoint (.pt).")
    parser.add_argument("amplitude_csv", help="Input .csv or .csv.gz amplitude file.")
    parser.add_argument(
        "--numeric-backend",
        required=True,
        choices=["sqed", "ym", "yang-mills", "gravity"],
        help="Numerical oracle appropriate for the amplitude.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "raw", "tokens", "token-pair", "expression", "feyn"],
        default="auto",
        help="CSV layout; auto recognizes the supported repository formats.",
    )
    parser.add_argument(
        "--tokens-column",
        default="tokens",
        help="Header name containing a JSON list of token IDs.",
    )
    parser.add_argument(
        "--expression-column",
        default=None,
        help=(
            "Expression header name or zero-based column index. Headerless "
            "Feynman data defaults to column 1, falling back to column 0."
        ),
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="External legs (defaults to 4, or 5 for gravity).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--decoding-method",
        choices=["greedy", "beam", "nucleus"],
        default="greedy",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--p-nucleus", type=float, default=None)
    parser.add_argument("--temperature-nucleus", type=float, default=None)
    parser.add_argument(
        "--rerank-numerical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For beam/nucleus, prefer the shortest numerically equivalent candidate.",
    )
    parser.add_argument("--input-token-limit", type=int, default=None)
    parser.add_argument("--max-decode-tokens", type=int, default=None)
    parser.add_argument("--tokenizer-max-particles", type=int, default=None)
    parser.add_argument("--numeric-mass", type=float, default=None)
    parser.add_argument("--numeric-energy-scale", type=float, default=None)
    parser.add_argument(
        "--numeric-pol-modes",
        nargs="+",
        choices=["coulomb", "covariant"],
        default=None,
    )
    parser.add_argument("--numeric-tol-abs", type=float, default=None)
    parser.add_argument("--numeric-tol-rel", type=float, default=None)
    parser.add_argument(
        "--gravity-process",
        choices=sorted(GRAVITY_PROCESSES),
        default=None,
        help="Required for gravity unless the selected row/metadata supplies it.",
    )
    parser.add_argument(
        "--gravity-metadata-csv",
        default=None,
        help="Optional aligned metadata CSV; only its first process row is used.",
    )
    parser.add_argument(
        "--gravity-reference-modes",
        nargs="+",
        choices=["first", "last", "random", "cyclic"],
        default=None,
    )
    parser.add_argument(
        "--gravity-gauge-shift",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Stem for files written under data_testing/outputs.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate diagnostic plots (normally unnecessary for one row).",
    )
    parser.add_argument(
        "--human-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--simple-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_path = resolve_repo_path(args.amplitude_csv)

    try:
        selected = select_first_amplitude(
            source_path,
            input_format=args.input_format,
            tokens_column=args.tokens_column,
            expression_column=args.expression_column,
        )
        metadata_path = (
            resolve_repo_path(args.gravity_metadata_csv)
            if args.gravity_metadata_csv is not None
            else None
        )
        backend = "ym" if args.numeric_backend == "yang-mills" else args.numeric_backend
        gravity_process = resolve_gravity_process(
            backend=backend,
            selected=selected,
            explicit_process=args.gravity_process,
            metadata_path=metadata_path,
            source_path=source_path,
            tokenizer_max_particles=args.tokenizer_max_particles or 8,
        )
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    plural = "entry" if selected.total_entries == 1 else "entries"
    if selected.total_entries == 1:
        print(
            f"Found 1 amplitude entry in {source_path}; using it "
            f"(format: {selected.source_format})."
        )
    else:
        print(
            f"Found {selected.total_entries} amplitude {plural} in {source_path}; "
            f"using the first and ignoring {selected.total_entries - 1} "
            f"(format: {selected.source_format})."
        )

    with tempfile.TemporaryDirectory(prefix="single_amplitude_eval_") as temp_dir:
        selected_path = Path(temp_dir) / "first_amplitude.csv"
        write_selected_amplitude(selected_path, selected)
        evaluator_args = build_evaluator_args(
            args,
            selected_path=selected_path,
            selected=selected,
            gravity_process=gravity_process,
        )

        # Import lazily so CSV selection and --help do not require PyTorch.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from data_testing import evaluate_model

        return evaluate_model.main(evaluator_args)


if __name__ == "__main__":
    raise SystemExit(main())
