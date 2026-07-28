#!/usr/bin/env python3
"""Verify that every source-minus-cleaned amplitude is exactly zero.

The cleaned dataset can contain fewer rows because the filtering pass drops
source rows whose entire ``simple`` expression vanishes.  Pairing is therefore
source-driven:

* a clean or mixed source row consumes one cleaned counterpart;
* an all-zero source row consumes no cleaned row and is compared with zero.

For each retained pair, this verifier forms ``old_simple - new_simple`` as a
signed multiset of top-level summands.  Exact matching summands cancel.  Every
remaining summand must then be proven zero by the same field-strength rules
and the same explicit, label-scoped on-shell assumptions used by
``filter_antisymmetry_zeros.py``.  This is an exact structural check, not a
floating-point near-zero test.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

if __package__:
    from .filter_antisymmetry_zeros import (
        OnShellAssumptions,
        _open_csv_text,
        _safe_resolve,
        _set_csv_field_size_limit,
        _write_json_report,
        add_onshell_cli_arguments,
        analyze_simple_expression,
        assumptions_report,
        onshell_assumptions_from_namespace,
        split_top_level_sum,
        zero_summand_reasons,
    )
else:
    from filter_antisymmetry_zeros import (
        OnShellAssumptions,
        _open_csv_text,
        _safe_resolve,
        _set_csv_field_size_limit,
        _write_json_report,
        add_onshell_cli_arguments,
        analyze_simple_expression,
        assumptions_report,
        onshell_assumptions_from_namespace,
        split_top_level_sum,
        zero_summand_reasons,
    )


class VerificationError(ValueError):
    """Raised when the processed dataset is not exactly source-equivalent."""


@dataclass(frozen=True)
class DeltaAnalysis:
    """Reduction of one ``old - new`` expression."""

    term_multiplicities: tuple[tuple[str, int], ...]
    zero_summand_count: int
    reason_counts: Counter[str]


@dataclass
class VerificationStats:
    """Counters for a complete source-to-cleaned reconciliation."""

    source_rows: int = 0
    cleaned_rows: int = 0
    paired_rows: int = 0
    clean_source_rows: int = 0
    mixed_source_rows: int = 0
    identical_pairs: int = 0
    zero_delta_modified_pairs: int = 0
    dropped_zero_rows: int = 0
    zero_differences: int = 0
    delta_zero_summands: int = 0
    non_simple_fields_compared: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reason_counts"] = dict(sorted(self.reason_counts.items()))
        return result


def _signed_term(term: str) -> tuple[int, str]:
    """Return a leading algebraic sign and sign-free term text."""

    text = term.strip()
    sign = 1
    if text.startswith("+"):
        text = text[1:].lstrip()
    elif text.startswith("-"):
        sign = -1
        text = text[1:].lstrip()
    if not text:
        raise VerificationError("encountered an empty signed summand")
    return sign, text


def signed_term_counter(expression: str | None) -> Counter[str]:
    """Represent a sum as exact signed multiplicities of its term bodies."""

    if expression is None:
        return Counter()
    compact = expression.strip().replace(" ", "")
    if compact in {"0", "+0", "-0"}:
        return Counter()

    terms: Counter[str] = Counter()
    for raw_term in split_top_level_sum(expression):
        sign, body = _signed_term(raw_term)
        terms[body] += sign
        if terms[body] == 0:
            del terms[body]
    return terms


def subtract_expressions(
    old_expression: str,
    new_expression: str | None,
    *,
    assumptions: OnShellAssumptions | None = None,
) -> DeltaAnalysis:
    """Form ``old - new`` and prove every uncancelled summand is zero."""

    delta = signed_term_counter(old_expression)
    delta.subtract(signed_term_counter(new_expression))
    delta = Counter({term: count for term, count in delta.items() if count})

    reason_counts: Counter[str] = Counter()
    nonzero_remainders: list[tuple[str, int]] = []
    zero_summand_count = 0
    for term, multiplicity in sorted(delta.items()):
        reasons = zero_summand_reasons(
            term,
            assumptions=assumptions,
        )
        if not reasons:
            nonzero_remainders.append((term, multiplicity))
            continue
        occurrences = abs(multiplicity)
        zero_summand_count += occurrences
        for reason in reasons:
            reason_counts[reason.code] += occurrences

    if nonzero_remainders:
        previews = [
            f"{coefficient:+d} * {term[:300]}"
            for term, coefficient in nonzero_remainders[:5]
        ]
        raise VerificationError(
            "old - new contains unproved nonzero summand(s): "
            + "; ".join(previews)
        )

    return DeltaAnalysis(
        term_multiplicities=tuple(sorted(delta.items())),
        zero_summand_count=zero_summand_count,
        reason_counts=reason_counts,
    )


def _validate_csv_row(
    row: dict[str | None, str | None],
    *,
    csv_name: str,
    line_number: int,
) -> None:
    if None in row:
        raise VerificationError(
            f"{csv_name} row {line_number} has more fields than its header"
        )
    if any(value is None for value in row.values()):
        raise VerificationError(
            f"{csv_name} row {line_number} has fewer fields than its header"
        )


def verify_filtered_csv(
    source_path: Path,
    cleaned_path: Path,
    *,
    assumptions: OnShellAssumptions | None = None,
    progress_every: int = 50_000,
) -> VerificationStats:
    """Reconcile source and ``prune-terms`` output CSV files exactly."""

    if _safe_resolve(source_path) == _safe_resolve(cleaned_path):
        raise VerificationError("source and cleaned paths must differ")
    _set_csv_field_size_limit()
    stats = VerificationStats()

    with (
        _open_csv_text(source_path, "r") as source_handle,
        _open_csv_text(cleaned_path, "r") as cleaned_handle,
    ):
        source_reader = csv.DictReader(source_handle)
        cleaned_reader = csv.DictReader(cleaned_handle)
        if source_reader.fieldnames is None or cleaned_reader.fieldnames is None:
            raise VerificationError("both CSV files must have headers")
        if source_reader.fieldnames != cleaned_reader.fieldnames:
            raise VerificationError(
                "CSV headers differ: "
                f"source={source_reader.fieldnames}, "
                f"cleaned={cleaned_reader.fieldnames}"
            )
        if "simple" not in source_reader.fieldnames:
            raise VerificationError(
                f"CSV must contain a 'simple' column; got {source_reader.fieldnames}"
            )
        if len(source_reader.fieldnames) != len(set(source_reader.fieldnames)):
            raise VerificationError(
                f"CSV contains duplicate headers: {source_reader.fieldnames}"
            )

        cleaned_iterator = iter(cleaned_reader)
        for source_row in source_reader:
            stats.source_rows += 1
            _validate_csv_row(
                source_row,
                csv_name="source",
                line_number=source_reader.line_num,
            )
            old_simple = source_row["simple"]
            assert old_simple is not None
            source_analysis = analyze_simple_expression(
                old_simple,
                assumptions=assumptions,
            )

            if source_analysis.classification == "all_zero":
                try:
                    delta = subtract_expressions(
                        old_simple,
                        None,
                        assumptions=assumptions,
                    )
                except VerificationError as exc:
                    raise VerificationError(
                        f"source row {source_reader.line_num}, compared with zero: "
                        f"{exc}"
                    ) from exc
                stats.dropped_zero_rows += 1
            else:
                if source_analysis.classification == "clean":
                    stats.clean_source_rows += 1
                else:
                    stats.mixed_source_rows += 1
                try:
                    cleaned_row = next(cleaned_iterator)
                except StopIteration as exc:
                    raise VerificationError(
                        f"cleaned CSV ended before source row "
                        f"{source_reader.line_num}"
                    ) from exc
                stats.cleaned_rows += 1
                stats.paired_rows += 1
                _validate_csv_row(
                    cleaned_row,
                    csv_name="cleaned",
                    line_number=cleaned_reader.line_num,
                )

                for field_name in source_reader.fieldnames:
                    if field_name == "simple":
                        continue
                    stats.non_simple_fields_compared += 1
                    if source_row[field_name] != cleaned_row[field_name]:
                        raise VerificationError(
                            f"source row {source_reader.line_num} and cleaned "
                            f"row {cleaned_reader.line_num} differ in "
                            f"non-simple field {field_name!r}"
                        )

                new_simple = cleaned_row["simple"]
                assert new_simple is not None
                assert source_analysis.cleaned is not None
                if new_simple != source_analysis.cleaned:
                    raise VerificationError(
                        f"source row {source_reader.line_num} produced an "
                        f"unexpected cleaned expression at cleaned row "
                        f"{cleaned_reader.line_num}; expected the exact ordered "
                        "source terms after removing only proven-zero summands"
                    )
                cleaned_analysis = analyze_simple_expression(
                    new_simple,
                    assumptions=assumptions,
                )
                if cleaned_analysis.classification != "clean":
                    raise VerificationError(
                        f"cleaned row {cleaned_reader.line_num} still contains "
                        f"{len(cleaned_analysis.zero_summands)} zero summand(s)"
                    )
                try:
                    delta = subtract_expressions(
                        old_simple,
                        new_simple,
                        assumptions=assumptions,
                    )
                except VerificationError as exc:
                    raise VerificationError(
                        f"source row {source_reader.line_num} minus cleaned row "
                        f"{cleaned_reader.line_num}: {exc}"
                    ) from exc

                if source_analysis.classification == "mixed":
                    stats.zero_delta_modified_pairs += 1
                else:
                    stats.identical_pairs += 1

            stats.zero_differences += 1
            stats.delta_zero_summands += len(source_analysis.zero_summands)
            for zero_summand in source_analysis.zero_summands:
                stats.reason_counts.update(
                    reason.code for reason in zero_summand.reasons
                )

            if progress_every and stats.source_rows % progress_every == 0:
                print(
                    f"verified {stats.source_rows:,} source rows "
                    f"({stats.zero_differences:,} zero differences)",
                    file=sys.stderr,
                )

        try:
            extra_cleaned_row = next(cleaned_iterator)
        except StopIteration:
            extra_cleaned_row = None
        if extra_cleaned_row is not None:
            raise VerificationError(
                f"cleaned CSV has extra data beginning at row "
                f"{cleaned_reader.line_num}"
            )

    if stats.source_rows != stats.zero_differences:
        raise VerificationError(
            f"only {stats.zero_differences:,}/{stats.source_rows:,} "
            "source differences were proven zero"
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify row-by-row that original_simple - cleaned_simple reduces "
            "exactly to zero under the antisymmetry rules."
        )
    )
    parser.add_argument("source", type=Path, help="original .csv or .csv.gz")
    parser.add_argument("cleaned", type=Path, help="processed .csv or .csv.gz")
    add_onshell_cli_arguments(parser)
    parser.add_argument(
        "--report-json",
        type=Path,
        help="optional machine-readable verification report",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="emit progress every N source rows; use 0 to disable",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    assumptions = onshell_assumptions_from_namespace(args, parser)

    source_path = args.source.expanduser()
    cleaned_path = args.cleaned.expanduser()
    for label, path in (("source", source_path), ("cleaned", cleaned_path)):
        if not path.is_file():
            raise SystemExit(f"{label} file does not exist: {path}")

    try:
        report_path = (
            args.report_json.expanduser()
            if args.report_json is not None
            else None
        )
        if report_path is not None:
            report_resolved = _safe_resolve(report_path)
            if report_resolved in {
                _safe_resolve(source_path),
                _safe_resolve(cleaned_path),
            }:
                raise VerificationError(
                    "the JSON report path must differ from both CSV files"
                )
            if report_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"report already exists: {report_path} "
                    "(pass --overwrite to replace it)"
                )

        stats = verify_filtered_csv(
            source_path,
            cleaned_path,
            assumptions=assumptions,
            progress_every=args.progress_every,
        )
        report: dict[str, object] = {
            "report_schema_version": 1,
            "verification": "passed",
            "source": str(_safe_resolve(source_path)),
            "cleaned": str(_safe_resolve(cleaned_path)),
            "on_shell_assumptions": assumptions_report(assumptions),
            "stats": stats.to_dict(),
        }
        if report_path is not None:
            _write_json_report(
                report_path,
                report,
                overwrite=args.overwrite,
            )
        print(json.dumps(report, indent=2, sort_keys=True))
    except (OSError, csv.Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
