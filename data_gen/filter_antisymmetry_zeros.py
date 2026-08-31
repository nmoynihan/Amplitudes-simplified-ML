#!/usr/bin/env python3
"""Remove manifestly zero field-strength summands from amplitude CSV files.

With both Lorentz indices lowered, a field strength is antisymmetric,
``F_{a,mu nu} = -F_{a,nu mu}``.  As a mixed-index operator in the
repository's Minkowski contractions, it is skew-adjoint with respect to the
metric rather than under ordinary Euclidean transpose:

    F_a^T eta = -eta F_a.

Consequently, for ``W = F_{a1} F_{a2} ... F_{an}``,

    p · W · q = (-1)^n q · F_{an} ... F_{a2} F_{a1} · p.

An open chain ``p_i · W · p_i`` is therefore zero when the label word is an
odd-length palindrome.  Taking a trace gives
``Tr(W) = (-1)^n Tr(reverse(W))``.  Trace cyclicity therefore makes an odd
trace zero when its reversed word is a cyclic rotation of the original word.

The default ``prune-terms`` mode removes only the proven-zero summands from
the ``simple`` column.  It keeps mixed rows (the unchanged ``scrambled``
expression is still algebraically equivalent) and drops rows for which every
simple summand vanishes.  ``drop-rows`` is available for stricter dataset
sanitisation.

Both plain CSV and ``.csv.gz`` files are streamed, so the script can process
datasets much larger than memory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence, TextIO


_DOT = r"[·.]"
_CHAIN_RE = re.compile(
    rf"p_(\d+)((?:\s*{_DOT}\s*F_\d+)+)\s*{_DOT}\s*p_(\d+)"
)
_TRACE_RE = re.compile(
    rf"Tr\((F_\d+(?:\s*[·.,]\s*F_\d+)*)\)"
)
_PP_RE = re.compile(rf"p_(\d+)\s*{_DOT}\s*p_(\d+)")
_INTEGER_RE = re.compile(r"\d+")
_ZERO_RE = re.compile(r"[+-]?0+(?:\.0*)?")
_POWER_RE = re.compile(r"\((.+)\)\s*\^\s*(\d+)", re.DOTALL)
_BARE_POWER_RE = re.compile(r"(.+?)\s*\^\s*(\d+)", re.DOTALL)


class ExpressionSyntaxError(ValueError):
    """Raised when a simple expression cannot be split safely."""


@dataclass(frozen=True)
class OnShellAssumptions:
    """Labels for which masslessness and transversality are explicitly known.

    A field-strength label must occur in both sets before boundary
    transversality or ``F_i^3 = 0`` may be used.  Keeping these assumptions
    label-scoped prevents an all-massless Yang--Mills shortcut from being
    applied accidentally to massive SQED endpoints.
    """

    massless_momenta: frozenset[int]
    transverse_field_strengths: frozenset[int]

    def __post_init__(self) -> None:
        massless = frozenset(self.massless_momenta)
        transverse = frozenset(self.transverse_field_strengths)
        for name, labels in (
            ("massless_momenta", massless),
            ("transverse_field_strengths", transverse),
        ):
            invalid = [
                label
                for label in labels
                if isinstance(label, bool)
                or not isinstance(label, int)
                or label <= 0
            ]
            if invalid:
                raise ValueError(
                    f"{name} must contain only positive integer labels; "
                    f"got {invalid!r}"
                )
        object.__setattr__(self, "massless_momenta", massless)
        object.__setattr__(self, "transverse_field_strengths", transverse)

    @classmethod
    def all_massless_ym(cls, particle_count: int) -> "OnShellAssumptions":
        """Construct all-massless transverse assumptions for labels ``1..N``."""

        if (
            isinstance(particle_count, bool)
            or not isinstance(particle_count, int)
            or particle_count <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        labels = frozenset(range(1, particle_count + 1))
        return cls(labels, labels)

    def supports_transverse_field_strength(self, label: int) -> bool:
        """Return whether both prerequisites for on-shell ``F_label`` hold."""

        return (
            label in self.massless_momenta
            and label in self.transverse_field_strengths
        )

    def to_dict(self) -> dict[str, list[int]]:
        """Return a stable JSON representation of the scientific assumptions."""

        return {
            "massless_momenta": sorted(self.massless_momenta),
            "transverse_field_strengths": sorted(
                self.transverse_field_strengths
            ),
        }


@dataclass(frozen=True)
class ZeroReason:
    """One exact reason why a multiplicative numerator factor is zero."""

    code: str
    factor: str
    details: str


@dataclass(frozen=True)
class ZeroSummand:
    """A top-level summand proven to vanish."""

    term_index: int
    term: str
    reasons: tuple[ZeroReason, ...]


@dataclass(frozen=True)
class ExpressionAnalysis:
    """Classification and optional cleaned form of one simple expression."""

    original: str
    cleaned: str | None
    term_count: int
    zero_summands: tuple[ZeroSummand, ...]

    @property
    def classification(self) -> str:
        if not self.zero_summands:
            return "clean"
        if self.cleaned is None:
            return "all_zero"
        return "mixed"


@dataclass
class FilterStats:
    """Streaming counters written to stdout and, optionally, JSON."""

    input_rows: int = 0
    output_rows: int = 0
    clean_rows: int = 0
    mixed_rows: int = 0
    all_zero_rows: int = 0
    rows_modified: int = 0
    rows_dropped: int = 0
    terms_examined: int = 0
    zero_summands: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reason_counts"] = dict(sorted(self.reason_counts.items()))
        return result


def _set_csv_field_size_limit() -> None:
    """Raise the CSV field limit for long scrambled amplitudes."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _validate_parentheses(text: str) -> None:
    depth = 0
    for position, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ExpressionSyntaxError(
                    f"unmatched ')' at character {position}"
                )
    if depth:
        raise ExpressionSyntaxError(f"{depth} unmatched '(' character(s)")


def strip_outer_parens(text: str) -> str:
    """Strip only parentheses enclosing the complete expression."""

    result = text.strip()
    _validate_parentheses(result)
    while result.startswith("(") and result.endswith(")"):
        depth = 0
        encloses_all = True
        for position, char in enumerate(result):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and position < len(result) - 1:
                encloses_all = False
                break
        if not encloses_all:
            break
        result = result[1:-1].strip()
    return result


def split_top_level(text: str, separator: str) -> list[str]:
    """Split on a one-character separator at parenthesis depth zero."""

    if len(separator) != 1:
        raise ValueError("separator must contain exactly one character")
    _validate_parentheses(text)
    pieces: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == separator and depth == 0:
            piece = "".join(current).strip()
            if not piece:
                raise ExpressionSyntaxError(
                    f"empty expression around top-level '{separator}'"
                )
            pieces.append(piece)
            current = []
        else:
            current.append(char)
    piece = "".join(current).strip()
    if not piece:
        raise ExpressionSyntaxError(
            f"empty expression around top-level '{separator}'"
        )
    pieces.append(piece)
    return pieces


def split_top_level_sum(expression: str) -> list[str]:
    """Return signed top-level summands without splitting nested polynomials."""

    expr = expression.strip()
    if not expr:
        raise ExpressionSyntaxError("empty expression")
    _validate_parentheses(expr)

    pieces: list[str] = []
    current: list[str] = []
    depth = 0
    for position, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and char in "+-" and position > 0:
            previous_position = position - 1
            while previous_position >= 0 and expr[previous_position].isspace():
                previous_position -= 1
            previous = expr[previous_position] if previous_position >= 0 else ""
            is_exponent_sign = (
                previous in "eE"
                and previous_position == position - 1
                and previous_position > 0
                and (
                    expr[previous_position - 1].isdigit()
                    or expr[previous_position - 1] == "."
                )
            )
            if not is_exponent_sign and previous not in "*/^(+-":
                piece = "".join(current).strip()
                if not piece:
                    raise ExpressionSyntaxError(
                        f"empty summand before character {position}"
                    )
                pieces.append(piece)
                current = [char]
                continue
        current.append(char)

    final_piece = "".join(current).strip()
    if not final_piece or final_piece in {"+", "-"}:
        raise ExpressionSyntaxError("expression ends with an empty summand")
    pieces.append(final_piece)
    return pieces


def _strip_term_prefix(term: str) -> tuple[int, str]:
    """Return the integer sign/coefficient and the remaining term body."""

    body = strip_outer_parens(term)
    sign = 1
    if body.startswith("+"):
        body = body[1:].lstrip()
    elif body.startswith("-"):
        sign = -1
        body = body[1:].lstrip()
    if not body:
        raise ExpressionSyntaxError("summand contains only a sign")

    depth = 0
    for position, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "*" and depth == 0:
            possible_coefficient = body[:position].strip()
            if _INTEGER_RE.fullmatch(possible_coefficient):
                return sign * int(possible_coefficient), body[position + 1 :].strip()
            break

    if _INTEGER_RE.fullmatch(body):
        return sign * int(body), "1"
    return sign, body


def split_term_num_den(term_body: str) -> tuple[str, str]:
    """Return numerator and denominator from one unsigned top-level term."""

    body = strip_outer_parens(term_body)
    depth = 0
    slash_position: int | None = None
    for position, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "/" and depth == 0:
            if slash_position is not None:
                raise ExpressionSyntaxError(
                    "multiple top-level division operators in one summand"
                )
            slash_position = position

    if slash_position is None:
        return body, ""
    numerator = strip_outer_parens(body[:slash_position])
    denominator = strip_outer_parens(body[slash_position + 1 :])
    if not numerator or not denominator:
        raise ExpressionSyntaxError("empty numerator or denominator")
    return numerator, denominator


def product_factors(product: str) -> Iterator[tuple[str, int]]:
    """Yield ``(base, positive_integer_power)`` for top-level factors."""

    expression = strip_outer_parens(product)
    for raw_factor in split_top_level(expression, "*"):
        raw = raw_factor.strip()
        power_match = _POWER_RE.fullmatch(raw)
        if power_match is None:
            power_match = _BARE_POWER_RE.fullmatch(raw)
        if power_match is None:
            base = strip_outer_parens(raw)
            power = 1
        else:
            base = strip_outer_parens(power_match.group(1))
            power = int(power_match.group(2))
        if power < 0:
            raise ExpressionSyntaxError("negative factor powers are unsupported")
        yield base, power


def _cyclically_equal(left: Sequence[int], right: Sequence[int]) -> bool:
    if len(left) != len(right):
        return False
    if not left:
        return True
    target = tuple(right)
    doubled = tuple(left) + tuple(left)
    width = len(left)
    return any(doubled[start : start + width] == target for start in range(width))


@lru_cache(maxsize=None)
def _chain_is_proven_zero(
    left: int,
    word: tuple[int, ...],
    right: int,
    assumptions: OnShellAssumptions | None,
) -> bool:
    """Prove a momentum/F-chain zero using exact recursive F splitting."""

    if not word:
        return bool(
            assumptions is not None
            and left == right
            and left in assumptions.massless_momenta
        )
    if left == right and len(word) % 2 == 1 and word == word[::-1]:
        return True
    if assumptions is not None:
        if (
            left == word[0]
            and assumptions.supports_transverse_field_strength(left)
        ):
            return True
        if (
            right == word[-1]
            and assumptions.supports_transverse_field_strength(right)
        ):
            return True
        if any(
            word[position] == word[position + 1] == word[position + 2]
            and assumptions.supports_transverse_field_strength(word[position])
            for position in range(max(0, len(word) - 2))
        ):
            return True

    return any(
        _chain_is_proven_zero(
            left,
            word[:position],
            field_label,
            assumptions,
        )
        and _chain_is_proven_zero(
            field_label,
            word[position + 1 :],
            right,
            assumptions,
        )
        for position, field_label in enumerate(word)
    )


def _chain_has_proven_zero_split(
    left: int,
    word: tuple[int, ...],
    right: int,
    assumptions: OnShellAssumptions | None,
) -> bool:
    """Return whether expanding one F leaves two proven-zero subchains."""

    return any(
        _chain_is_proven_zero(
            left,
            word[:position],
            field_label,
            assumptions,
        )
        and _chain_is_proven_zero(
            field_label,
            word[position + 1 :],
            right,
            assumptions,
        )
        for position, field_label in enumerate(word)
    )


def zero_factor_reasons(
    factor: str,
    *,
    assumptions: OnShellAssumptions | None = None,
) -> tuple[ZeroReason, ...]:
    """Return exact zero identities applying to one numerator factor."""

    base = strip_outer_parens(factor)
    reasons: list[ZeroReason] = []

    if _ZERO_RE.fullmatch(base):
        reasons.append(
            ZeroReason("explicit_zero", base, "factor is the scalar zero")
        )
        return tuple(reasons)

    chain_match = _CHAIN_RE.fullmatch(base)
    if chain_match:
        left = int(chain_match.group(1))
        word = tuple(int(label) for label in re.findall(r"F_(\d+)", chain_match.group(2)))
        right = int(chain_match.group(3))
        if left == right and len(word) % 2 == 1 and word == word[::-1]:
            reasons.append(
                ZeroReason(
                    "antisymmetric_palindrome_chain",
                    base,
                    f"same endpoint p_{left}; odd palindromic F word {word}",
                )
            )
        if _chain_has_proven_zero_split(
            left,
            word,
            right,
            assumptions,
        ):
            reasons.append(
                ZeroReason(
                    "field_strength_split_chain",
                    base,
                    "expanding an interior field strength leaves a "
                    "proven-zero momentum/F-chain in both terms",
                )
            )
        if assumptions is not None:
            if (
                left == word[0]
                and assumptions.supports_transverse_field_strength(left)
            ):
                reasons.append(
                    ZeroReason(
                        "ym_left_self_contraction",
                        base,
                        f"massless transverse p_{left} · F_{word[0]} = 0",
                    )
                )
            if (
                right == word[-1]
                and assumptions.supports_transverse_field_strength(right)
            ):
                reasons.append(
                    ZeroReason(
                        "ym_right_self_contraction",
                        base,
                        f"massless transverse F_{word[-1]} · p_{right} = 0",
                    )
                )
            if any(
                word[position] == word[position + 1] == word[position + 2]
                and assumptions.supports_transverse_field_strength(
                    word[position]
                )
                for position in range(max(0, len(word) - 2))
            ):
                reasons.append(
                    ZeroReason(
                        "ym_nilpotent_field_cube",
                        base,
                        "contains F_i · F_i · F_i = 0 for a declared "
                        "massless transverse field strength",
                    )
                )
        return tuple(reasons)

    trace_match = _TRACE_RE.fullmatch(base)
    if trace_match:
        word = tuple(int(label) for label in re.findall(r"F_(\d+)", trace_match.group(1)))
        if (
            len(word) % 2 == 1
            and _cyclically_equal(word, tuple(reversed(word)))
        ):
            reasons.append(
                ZeroReason(
                    "antisymmetric_cyclic_trace",
                    base,
                    f"odd F word {word} equals its reversal up to a cyclic rotation",
                )
            )
        if assumptions is not None and len(word) >= 3 and any(
            word[position] == word[(position + 1) % len(word)]
            == word[(position + 2) % len(word)]
            and assumptions.supports_transverse_field_strength(
                word[position]
            )
            for position in range(len(word))
        ):
            reasons.append(
                ZeroReason(
                    "ym_nilpotent_field_cube",
                    base,
                    "cyclic trace contains F_i · F_i · F_i = 0 for a "
                    "declared massless transverse field strength",
                )
            )
        return tuple(reasons)

    if assumptions is not None:
        momentum_match = _PP_RE.fullmatch(base)
        if (
            momentum_match
            and momentum_match.group(1) == momentum_match.group(2)
            and int(momentum_match.group(1)) in assumptions.massless_momenta
        ):
            label = int(momentum_match.group(1))
            reasons.append(
                ZeroReason(
                    "ym_massless_momentum_square",
                    base,
                    f"p_{label} · p_{label} = 0 for a declared "
                    "massless momentum",
                )
            )
    return tuple(reasons)


def zero_summand_reasons(
    summand: str,
    *,
    assumptions: OnShellAssumptions | None = None,
) -> tuple[ZeroReason, ...]:
    """Return reasons proving that a complete top-level summand is zero."""

    coefficient, body = _strip_term_prefix(summand)
    if coefficient == 0:
        return (
            ZeroReason(
                "explicit_zero_coefficient",
                summand.strip(),
                "top-level coefficient is zero",
            ),
        )

    numerator, _denominator = split_term_num_den(body)
    reasons: list[ZeroReason] = []
    for factor, power in product_factors(numerator):
        if power == 0:
            continue
        reasons.extend(
            zero_factor_reasons(
                factor,
                assumptions=assumptions,
            )
        )
    return tuple(reasons)


def _join_signed_terms(terms: Sequence[str]) -> str:
    """Join already-signed summands without changing their algebraic signs."""

    if not terms:
        raise ValueError("cannot join an empty term sequence")
    pieces: list[str] = []
    for position, raw_term in enumerate(terms):
        term = raw_term.strip()
        if position == 0:
            if term.startswith("+"):
                term = term[1:].lstrip()
            elif term.startswith("-"):
                term = "-" + term[1:].lstrip()
            pieces.append(term)
            continue
        if term.startswith("+"):
            pieces.append("+ " + term[1:].lstrip())
        elif term.startswith("-"):
            pieces.append("- " + term[1:].lstrip())
        else:
            pieces.append("+ " + term)
    return " ".join(pieces)


def analyze_simple_expression(
    expression: str,
    *,
    assumptions: OnShellAssumptions | None = None,
) -> ExpressionAnalysis:
    """Classify an expression and remove only individually proven-zero summands."""

    terms = split_top_level_sum(expression)
    kept_terms: list[str] = []
    zero_summands: list[ZeroSummand] = []
    for term_index, term in enumerate(terms):
        reasons = zero_summand_reasons(
            term,
            assumptions=assumptions,
        )
        if reasons:
            zero_summands.append(ZeroSummand(term_index, term, reasons))
        else:
            kept_terms.append(term)

    if not zero_summands:
        cleaned = expression
    elif kept_terms:
        cleaned = _join_signed_terms(kept_terms)
    else:
        cleaned = None
    return ExpressionAnalysis(
        original=expression,
        cleaned=cleaned,
        term_count=len(terms),
        zero_summands=tuple(zero_summands),
    )


def _open_csv_text(
    path: Path,
    mode: str,
    *,
    compressed: bool | None = None,
) -> TextIO:
    if compressed is None:
        compressed = path.suffix.lower() == ".gz"
    if compressed:
        return gzip.open(
            path,
            mode + "t",
            newline="",
            encoding="utf-8",
            compresslevel=6,
        )
    return path.open(mode, newline="", encoding="utf-8")


def _default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".csv.gz"):
        name = name[:-7] + "_antisymmetry_clean.csv.gz"
    elif name.endswith(".gz"):
        name = name[:-3] + "_antisymmetry_clean.gz"
    elif name.endswith(".csv"):
        name = name[:-4] + "_antisymmetry_clean.csv"
    else:
        name += "_antisymmetry_clean.csv"
    return input_path.with_name(name)


def _safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _make_temp_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(temp_name)


def _commit_temp_file(
    temp_path: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically publish a sibling temp file without a no-clobber race."""

    if overwrite:
        os.replace(temp_path, destination)
        return
    try:
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"destination appeared during processing and was not replaced: "
            f"{destination}"
        ) from exc
    temp_path.unlink()


def _example_dict(
    row_number: int,
    analysis: ExpressionAnalysis,
    *,
    max_expression_chars: int = 500,
) -> dict[str, object]:
    summands: list[dict[str, object]] = []
    for zero_summand in analysis.zero_summands:
        term = zero_summand.term
        if len(term) > max_expression_chars:
            term = term[:max_expression_chars] + "…"
        summands.append(
            {
                "term_index": zero_summand.term_index,
                "term": term,
                "reasons": [asdict(reason) for reason in zero_summand.reasons],
            }
        )
    return {
        "csv_row": row_number,
        "classification": analysis.classification,
        "zero_summands": summands,
    }


def filter_csv(
    input_path: Path,
    output_path: Path | None,
    *,
    mode: str = "prune-terms",
    assumptions: OnShellAssumptions | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    max_examples: int = 20,
    progress_every: int = 50_000,
) -> tuple[FilterStats, list[dict[str, object]]]:
    """Stream a CSV through the exact-zero classifier."""

    if mode not in {"prune-terms", "drop-rows"}:
        raise ValueError("mode must be 'prune-terms' or 'drop-rows'")
    if not dry_run and output_path is None:
        raise ValueError("an output path is required unless --dry-run is used")
    if output_path is not None:
        if _safe_resolve(input_path) == _safe_resolve(output_path):
            raise ValueError("input and output paths must differ; the source is preserved")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {output_path} (pass --overwrite to replace it)"
            )

    stats = FilterStats()
    examples: list[dict[str, object]] = []
    temp_output: Path | None = None
    _set_csv_field_size_limit()

    try:
        with ExitStack() as stack:
            input_handle = stack.enter_context(_open_csv_text(input_path, "r"))
            reader = csv.DictReader(input_handle)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header")
            if "simple" not in reader.fieldnames:
                raise ValueError(
                    f"CSV must contain a 'simple' column; got {reader.fieldnames}"
                )
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"CSV contains duplicate headers: {reader.fieldnames}")

            writer: csv.DictWriter[str] | None = None
            if not dry_run:
                assert output_path is not None
                temp_output = _make_temp_sibling(output_path)
                output_handle = stack.enter_context(
                    _open_csv_text(
                        temp_output,
                        "w",
                        compressed=output_path.suffix.lower() == ".gz",
                    )
                )
                writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
                writer.writeheader()

            for row in reader:
                stats.input_rows += 1
                if None in row:
                    raise ValueError(
                        f"CSV row {reader.line_num} has more fields than the header"
                    )
                if any(value is None for value in row.values()):
                    raise ValueError(
                        f"CSV row {reader.line_num} has fewer fields than the header"
                    )

                simple = row["simple"]
                assert simple is not None
                try:
                    analysis = analyze_simple_expression(
                        simple,
                        assumptions=assumptions,
                    )
                except (ExpressionSyntaxError, ValueError) as exc:
                    raise ValueError(
                        f"could not safely parse 'simple' at CSV row "
                        f"{reader.line_num}: {exc}"
                    ) from exc

                stats.terms_examined += analysis.term_count
                stats.zero_summands += len(analysis.zero_summands)
                for zero_summand in analysis.zero_summands:
                    stats.reason_counts.update(
                        reason.code for reason in zero_summand.reasons
                    )

                classification = analysis.classification
                if classification == "clean":
                    stats.clean_rows += 1
                    keep_row = True
                elif classification == "all_zero":
                    stats.all_zero_rows += 1
                    keep_row = False
                else:
                    stats.mixed_rows += 1
                    keep_row = mode == "prune-terms"
                    if keep_row:
                        assert analysis.cleaned is not None
                        row["simple"] = analysis.cleaned
                        stats.rows_modified += 1

                if classification != "clean" and len(examples) < max_examples:
                    examples.append(_example_dict(reader.line_num, analysis))

                if keep_row:
                    stats.output_rows += 1
                    if writer is not None:
                        writer.writerow(row)
                else:
                    stats.rows_dropped += 1

                if progress_every and stats.input_rows % progress_every == 0:
                    print(
                        f"processed {stats.input_rows:,} rows "
                        f"({stats.zero_summands:,} zero summands)",
                        file=sys.stderr,
                    )

        if temp_output is not None:
            assert output_path is not None
            _commit_temp_file(
                temp_output,
                output_path,
                overwrite=overwrite,
            )
            os.chmod(
                output_path,
                stat.S_IMODE(input_path.stat().st_mode) & 0o666,
            )
            temp_output = None
    finally:
        if temp_output is not None:
            try:
                temp_output.unlink()
            except FileNotFoundError:
                pass

    return stats, examples


def _write_json_report(path: Path, report: dict[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"report already exists: {path} (pass --overwrite to replace it)"
        )
    temp_path = _make_temp_sibling(path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _commit_temp_file(temp_path, path, overwrite=overwrite)
        os.chmod(path, 0o644)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _parse_label_set(value: str) -> frozenset[int]:
    """Parse a comma-separated set of positive particle labels."""

    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError(
            "labels must be a comma-separated list such as 2,3"
        )
    try:
        labels = frozenset(int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "labels must be comma-separated positive integers"
        ) from exc
    if any(label <= 0 for label in labels):
        raise argparse.ArgumentTypeError("labels must be positive integers")
    return labels


def add_onshell_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared, label-scoped on-shell options to a CLI parser."""

    parser.add_argument(
        "--all-massless-ym",
        type=int,
        metavar="N",
        help=(
            "declare labels 1..N massless with transverse field strengths "
            "(appropriate for an N-gluon Yang-Mills dataset)"
        ),
    )
    parser.add_argument(
        "--massless-labels",
        type=_parse_label_set,
        metavar="I,J,...",
        help="comma-separated labels whose momenta are explicitly massless",
    )
    parser.add_argument(
        "--transverse-field-labels",
        type=_parse_label_set,
        metavar="I,J,...",
        help=(
            "comma-separated field-strength labels explicitly known to be "
            "transverse; F_i rules also require i in --massless-labels"
        ),
    )
    parser.add_argument(
        "--include-ym-onshell",
        action="store_true",
        help=(
            "deprecated all-massless shorthand; requires --n-particles N "
            "so the affected labels are explicit"
        ),
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        metavar="N",
        help="particle count required by deprecated --include-ym-onshell",
    )


def onshell_assumptions_from_namespace(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> OnShellAssumptions | None:
    """Build explicit assumptions and reject ambiguous CLI combinations."""

    legacy = bool(args.include_ym_onshell)
    all_massless_count: int | None = args.all_massless_ym
    massless_labels: frozenset[int] | None = args.massless_labels
    transverse_labels: frozenset[int] | None = args.transverse_field_labels
    explicit_labels = (
        massless_labels is not None or transverse_labels is not None
    )

    if legacy:
        if args.n_particles is None:
            parser.error(
                "--include-ym-onshell requires --n-particles N; prefer "
                "--all-massless-ym N"
            )
        if all_massless_count is not None or explicit_labels:
            parser.error(
                "--include-ym-onshell cannot be combined with "
                "--all-massless-ym or explicit label sets"
            )
        try:
            assumptions = OnShellAssumptions.all_massless_ym(args.n_particles)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            "warning: --include-ym-onshell is deprecated; use "
            "--all-massless-ym N",
            file=sys.stderr,
        )
        return assumptions

    if args.n_particles is not None:
        parser.error("--n-particles is only valid with --include-ym-onshell")

    if all_massless_count is not None:
        if explicit_labels:
            parser.error(
                "--all-massless-ym cannot be combined with explicit label sets"
            )
        try:
            return OnShellAssumptions.all_massless_ym(all_massless_count)
        except ValueError as exc:
            parser.error(str(exc))

    if explicit_labels:
        return OnShellAssumptions(
            massless_momenta=massless_labels or frozenset(),
            transverse_field_strengths=transverse_labels or frozenset(),
        )
    return None


def assumptions_report(
    assumptions: OnShellAssumptions | None,
) -> dict[str, list[int]]:
    """Serialize the exact on-shell label sets, including the empty default."""

    if assumptions is None:
        return {
            "massless_momenta": [],
            "transverse_field_strengths": [],
        }
    return assumptions.to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prune simple-expression summands that vanish by field-strength "
            "antisymmetry, streaming plain or gzip-compressed CSV files."
        )
    )
    parser.add_argument("input", type=Path, help="source .csv or .csv.gz")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help=(
            "cleaned output path (default: INPUT_antisymmetry_clean.csv[.gz]); "
            "not used with --dry-run"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("prune-terms", "drop-rows"),
        default="prune-terms",
        help=(
            "prune only zero summands and drop all-zero rows (default), or "
            "drop every row containing a zero summand"
        ),
    )
    add_onshell_cli_arguments(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and report without writing a cleaned CSV",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="optional machine-readable report path",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="maximum flagged-row examples stored in the JSON/stdout report",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="emit progress every N input rows; use 0 to disable",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing output/report (never the input)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    assumptions = onshell_assumptions_from_namespace(args, parser)

    input_path: Path = args.input.expanduser()
    if not input_path.is_file():
        raise SystemExit(f"input file does not exist: {input_path}")

    output_path: Path | None
    if args.dry_run:
        output_path = None
    else:
        output_path = (
            args.output.expanduser()
            if args.output is not None
            else _default_output_path(input_path)
        )

    try:
        report_path = (
            args.report_json.expanduser()
            if args.report_json is not None
            else None
        )
        if report_path is not None:
            report_resolved = _safe_resolve(report_path)
            if report_resolved == _safe_resolve(input_path):
                raise ValueError("the JSON report path must differ from the input")
            if (
                output_path is not None
                and report_resolved == _safe_resolve(output_path)
            ):
                raise ValueError("the JSON report path must differ from the CSV output")
            if report_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"report already exists: {report_path} "
                    "(pass --overwrite to replace it)"
                )

        stats, examples = filter_csv(
            input_path,
            output_path,
            mode=args.mode,
            assumptions=assumptions,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            max_examples=args.max_examples,
            progress_every=args.progress_every,
        )
        report: dict[str, object] = {
            "report_schema_version": 1,
            "input": str(_safe_resolve(input_path)),
            "output": str(_safe_resolve(output_path)) if output_path else None,
            "mode": args.mode,
            "on_shell_assumptions": assumptions_report(assumptions),
            "stats": stats.to_dict(),
            "examples": examples,
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
