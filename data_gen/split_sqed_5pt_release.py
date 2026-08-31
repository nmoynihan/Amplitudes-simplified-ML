#!/usr/bin/env python3
"""Build a verified 499,800/200 release from a 5PT SQED candidate pool.

The SQED generator writes raw and tokenised rows in the same order.  This
module keeps that alignment while it:

* removes exact pair duplicates and ambiguous repeated inputs globally;
* takes exact broad/SQED-cover/hard quotas for a 500,000-row release;
* selects the held-out rows only from compact targets that occur once;
* verifies every accepted raw/tokenised row before publishing; and
* writes deterministic gzip CSVs plus a JSON manifest.

The default source is the 550,000-row candidate pool produced by
``generate_sqed_5pt_500k.sh``.  Its 10% headroom allows duplicate rows to be
discarded without making the requested release short.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import os
import platform
import random
import re
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence, TextIO

if __package__:
    from .filter_antisymmetry_zeros import (
        OnShellAssumptions,
        analyze_simple_expression,
    )
    from .Tokenizer import ScatteringAmplitudeTokenizer
else:  # Allow direct script execution from the repository root.
    data_gen_dir = Path(__file__).resolve().parent
    if str(data_gen_dir) not in sys.path:
        sys.path.insert(0, str(data_gen_dir))
    from filter_antisymmetry_zeros import (
        OnShellAssumptions,
        analyze_simple_expression,
    )
    from Tokenizer import ScatteringAmplitudeTokenizer


CSV_HEADER = ("simple", "scrambled")
STRATUM_NAMES = ("broad", "sqed_cover", "hard")
DEFAULT_GENERATION_SEED = 42
DEFAULT_SPLIT_SEED = 5_020_005
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOKENIZER_MAX_PARTICLES = 8
DEFAULT_PROGRESS_EVERY = 50_000
SQED_5PT_ASSUMPTIONS = OnShellAssumptions(
    massless_momenta=frozenset({2, 3, 4}),
    transverse_field_strengths=frozenset({2, 3, 4}),
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_DIR = REPO_ROOT / "data" / "sqed" / "sqed_5pt_500k_staging"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "sqed" / "sqed_5pt_500k"
DEFAULT_RAW = DEFAULT_STAGING_DIR / "sqed_5pt_oneshot_candidates_550000.csv"
DEFAULT_TOKENISED = (
    DEFAULT_STAGING_DIR / "sqed_5pt_oneshot_candidates_550000_tok.csv"
)
DEFAULT_GENERATION_LOG = (
    DEFAULT_STAGING_DIR / "sqed_5pt_oneshot_candidates_550000.log"
)

DEFAULT_PROFILE = {
    "theory": "scalar_qed",
    "n_particles": 5,
    "dataset_kind": "oneshot",
    "mass": 2.0,
    "candidate_rows": 550_000,
    "release_rows": 500_000,
    "min_scrambles": 1,
    "max_scrambles": 4,
    "min_terms": 1,
    "max_terms": 3,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "tokenizer_max_particles": DEFAULT_TOKENIZER_MAX_PARTICLES,
    "jobs": 8,
    "batch_size": 2000,
    "generator_oversample_factor": 1.2,
    "python_hash_seed": 0,
    "stratum_seed_offsets": {
        "broad": 0,
        "sqed_cover": 1_000_000_007,
        "hard": 2_000_000_033,
    },
    "validation": True,
    "reject_manifest_zero_target_summands": True,
    "onshell_assumptions": SQED_5PT_ASSUMPTIONS.to_dict(),
    "validation_polarisation_modes": ["coulomb", "covariant"],
    "full_expand_scrambled": True,
    "profile": {
        "broad": {
            "candidate_rows": 330_000,
            "release_rows": 300_000,
            "test_rows": 120,
            "unit_probability": 0.9,
            "old_style_probability": 0.0,
            "spurious_repeat_probability": 0.35,
            "scalar_power_probability": 0.15,
            "scrambles": "all",
        },
        "sqed_cover": {
            "candidate_rows": 165_000,
            "release_rows": 150_000,
            "test_rows": 60,
            "min_terms": 3,
            "max_terms": 3,
            "min_scrambles": 1,
            "max_scrambles": 4,
            "unit_probability": 1.0,
            "old_style_probability": 0.0,
            "spurious_repeat_probability": 0.0,
            "scalar_power_probability": 0.0,
            "scrambles": [
                "multiply_one",
                "ward",
                "momentum",
                "commute_dot",
                "ratio",
                "partial_fraction",
                "term_reorder",
            ],
        },
        "hard": {
            "candidate_rows": 55_000,
            "release_rows": 50_000,
            "test_rows": 20,
            "min_terms": 3,
            "max_terms": 3,
            "min_scrambles": 3,
            "max_scrambles": 4,
            "unit_probability": 0.9,
            "old_style_probability": 0.0,
            "spurious_repeat_probability": 0.35,
            "scalar_power_probability": 0.15,
            "scrambles": "all",
        },
    },
}


@dataclass(frozen=True)
class StratumSpec:
    name: str
    candidate_rows: int
    release_rows: int
    test_rows: int

    @property
    def train_rows(self) -> int:
        return self.release_rows - self.test_rows


@dataclass(frozen=True)
class OutputPaths:
    train_raw: Path
    train_tokenised: Path
    test_raw: Path
    test_tokenised: Path
    manifest: Path

    def csv_paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.train_raw,
            self.train_tokenised,
            self.test_raw,
            self.test_tokenised,
        )

    def all(self) -> tuple[Path, Path, Path, Path, Path]:
        return (*self.csv_paths(), self.manifest)


@dataclass
class PassStats:
    source_rows: int
    accepted_by_stratum: dict[str, int]
    duplicates_by_stratum: dict[str, int]
    repeated_inputs_by_stratum: dict[str, int]
    ambiguous_inputs_by_stratum: dict[str, int]
    mixed_targets_by_stratum: dict[str, int]
    all_zero_targets_by_stratum: dict[str, int]


@dataclass(frozen=True)
class ScanSummary:
    source_rows: int
    release_rows: int
    unique_targets: int
    singleton_targets_by_stratum: dict[str, int]
    duplicates_by_stratum: dict[str, int]
    repeated_inputs_by_stratum: dict[str, int]
    ambiguous_inputs_by_stratum: dict[str, int]
    mixed_targets_by_stratum: dict[str, int]
    all_zero_targets_by_stratum: dict[str, int]
    max_simple_tokens: int
    max_scrambled_tokens: int
    selected_test_targets: frozenset[bytes]
    excluded_ambiguous_inputs: frozenset[bytes]
    full_candidate_ambiguous_input_groups: int


def _is_gzip(path: Path) -> bool:
    return path.name.endswith(".gz")


@contextmanager
def _open_text(path: Path, mode: str) -> Iterator[TextIO]:
    """Open CSV text, using deterministic gzip output when requested."""

    if "r" in mode:
        if _is_gzip(path):
            with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
                yield handle
        else:
            with path.open("r", newline="", encoding="utf-8") as handle:
                yield handle
        return

    if "w" not in mode:
        raise ValueError(f"unsupported mode: {mode}")
    if _is_gzip(path):
        with path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                with io.TextIOWrapper(
                    gzip_handle,
                    encoding="utf-8",
                    newline="",
                ) as text_handle:
                    yield text_handle
    else:
        with path.open("w", newline="", encoding="utf-8") as handle:
            yield handle


def _read_header(reader: Iterator[list[str]], path: Path) -> None:
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"CSV is empty: {path}") from exc
    if tuple(header) != CSV_HEADER:
        raise ValueError(
            f"expected header {list(CSV_HEADER)!r} in {path}, got {header!r}"
        )


def _aligned_rows(
    raw_path: Path,
    tokenised_path: Path,
) -> Iterator[tuple[int, list[str], list[str]]]:
    missing = object()
    with _open_text(raw_path, "r") as raw_handle, _open_text(
        tokenised_path, "r"
    ) as token_handle:
        raw_reader = csv.reader(raw_handle)
        token_reader = csv.reader(token_handle)
        _read_header(raw_reader, raw_path)
        _read_header(token_reader, tokenised_path)
        for index, (raw_row, token_row) in enumerate(
            itertools.zip_longest(raw_reader, token_reader, fillvalue=missing)
        ):
            if raw_row is missing:
                raise ValueError(
                    "tokenised input has more rows than raw input; first extra "
                    f"row is data row {index + 1}"
                )
            if token_row is missing:
                raise ValueError(
                    "raw input has more rows than tokenised input; first extra "
                    f"row is data row {index + 1}"
                )
            if len(raw_row) != 2 or len(token_row) != 2:
                raise ValueError(
                    f"data row {index + 1} must have exactly two columns"
                )
            yield index, raw_row, token_row


def _digest_fields(*fields: str) -> bytes:
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _decode_token_list(text: str, *, row_index: int, column: str) -> list[int]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"tokenised data row {row_index + 1}, column {column!r}, "
            "is not valid JSON"
        ) from exc
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError(
            f"tokenised data row {row_index + 1}, column {column!r}, "
            "is not a JSON list of integers"
        )
    return value


def _target_digest_from_tokens(tokens: Sequence[int]) -> bytes:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "big", signed=False))
    return digest.digest()


def _validate_5pt_lexical_scope(
    expression: str,
    *,
    tokenizer: ScatteringAmplitudeTokenizer,
    row_index: int,
    column: str,
) -> None:
    """Reject ignored characters and particle labels outside the 5PT theory."""

    position = 0
    for match in tokenizer._token_re.finditer(expression):
        if expression[position : match.start()].strip():
            raise ValueError(
                f"unsupported text at data row {row_index + 1}, column {column!r}"
            )
        position = match.end()
    if expression[position:].strip():
        raise ValueError(
            f"unsupported text at data row {row_index + 1}, column {column!r}"
        )
    labels = [
        int(label)
        for label in re.findall(r"(?:p|e|F|M)_(\d+)", expression)
    ]
    if any(label < 1 or label > 5 for label in labels):
        raise ValueError(
            f"particle label outside 1..5 at data row {row_index + 1}, "
            f"column {column!r}"
        )


def _validate_token_row(
    raw_row: list[str],
    token_row: list[str],
    *,
    tokenizer: ScatteringAmplitudeTokenizer,
    row_index: int,
    max_tokens: int,
) -> tuple[bytes, bytes, int, int]:
    encoded_columns: list[list[int]] = []
    for column, raw_expression, token_text in zip(
        CSV_HEADER, raw_row, token_row
    ):
        _validate_5pt_lexical_scope(
            raw_expression,
            tokenizer=tokenizer,
            row_index=row_index,
            column=column,
        )
        stored = _decode_token_list(
            token_text,
            row_index=row_index,
            column=column,
        )
        expected = tokenizer.encode_infix(raw_expression)
        if stored != expected:
            raise ValueError(
                f"raw/token mismatch at data row {row_index + 1}, "
                f"column {column!r}"
            )
        if len(expected) > max_tokens:
            raise ValueError(
                f"data row {row_index + 1}, column {column!r}, has "
                f"{len(expected)} tokens (limit {max_tokens})"
            )
        encoded_columns.append(expected)
    return (
        _target_digest_from_tokens(encoded_columns[0]),
        _target_digest_from_tokens(encoded_columns[1]),
        len(encoded_columns[0]),
        len(encoded_columns[1]),
    )


def _stratum_boundaries(specs: Sequence[StratumSpec]) -> list[int]:
    boundaries: list[int] = []
    total = 0
    for spec in specs:
        total += spec.candidate_rows
        boundaries.append(total)
    return boundaries


def _stratum_for_index(
    index: int,
    specs: Sequence[StratumSpec],
    boundaries: Sequence[int],
) -> StratumSpec:
    for spec, boundary in zip(specs, boundaries):
        if index < boundary:
            return spec
    raise ValueError(f"source has more rows than declared (first extra {index + 1})")


def _find_ambiguous_inputs(
    raw_path: Path,
    tokenised_path: Path,
    *,
    expected_rows: int,
    progress_every: int,
) -> frozenset[bytes]:
    """Find every tokenized input paired with more than one tokenized target."""

    first_target: dict[bytes, bytes] = {}
    ambiguous: set[bytes] = set()
    processed = 0
    for index, _raw_row, token_row in _aligned_rows(raw_path, tokenised_path):
        processed = index + 1
        target_tokens = _decode_token_list(
            token_row[0],
            row_index=index,
            column="simple",
        )
        scrambled_tokens = _decode_token_list(
            token_row[1],
            row_index=index,
            column="scrambled",
        )
        target_digest = _target_digest_from_tokens(target_tokens)
        scrambled_digest = _target_digest_from_tokens(scrambled_tokens)
        previous = first_target.setdefault(scrambled_digest, target_digest)
        if previous != target_digest:
            ambiguous.add(scrambled_digest)
        if progress_every and processed % progress_every == 0:
            print(
                f"preflighted {processed:,}/{expected_rows:,} candidate rows",
                file=sys.stderr,
            )
    if processed != expected_rows:
        raise ValueError(
            f"expected {expected_rows:,} candidate rows, read {processed:,}"
        )
    return frozenset(ambiguous)


def _iter_release_rows(
    raw_path: Path,
    tokenised_path: Path,
    specs: Sequence[StratumSpec],
    *,
    tokenizer: ScatteringAmplitudeTokenizer,
    max_tokens: int,
    verify_tokens: bool,
    progress_every: int,
    stats: PassStats,
    excluded_ambiguous_inputs: frozenset[bytes],
) -> Iterator[tuple[str, list[str], list[str], bytes, bytes, int, int]]:
    """Yield globally unique rows after dropping every ambiguous input group."""

    boundaries = _stratum_boundaries(specs)
    expected_source_rows = boundaries[-1]
    seen_pairs: set[bytes] = set()
    scrambled_to_target: dict[bytes, bytes] = {}
    accepted = {spec.name: 0 for spec in specs}
    duplicates = {spec.name: 0 for spec in specs}
    repeated_inputs = {spec.name: 0 for spec in specs}
    ambiguous_inputs = {spec.name: 0 for spec in specs}
    mixed_targets = {spec.name: 0 for spec in specs}
    all_zero_targets = {spec.name: 0 for spec in specs}
    processed = 0

    for index, raw_row, token_row in _aligned_rows(raw_path, tokenised_path):
        spec = _stratum_for_index(index, specs, boundaries)
        processed = index + 1
        if accepted[spec.name] >= spec.release_rows:
            if progress_every and processed % progress_every == 0:
                print(f"scanned {processed:,}/{expected_source_rows:,} rows", file=sys.stderr)
            continue

        target_analysis = analyze_simple_expression(
            raw_row[0],
            assumptions=SQED_5PT_ASSUMPTIONS,
        )
        if target_analysis.classification == "mixed":
            mixed_targets[spec.name] += 1
            continue
        if target_analysis.classification == "all_zero":
            all_zero_targets[spec.name] += 1
            continue

        pair_digest = _digest_fields(raw_row[0], raw_row[1])
        if pair_digest in seen_pairs:
            duplicates[spec.name] += 1
            continue

        if verify_tokens:
            (
                target_digest,
                scrambled_digest,
                simple_length,
                scrambled_length,
            ) = _validate_token_row(
                raw_row,
                token_row,
                tokenizer=tokenizer,
                row_index=index,
                max_tokens=max_tokens,
            )
        else:
            simple_tokens = _decode_token_list(
                token_row[0],
                row_index=index,
                column="simple",
            )
            target_digest = _target_digest_from_tokens(simple_tokens)
            simple_length = len(simple_tokens)
            scrambled_tokens = _decode_token_list(
                token_row[1],
                row_index=index,
                column="scrambled",
            )
            scrambled_digest = _target_digest_from_tokens(scrambled_tokens)
            scrambled_length = len(scrambled_tokens)

        if scrambled_digest in excluded_ambiguous_inputs:
            ambiguous_inputs[spec.name] += 1
            continue

        previous_target = scrambled_to_target.get(scrambled_digest)
        if previous_target is not None:
            if previous_target == target_digest:
                repeated_inputs[spec.name] += 1
            else:
                ambiguous_inputs[spec.name] += 1
            continue

        scrambled_to_target[scrambled_digest] = target_digest
        seen_pairs.add(pair_digest)

        accepted[spec.name] += 1
        yield (
            spec.name,
            raw_row,
            token_row,
            pair_digest,
            target_digest,
            simple_length,
            scrambled_length,
        )

        if progress_every and processed % progress_every == 0:
            print(
                f"scanned and verified {processed:,}/{expected_source_rows:,} rows",
                file=sys.stderr,
            )

    if processed != expected_source_rows:
        raise ValueError(
            f"expected {expected_source_rows:,} candidate rows, read {processed:,}"
        )
    short = {
        spec.name: (accepted[spec.name], spec.release_rows)
        for spec in specs
        if accepted[spec.name] != spec.release_rows
    }
    if short:
        details = ", ".join(
            f"{name}={actual:,}/{expected:,}"
            for name, (actual, expected) in short.items()
        )
        raise ValueError(
            "candidate headroom was insufficient after global dedupe and "
            "ambiguous-input removal: " + details
        )
    stats.source_rows = processed
    stats.accepted_by_stratum = accepted
    stats.duplicates_by_stratum = duplicates
    stats.repeated_inputs_by_stratum = repeated_inputs
    stats.ambiguous_inputs_by_stratum = ambiguous_inputs
    stats.mixed_targets_by_stratum = mixed_targets
    stats.all_zero_targets_by_stratum = all_zero_targets


def _scan_release(
    raw_path: Path,
    tokenised_path: Path,
    specs: Sequence[StratumSpec],
    *,
    tokenizer: ScatteringAmplitudeTokenizer,
    max_tokens: int,
    split_seed: int,
    progress_every: int,
) -> ScanSummary:
    expected_source_rows = sum(spec.candidate_rows for spec in specs)
    excluded_ambiguous_inputs = _find_ambiguous_inputs(
        raw_path,
        tokenised_path,
        expected_rows=expected_source_rows,
        progress_every=progress_every,
    )
    stats = PassStats(0, {}, {}, {}, {}, {}, {})
    target_counts: dict[bytes, int] = {}
    target_strata: dict[bytes, str] = {}
    max_simple_tokens = 0
    max_scrambled_tokens = 0
    release_rows = 0

    for (
        stratum,
        raw_row,
        _token_row,
        _pair_digest,
        target_digest,
        simple_length,
        scrambled_length,
    ) in _iter_release_rows(
        raw_path,
        tokenised_path,
        specs,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        verify_tokens=True,
        progress_every=progress_every,
        stats=stats,
        excluded_ambiguous_inputs=excluded_ambiguous_inputs,
    ):
        release_rows += 1
        target_counts[target_digest] = target_counts.get(target_digest, 0) + 1
        target_strata.setdefault(target_digest, stratum)

        max_simple_tokens = max(max_simple_tokens, simple_length)
        max_scrambled_tokens = max(max_scrambled_tokens, scrambled_length)

    singleton_by_stratum = {
        spec.name: sorted(
            target
            for target, count in target_counts.items()
            if count == 1 and target_strata[target] == spec.name
        )
        for spec in specs
    }
    selected: set[bytes] = set()
    for offset, spec in enumerate(specs):
        eligible = singleton_by_stratum[spec.name]
        if len(eligible) < spec.test_rows:
            raise ValueError(
                f"{spec.name} has only {len(eligible):,} singleton targets; "
                f"need {spec.test_rows:,} for the held-out split"
            )
        rng = random.Random(split_seed + 1_000_003 * offset)
        selected.update(rng.sample(eligible, spec.test_rows))

    return ScanSummary(
        source_rows=stats.source_rows,
        release_rows=release_rows,
        unique_targets=len(target_counts),
        singleton_targets_by_stratum={
            name: len(targets) for name, targets in singleton_by_stratum.items()
        },
        duplicates_by_stratum=dict(stats.duplicates_by_stratum),
        repeated_inputs_by_stratum=dict(stats.repeated_inputs_by_stratum),
        ambiguous_inputs_by_stratum=dict(stats.ambiguous_inputs_by_stratum),
        mixed_targets_by_stratum=dict(stats.mixed_targets_by_stratum),
        all_zero_targets_by_stratum=dict(stats.all_zero_targets_by_stratum),
        max_simple_tokens=max_simple_tokens,
        max_scrambled_tokens=max_scrambled_tokens,
        selected_test_targets=frozenset(selected),
        excluded_ambiguous_inputs=excluded_ambiguous_inputs,
        full_candidate_ambiguous_input_groups=len(excluded_ambiguous_inputs),
    )


def output_paths(
    output_dir: Path,
    *,
    release_rows: int,
    test_rows: int,
) -> OutputPaths:
    train_rows = release_rows - test_rows
    return OutputPaths(
        train_raw=output_dir / f"sqed_5pt_oneshot_train_{train_rows}.csv.gz",
        train_tokenised=(
            output_dir / f"sqed_5pt_oneshot_train_{train_rows}_tok.csv.gz"
        ),
        test_raw=output_dir / f"sqed_5pt_oneshot_test_{test_rows}.csv.gz",
        test_tokenised=(
            output_dir / f"sqed_5pt_oneshot_test_{test_rows}_tok.csv.gz"
        ),
        manifest=(
            output_dir / f"sqed_5pt_oneshot_{release_rows}_manifest.json"
        ),
    )


def _temporary_sibling(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.name.endswith(".csv.gz"):
        suffix = ".csv.gz"
    else:
        suffix = destination.suffix or ".tmp"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _publish_all(temporaries: Sequence[Path], destinations: Sequence[Path]) -> None:
    existing = [path for path in destinations if _lexists(path)]
    if existing:
        raise FileExistsError(
            "release output already exists: " + ", ".join(str(path) for path in existing)
        )

    published: list[Path] = []
    try:
        for temporary, destination in zip(temporaries, destinations):
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"release destination appeared during publication: {destination}"
                ) from exc
            published.append(destination)
            temporary.unlink()
            os.chmod(destination, 0o644)
    except BaseException:
        for destination in reversed(published):
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_uncompressed(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if _is_gzip(path) else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_metadata(
    temporary: Path,
    destination: Path,
    *,
    expected_rows: int,
) -> dict[str, object]:
    rows = 0
    with _open_text(temporary, "r") as handle:
        reader = csv.reader(handle)
        _read_header(reader, temporary)
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"malformed output row in {temporary}")
            rows += 1
    if rows != expected_rows:
        raise RuntimeError(
            f"output row count mismatch for {destination}: "
            f"{rows:,}/{expected_rows:,}"
        )
    if _is_gzip(temporary):
        with temporary.open("rb") as handle:
            if handle.read(2) != b"\x1f\x8b":
                raise RuntimeError(f"output is not valid gzip: {temporary}")
    return {
        "path": str(destination.resolve(strict=False)),
        "rows": rows,
        "bytes": temporary.stat().st_size,
        "sha256": _sha256_file(temporary),
        "uncompressed_sha256": _sha256_uncompressed(temporary),
    }


def _code_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "data_gen" / "gen_data.py",
        REPO_ROOT / "data_gen" / "filter_antisymmetry_zeros.py",
        REPO_ROOT / "data_gen" / "Tokenizer.py",
        REPO_ROOT / "data_gen" / "kinematics.py",
        REPO_ROOT / "data_gen" / "numeric_utils.py",
        REPO_ROOT / "gen_data.sh",
        REPO_ROOT / "generate_sqed_5pt_500k.sh",
        REPO_ROOT / "data_gen" / "verify_sqed_5pt_release.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _write_release(
    raw_path: Path,
    tokenised_path: Path,
    specs: Sequence[StratumSpec],
    scan: ScanSummary,
    destinations: OutputPaths,
    *,
    tokenizer: ScatteringAmplitudeTokenizer,
    max_tokens: int,
    generation_seed: int,
    sqed_cover_seed_offset: int,
    hard_seed_offset: int,
    split_seed: int,
    python_hash_seed: int,
    jobs: int,
    batch_size: int,
    generation_log: Path | None,
    progress_every: int,
) -> dict[str, object]:
    temporary_paths: list[Path] = []
    train_counts = {spec.name: 0 for spec in specs}
    test_counts = {spec.name: 0 for spec in specs}
    train_targets: set[bytes] = set()
    test_targets: set[bytes] = set()
    stats = PassStats(0, {}, {}, {}, {}, {}, {})

    try:
        for destination in destinations.csv_paths():
            temporary_paths.append(_temporary_sibling(destination))
        temp_train_raw, temp_train_tok, temp_test_raw, temp_test_tok = (
            temporary_paths
        )
        with ExitStack() as stack:
            handles = [
                stack.enter_context(_open_text(path, "w"))
                for path in temporary_paths
            ]
            writers = [csv.writer(handle) for handle in handles]
            for writer in writers:
                writer.writerow(CSV_HEADER)
            train_raw_writer, train_tok_writer, test_raw_writer, test_tok_writer = (
                writers
            )

            for (
                stratum,
                raw_row,
                token_row,
                _pair_digest,
                target_digest,
                _simple_length,
                _scrambled_length,
            ) in _iter_release_rows(
                raw_path,
                tokenised_path,
                specs,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                verify_tokens=False,
                progress_every=progress_every,
                stats=stats,
                excluded_ambiguous_inputs=scan.excluded_ambiguous_inputs,
            ):
                if target_digest in scan.selected_test_targets:
                    test_raw_writer.writerow(raw_row)
                    test_tok_writer.writerow(token_row)
                    test_counts[stratum] += 1
                    test_targets.add(target_digest)
                else:
                    train_raw_writer.writerow(raw_row)
                    train_tok_writer.writerow(token_row)
                    train_counts[stratum] += 1
                    train_targets.add(target_digest)

        expected_train_counts = {spec.name: spec.train_rows for spec in specs}
        expected_test_counts = {spec.name: spec.test_rows for spec in specs}
        if train_counts != expected_train_counts:
            raise RuntimeError(
                f"training stratum counts are wrong: {train_counts!r}"
            )
        if test_counts != expected_test_counts:
            raise RuntimeError(f"test stratum counts are wrong: {test_counts!r}")
        overlap = train_targets.intersection(test_targets)
        if overlap:
            raise RuntimeError(
                f"target leakage detected across train/test ({len(overlap)} targets)"
            )

        train_rows = sum(train_counts.values())
        test_rows = sum(test_counts.values())
        output_metadata = {
            "train_raw": _csv_metadata(
                temp_train_raw,
                destinations.train_raw,
                expected_rows=train_rows,
            ),
            "train_tokenized": _csv_metadata(
                temp_train_tok,
                destinations.train_tokenised,
                expected_rows=train_rows,
            ),
            "test_raw": _csv_metadata(
                temp_test_raw,
                destinations.test_raw,
                expected_rows=test_rows,
            ),
            "test_tokenized": _csv_metadata(
                temp_test_tok,
                destinations.test_tokenised,
                expected_rows=test_rows,
            ),
        }

        profile = json.loads(json.dumps(DEFAULT_PROFILE))
        profile["candidate_rows"] = sum(spec.candidate_rows for spec in specs)
        profile["release_rows"] = sum(spec.release_rows for spec in specs)
        profile["max_tokens"] = max_tokens
        profile["tokenizer_max_particles"] = tokenizer.max_particles
        profile["jobs"] = jobs
        profile["batch_size"] = batch_size
        profile["python_hash_seed"] = python_hash_seed
        profile["stratum_seed_offsets"] = {
            "broad": 0,
            "sqed_cover": sqed_cover_seed_offset,
            "hard": hard_seed_offset,
        }
        profile["stratum_base_seeds"] = {
            "broad": generation_seed,
            "sqed_cover": generation_seed + sqed_cover_seed_offset,
            "hard": generation_seed + hard_seed_offset,
        }
        for spec in specs:
            profile["profile"][spec.name].update(
                {
                    "candidate_rows": spec.candidate_rows,
                    "release_rows": spec.release_rows,
                    "test_rows": spec.test_rows,
                }
            )

        generation_log_metadata: dict[str, object] | None = None
        if generation_log is not None and generation_log.is_file():
            generation_log_metadata = {
                "path": str(generation_log.resolve()),
                "bytes": generation_log.stat().st_size,
                "sha256": _sha256_file(generation_log),
            }

        source_artifacts = {
            "raw": {
                "path": str(raw_path.resolve()),
                "rows": scan.source_rows,
                "bytes": raw_path.stat().st_size,
                "sha256": _sha256_file(raw_path),
            },
            "tokenized": {
                "path": str(tokenised_path.resolve()),
                "rows": scan.source_rows,
                "bytes": tokenised_path.stat().st_size,
                "sha256": _sha256_file(tokenised_path),
            },
        }

        import numpy
        import sympy

        software = {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": numpy.__version__,
            "sympy": sympy.__version__,
        }

        manifest: dict[str, object] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "theory": "scalar_qed",
                "n_particles": 5,
                "kind": "oneshot",
                "total_rows": scan.release_rows,
                "train_rows": train_rows,
                "test_rows": test_rows,
            },
            "generation": {
                "seed": generation_seed,
                "settings": profile,
                "source_raw": str(raw_path.resolve()),
                "source_tokenized": str(tokenised_path.resolve()),
                "source_rows": scan.source_rows,
                "source_artifacts": source_artifacts,
                "log": generation_log_metadata,
            },
            "split": {
                "seed": split_seed,
                "method": "stratified_singleton_target_holdout",
                "leakage_key": "sha256(tokenized_simple)",
                "train_by_stratum": train_counts,
                "test_by_stratum": test_counts,
                "eligible_singleton_targets_by_stratum": (
                    scan.singleton_targets_by_stratum
                ),
            },
            "verification": {
                "all_accepted_raw_token_rows_retokenized": True,
                "global_pair_duplicates_removed_by_stratum": (
                    scan.duplicates_by_stratum
                ),
                "repeated_scrambled_inputs_removed_by_stratum": (
                    scan.repeated_inputs_by_stratum
                ),
                "ambiguous_scrambled_inputs_removed_by_stratum": (
                    scan.ambiguous_inputs_by_stratum
                ),
                "full_candidate_ambiguous_scrambled_input_groups": (
                    scan.full_candidate_ambiguous_input_groups
                ),
                "mixed_zero_summand_targets_removed_by_stratum": (
                    scan.mixed_targets_by_stratum
                ),
                "all_zero_targets_removed_by_stratum": (
                    scan.all_zero_targets_by_stratum
                ),
                "released_pair_duplicates": 0,
                "released_duplicate_scrambled_inputs": 0,
                "released_manifest_zero_target_summands": 0,
                "conflicting_scrambled_targets": 0,
                "train_test_target_overlap": 0,
                "unique_targets": scan.unique_targets,
                "test_unique_targets": len(test_targets),
                "max_simple_tokens": scan.max_simple_tokens,
                "max_scrambled_tokens": scan.max_scrambled_tokens,
                "token_limit": max_tokens,
            },
            "software": software,
            "outputs": output_metadata,
            "code_sha256": _code_hashes(),
        }

        temp_manifest = _temporary_sibling(destinations.manifest)
        temporary_paths.append(temp_manifest)
        temp_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        _publish_all(temporary_paths, destinations.all())
        temporary_paths.clear()
        return manifest
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _validate_specs(specs: Sequence[StratumSpec]) -> None:
    if tuple(spec.name for spec in specs) != STRATUM_NAMES:
        raise ValueError(f"strata must be ordered as {STRATUM_NAMES!r}")
    for spec in specs:
        if spec.candidate_rows < spec.release_rows:
            raise ValueError(
                f"{spec.name} candidate rows must be at least release rows"
            )
        if spec.test_rows < 1 or spec.test_rows > spec.release_rows:
            raise ValueError(f"invalid test row count for {spec.name}")


def build_release(args: argparse.Namespace) -> dict[str, object]:
    raw_path = args.raw.expanduser()
    tokenised_path = args.tokenised.expanduser()
    missing = [path for path in (raw_path, tokenised_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "candidate CSV does not exist: " + ", ".join(str(path) for path in missing)
        )

    specs = (
        StratumSpec(
            "broad",
            args.broad_candidate_rows,
            args.broad_release_rows,
            args.broad_test_rows,
        ),
        StratumSpec(
            "sqed_cover",
            args.sqed_cover_candidate_rows,
            args.sqed_cover_release_rows,
            args.sqed_cover_test_rows,
        ),
        StratumSpec(
            "hard",
            args.hard_candidate_rows,
            args.hard_release_rows,
            args.hard_test_rows,
        ),
    )
    _validate_specs(specs)
    release_rows = sum(spec.release_rows for spec in specs)
    test_rows = sum(spec.test_rows for spec in specs)
    destinations = output_paths(
        args.output_dir.expanduser(),
        release_rows=release_rows,
        test_rows=test_rows,
    )
    existing = [path for path in destinations.all() if _lexists(path)]
    if existing:
        raise FileExistsError(
            "release output already exists: " + ", ".join(str(path) for path in existing)
        )

    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=args.tokenizer_max_particles,
        max_sequence_length=None,
    )
    scan = _scan_release(
        raw_path,
        tokenised_path,
        specs,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        split_seed=args.split_seed,
        progress_every=args.progress_every,
    )
    return _write_release(
        raw_path,
        tokenised_path,
        specs,
        scan,
        destinations,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        generation_seed=args.generation_seed,
        sqed_cover_seed_offset=args.sqed_cover_seed_offset,
        hard_seed_offset=args.hard_seed_offset,
        split_seed=args.split_seed,
        python_hash_seed=args.python_hash_seed,
        jobs=args.jobs,
        batch_size=args.batch_size,
        generation_log=args.generation_log.expanduser()
        if args.generation_log is not None
        else None,
        progress_every=args.progress_every,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a verified target-disjoint 499,800/200 5PT SQED release "
            "from aligned raw/tokenised candidates."
        )
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--tokenised", type=Path, default=DEFAULT_TOKENISED)
    parser.add_argument(
        "--generation-log",
        type=Path,
        default=DEFAULT_GENERATION_LOG,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generation-seed", type=int, default=DEFAULT_GENERATION_SEED)
    parser.add_argument("--sqed-cover-seed-offset", type=int, default=1_000_000_007)
    parser.add_argument("--hard-seed-offset", type=int, default=2_000_000_033)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--python-hash-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--tokenizer-max-particles",
        type=int,
        default=DEFAULT_TOKENIZER_MAX_PARTICLES,
    )
    parser.add_argument("--broad-candidate-rows", type=int, default=330_000)
    parser.add_argument("--sqed-cover-candidate-rows", type=int, default=165_000)
    parser.add_argument("--hard-candidate-rows", type=int, default=55_000)
    parser.add_argument("--broad-release-rows", type=int, default=300_000)
    parser.add_argument("--sqed-cover-release-rows", type=int, default=150_000)
    parser.add_argument("--hard-release-rows", type=int, default=50_000)
    parser.add_argument("--broad-test-rows", type=int, default=120)
    parser.add_argument("--sqed-cover-test-rows", type=int, default=60)
    parser.add_argument("--hard-test-rows", type=int, default=20)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.tokenizer_max_particles < 5:
        parser.error("--tokenizer-max-particles must be at least 5")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.python_hash_seed < 0 or args.python_hash_seed > 4_294_967_295:
        parser.error("--python-hash-seed must be between 0 and 4294967295")
    if args.sqed_cover_seed_offset <= 0 or args.hard_seed_offset <= 0:
        parser.error("stratum seed offsets must be positive")
    if args.sqed_cover_seed_offset == args.hard_seed_offset:
        parser.error("stratum seed offsets must be distinct")
    for name, value in vars(args).items():
        if name.endswith("_rows") and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    try:
        manifest = build_release(args)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    print(
        "Created verified 5PT SQED release: "
        f"{dataset['train_rows']:,} train + {dataset['test_rows']:,} test"
    )
    outputs = manifest["outputs"]
    assert isinstance(outputs, dict)
    for metadata in outputs.values():
        assert isinstance(metadata, dict)
        print(f"  {metadata['path']}")
    release_rows = int(dataset["total_rows"])
    test_rows = int(dataset["test_rows"])
    manifest_path = output_paths(
        args.output_dir.expanduser(),
        release_rows=release_rows,
        test_rows=test_rows,
    ).manifest
    print(f"  {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
