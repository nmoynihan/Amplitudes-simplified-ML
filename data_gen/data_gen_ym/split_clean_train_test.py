#!/usr/bin/env python3
"""Extract aligned train/test splits from the corrected Yang--Mills data.

The corrected generators write a raw expression row and its tokenised row at
the same position.  This utility preserves that positional pairing while it
removes a deterministic random test sample from the four- and five-point
datasets.  The source files are never modified.

With no arguments, the command reads the completed 500,000-row corrected
datasets and writes eight *plain* CSV files under
``data/data_ym/train_test_split``::

    python -m data_gen.data_gen_ym.split_clean_train_test

For each particle count the outputs are a 499,800-row raw/tokenised training
pair and a 200-row raw/tokenised test pair.  Token alignment is checked by
re-tokenising every raw expression before any temporary output is published.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import os
import random
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

if __package__:
    from . import generate_clean_4pt as _core
else:  # Allow ``python data_gen/data_gen_ym/split_clean_train_test.py``.
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from data_gen.data_gen_ym import generate_clean_4pt as _core


CSV_HEADER = ("simple", "scrambled")
DEFAULT_TEST_SIZE = 200
DEFAULT_4PT_SEED = 4_020_004
DEFAULT_5PT_SEED = 5_020_005
DEFAULT_PROGRESS_EVERY = 50_000
DEFAULT_INPUT_DIR = _core.REPO_ROOT / "data" / "data_ym"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "train_test_split"
DEFAULT_4PT_RAW = (
    DEFAULT_INPUT_DIR / "ym_4pt_500000_canonical_nonzero.csv.gz"
)
DEFAULT_4PT_TOKENISED = (
    DEFAULT_INPUT_DIR / "ym_4pt_500000_canonical_nonzero_tok.csv.gz"
)
DEFAULT_5PT_RAW = (
    DEFAULT_INPUT_DIR / "ym_5pt_500000_canonical_nonzero.csv.gz"
)
DEFAULT_5PT_TOKENISED = (
    DEFAULT_INPUT_DIR / "ym_5pt_500000_canonical_nonzero_tok.csv.gz"
)


@dataclass(frozen=True)
class InputPair:
    """The aligned raw/tokenised inputs for one particle count."""

    n_particles: int
    raw: Path
    tokenised: Path
    seed: int
    expected_rows: int | None = None
    tokenizer_max_particles: int | None = None


@dataclass(frozen=True)
class SplitPaths:
    """The four plain-CSV destinations for one particle count."""

    train_raw: Path
    train_tokenised: Path
    test_raw: Path
    test_tokenised: Path

    def all(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.train_raw,
            self.train_tokenised,
            self.test_raw,
            self.test_tokenised,
        )


@dataclass(frozen=True)
class SplitSummary:
    """Counts and destinations for a completed in-memory split operation."""

    n_particles: int
    source_rows: int
    train_rows: int
    test_rows: int
    seed: int
    paths: SplitPaths


def _open_csv(path: Path, mode: str) -> TextIO:
    """Open a UTF-8 CSV, transparently reading gzip inputs by suffix."""

    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _temporary_sibling(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _publish(temp_path: Path, destination: Path, *, overwrite: bool) -> None:
    """Publish one completed temporary file without silent replacement."""

    if overwrite:
        os.replace(temp_path, destination)
    else:
        try:
            os.link(temp_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"destination appeared during splitting: {destination}"
            ) from exc
        temp_path.unlink()
    os.chmod(destination, 0o644)


def _identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _backup_destinations(destinations: Sequence[Path]) -> dict[Path, Path]:
    """Hard-link existing outputs so an overwrite commit can be rolled back."""

    backups: dict[Path, Path] = {}
    try:
        for destination in destinations:
            if not destination.exists():
                continue
            backup = _temporary_sibling(destination)
            backup.unlink()
            os.link(destination, backup)
            backups[destination] = backup
    except BaseException:
        for backup in backups.values():
            try:
                backup.unlink()
            except FileNotFoundError:
                pass
        raise
    return backups


def _publish_all(
    temporary_paths: Sequence[Path],
    destinations: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    """Publish every output, rolling the set back if a later publish fails."""

    backups = _backup_destinations(destinations) if overwrite else {}
    published: dict[Path, tuple[int, int]] = {}
    try:
        for temp_path, destination in zip(temporary_paths, destinations):
            temp_identity = _identity(temp_path)
            try:
                _publish(temp_path, destination, overwrite=overwrite)
            except BaseException:
                # Publication may have moved/linked the file before a later
                # chmod or unlink failed.  Track it so rollback still sees it.
                try:
                    if (
                        destination.exists()
                        and _identity(destination) == temp_identity
                    ):
                        published[destination] = temp_identity
                except OSError:
                    pass
                raise
            published[destination] = temp_identity
    except BaseException as exc:
        rollback_errors: list[str] = []
        for destination in reversed(tuple(published)):
            backup = backups.get(destination)
            if backup is not None:
                try:
                    os.replace(backup, destination)
                    del backups[destination]
                except OSError as rollback_exc:
                    rollback_errors.append(
                        f"could not restore {destination} from {backup}: "
                        f"{rollback_exc}"
                    )
                continue
            try:
                if destination.exists() and _identity(destination) == published[
                    destination
                ]:
                    destination.unlink()
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"could not remove newly published {destination}: "
                    f"{rollback_exc}"
                )

        # Backups for destinations that were never replaced are redundant.
        # Keep a backup on disk if restoration failed, so the old data remains
        # recoverable rather than deleting the only safe copy.
        for destination, backup in tuple(backups.items()):
            if destination in published:
                continue
            try:
                backup.unlink()
                del backups[destination]
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"could not clean unused backup {backup}: {rollback_exc}"
                )
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"output publication failed and rollback was incomplete: {details}"
            ) from exc
        raise
    else:
        for backup in backups.values():
            backup.unlink()


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
    """Yield positional raw/token rows and reject structural misalignment."""

    missing = object()
    with _open_csv(raw_path, "r") as raw_handle, _open_csv(
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
                    f"tokenised input has more rows than raw input; first "
                    f"extra row is data row {index + 1}"
                )
            if token_row is missing:
                raise ValueError(
                    f"raw input has more rows than tokenised input; first "
                    f"extra row is data row {index + 1}"
                )
            if len(raw_row) != len(CSV_HEADER):
                raise ValueError(
                    f"raw data row {index + 1} in {raw_path} has "
                    f"{len(raw_row)} columns; expected 2"
                )
            if len(token_row) != len(CSV_HEADER):
                raise ValueError(
                    f"tokenised data row {index + 1} in {tokenised_path} has "
                    f"{len(token_row)} columns; expected 2"
                )
            yield index, raw_row, token_row


def _report_path_for(raw_path: Path) -> Path | None:
    name = raw_path.name
    if name.endswith(".csv.gz"):
        stem = name[: -len(".csv.gz")]
    elif name.endswith(".csv"):
        stem = name[: -len(".csv")]
    else:
        return None
    return raw_path.with_name(f"{stem}.report.json")


def _matching_generator_report(
    pair: InputPair,
) -> tuple[dict[str, Any], Path] | None:
    """Load a report only when it names these exact two source files."""

    report_path = _report_path_for(pair.raw)
    if report_path is None or not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        outputs = report["outputs"]
        raw_output = outputs["raw"]
        token_output = outputs["tokenized"]
        reported_raw = Path(raw_output["path"]).expanduser().resolve(
            strict=False
        )
        reported_token = Path(token_output["path"]).expanduser().resolve(
            strict=False
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if reported_raw != pair.raw.resolve(strict=False):
        return None
    if reported_token != pair.tokenised.resolve(strict=False):
        return None
    return report, report_path


def _reported_row_count(pair: InputPair) -> int | None:
    """Use a generator report only when it names these exact two inputs."""

    match = _matching_generator_report(pair)
    if match is None:
        return None
    report, report_path = match
    raw_rows = report["outputs"]["raw"]["rows"]
    token_rows = report["outputs"]["tokenized"]["rows"]
    if (
        isinstance(raw_rows, bool)
        or not isinstance(raw_rows, int)
        or isinstance(token_rows, bool)
        or not isinstance(token_rows, int)
        or raw_rows < 0
        or raw_rows != token_rows
    ):
        raise ValueError(
            f"invalid or unequal row counts in generator report {report_path}"
        )
    return raw_rows


def _tokenizer_size(pair: InputPair) -> int:
    """Resolve the vocabulary setting independently for each input pair."""

    if pair.tokenizer_max_particles is not None:
        return pair.tokenizer_max_particles
    match = _matching_generator_report(pair)
    if match is not None:
        report, report_path = match
        value = report.get("settings", {}).get("tokenizer_max_particles")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < pair.n_particles
        ):
            raise ValueError(
                "invalid tokenizer_max_particles in generator report "
                f"{report_path}"
            )
        print(
            f"{pair.n_particles}PT: using tokenizer max_particles={value} "
            "from the generator report",
            file=sys.stderr,
        )
        return value
    fallback = 8
    print(
        f"{pair.n_particles}PT: no matching generator report; assuming "
        f"tokenizer max_particles={fallback}",
        file=sys.stderr,
    )
    return fallback


def _count_aligned_rows(pair: InputPair, *, progress_every: int) -> int:
    print(
        f"{pair.n_particles}PT: counting aligned source rows "
        "(no matching generator report found)",
        file=sys.stderr,
    )
    count = 0
    for index, _raw_row, _token_row in _aligned_rows(
        pair.raw, pair.tokenised
    ):
        count = index + 1
        if progress_every and count % progress_every == 0:
            print(
                f"{pair.n_particles}PT: counted {count:,} rows",
                file=sys.stderr,
            )
    return count


def _source_row_count(pair: InputPair, *, progress_every: int) -> int:
    if pair.expected_rows is not None:
        return pair.expected_rows
    reported = _reported_row_count(pair)
    if reported is not None:
        print(
            f"{pair.n_particles}PT: generator report declares "
            f"{reported:,} aligned rows",
            file=sys.stderr,
        )
        return reported
    return _count_aligned_rows(pair, progress_every=progress_every)


def output_paths(
    output_dir: Path,
    *,
    n_particles: int,
    source_rows: int,
    test_size: int,
) -> SplitPaths:
    """Return the conventional four output names for one split."""

    train_rows = source_rows - test_size
    train_stem = (
        f"ym_{n_particles}pt_train_{train_rows}_canonical_nonzero"
    )
    test_stem = f"ym_{n_particles}pt_test_{test_size}_canonical_nonzero"
    return SplitPaths(
        train_raw=output_dir / f"{train_stem}.csv",
        train_tokenised=output_dir / f"{train_stem}_tok.csv",
        test_raw=output_dir / f"{test_stem}.csv",
        test_tokenised=output_dir / f"{test_stem}_tok.csv",
    )


def _validate_token_row(
    raw_row: list[str],
    token_row: list[str],
    *,
    tokenizer: Any,
    n_particles: int,
    index: int,
) -> None:
    for column, raw_expression, encoded_text in zip(
        CSV_HEADER, raw_row, token_row
    ):
        try:
            encoded = json.loads(encoded_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{n_particles}PT tokenised row {index + 1}, column "
                f"{column!r}, is not valid JSON"
            ) from exc
        if not isinstance(encoded, list) or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in encoded
        ):
            raise ValueError(
                f"{n_particles}PT tokenised row {index + 1}, column "
                f"{column!r}, is not a JSON list of integers"
            )
        expected = tokenizer.encode_infix(raw_expression)
        if encoded != expected:
            raise ValueError(
                f"{n_particles}PT raw/token mismatch at data row "
                f"{index + 1}, column {column!r}"
            )


def _write_split_to_temporaries(
    pair: InputPair,
    *,
    source_rows: int,
    test_size: int,
    destinations: SplitPaths,
    temporaries: SplitPaths,
    verify_alignment: str,
    tokenizer_max_particles: int,
    progress_every: int,
) -> SplitSummary:
    selected = set(
        random.Random(pair.seed).sample(range(source_rows), test_size)
    )
    tokenizer = None
    if verify_alignment != "none":
        tokenizer = _core.ScatteringAmplitudeTokenizer(
            max_particles=tokenizer_max_particles,
            max_sequence_length=None,
        )

    train_count = 0
    test_count = 0
    processed = 0
    with ExitStack() as stack:
        handles = [
            stack.enter_context(path.open("w", newline="", encoding="utf-8"))
            for path in temporaries.all()
        ]
        writers = [csv.writer(handle) for handle in handles]
        for writer in writers:
            writer.writerow(CSV_HEADER)
        train_raw_writer, train_token_writer, test_raw_writer, test_token_writer = (
            writers
        )

        for index, raw_row, token_row in _aligned_rows(
            pair.raw, pair.tokenised
        ):
            is_test = index in selected
            if verify_alignment == "all" or (
                verify_alignment == "test" and is_test
            ):
                assert tokenizer is not None
                _validate_token_row(
                    raw_row,
                    token_row,
                    tokenizer=tokenizer,
                    n_particles=pair.n_particles,
                    index=index,
                )

            if is_test:
                test_raw_writer.writerow(raw_row)
                test_token_writer.writerow(token_row)
                test_count += 1
            else:
                train_raw_writer.writerow(raw_row)
                train_token_writer.writerow(token_row)
                train_count += 1

            processed = index + 1
            if progress_every and processed % progress_every == 0:
                verification = (
                    " and verified" if verify_alignment == "all" else ""
                )
                print(
                    f"{pair.n_particles}PT: split{verification} "
                    f"{processed:,}/{source_rows:,} rows",
                    file=sys.stderr,
                )

    if processed != source_rows:
        raise ValueError(
            f"{pair.n_particles}PT source row count changed or was reported "
            f"incorrectly: expected {source_rows:,}, read {processed:,}"
        )
    if test_count != test_size or train_count != source_rows - test_size:
        raise RuntimeError(
            f"{pair.n_particles}PT split count error: train={train_count:,}, "
            f"test={test_count:,}"
        )

    return SplitSummary(
        n_particles=pair.n_particles,
        source_rows=source_rows,
        train_rows=train_count,
        test_rows=test_count,
        seed=pair.seed,
        paths=destinations,
    )


def _validate_inputs(pairs: Sequence[InputPair], *, test_size: int) -> None:
    if test_size < 1:
        raise ValueError("test size must be at least 1")
    input_paths = [path for pair in pairs for path in (pair.raw, pair.tokenised)]
    resolved = [path.resolve(strict=False) for path in input_paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("the four input paths must be distinct")
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "input CSV does not exist: " + ", ".join(str(path) for path in missing)
        )
    for pair in pairs:
        if pair.expected_rows is not None and pair.expected_rows < 0:
            raise ValueError(
                f"{pair.n_particles}PT expected row count must be non-negative"
            )
        if (
            pair.tokenizer_max_particles is not None
            and pair.tokenizer_max_particles < pair.n_particles
        ):
            raise ValueError(
                f"{pair.n_particles}PT tokenizer max_particles must be at "
                f"least {pair.n_particles}"
            )


def _validate_destinations(
    pairs: Sequence[InputPair],
    destinations: Sequence[SplitPaths],
    *,
    overwrite: bool,
) -> None:
    input_paths = {
        path.resolve(strict=False)
        for pair in pairs
        for path in (pair.raw, pair.tokenised)
    }
    output_paths_flat = [path for group in destinations for path in group.all()]
    resolved_outputs = [path.resolve(strict=False) for path in output_paths_flat]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise ValueError("all eight output paths must be distinct")
    collisions = [
        path
        for path in output_paths_flat
        if path.resolve(strict=False) in input_paths
    ]
    if collisions:
        raise ValueError(
            "an output path would replace an input: "
            + ", ".join(str(path) for path in collisions)
        )
    if not overwrite:
        existing = [path for path in output_paths_flat if path.exists()]
        if existing:
            raise FileExistsError(
                "split output already exists (use --overwrite to replace): "
                + ", ".join(str(path) for path in existing)
            )


def split_datasets(args: argparse.Namespace) -> tuple[SplitSummary, ...]:
    """Split both particle-count datasets and publish exactly eight CSVs."""

    pairs = (
        InputPair(
            n_particles=4,
            raw=args.raw_4pt.expanduser(),
            tokenised=args.tokenised_4pt.expanduser(),
            seed=args.seed_4pt,
            expected_rows=args.expected_rows_4pt,
            tokenizer_max_particles=args.tokenizer_max_particles_4pt,
        ),
        InputPair(
            n_particles=5,
            raw=args.raw_5pt.expanduser(),
            tokenised=args.tokenised_5pt.expanduser(),
            seed=args.seed_5pt,
            expected_rows=args.expected_rows_5pt,
            tokenizer_max_particles=args.tokenizer_max_particles_5pt,
        ),
    )
    _validate_inputs(pairs, test_size=args.test_size)

    source_rows = tuple(
        _source_row_count(pair, progress_every=args.progress_every)
        for pair in pairs
    )
    for pair, count in zip(pairs, source_rows):
        if count < args.test_size:
            raise ValueError(
                f"{pair.n_particles}PT input has {count:,} rows, fewer than "
                f"the requested {args.test_size:,} test rows"
            )

    output_dir = args.output_dir.expanduser()
    destinations = tuple(
        output_paths(
            output_dir,
            n_particles=pair.n_particles,
            source_rows=count,
            test_size=args.test_size,
        )
        for pair, count in zip(pairs, source_rows)
    )
    _validate_destinations(pairs, destinations, overwrite=args.overwrite)

    temporary_paths: list[Path] = []
    summaries: list[SplitSummary] = []
    try:
        for pair, count, paths in zip(pairs, source_rows, destinations):
            temp_group = SplitPaths(
                *(_temporary_sibling(path) for path in paths.all())
            )
            temporary_paths.extend(temp_group.all())
            summaries.append(
                _write_split_to_temporaries(
                    pair,
                    source_rows=count,
                    test_size=args.test_size,
                    destinations=paths,
                    temporaries=temp_group,
                    verify_alignment=args.verify_alignment,
                    tokenizer_max_particles=_tokenizer_size(pair),
                    progress_every=args.progress_every,
                )
            )

        final_paths = [
            path for group in destinations for path in group.all()
        ]
        _publish_all(
            temporary_paths,
            final_paths,
            overwrite=args.overwrite,
        )
        temporary_paths.clear()
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    return tuple(summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove an aligned random test subset from corrected 4PT and 5PT "
            "Yang--Mills raw/tokenised datasets and write eight plain CSVs."
        )
    )
    parser.add_argument("--raw-4pt", type=Path, default=DEFAULT_4PT_RAW)
    parser.add_argument(
        "--tokenised-4pt", type=Path, default=DEFAULT_4PT_TOKENISED
    )
    parser.add_argument("--raw-5pt", type=Path, default=DEFAULT_5PT_RAW)
    parser.add_argument(
        "--tokenised-5pt", type=Path, default=DEFAULT_5PT_TOKENISED
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed-4pt", type=int, default=DEFAULT_4PT_SEED)
    parser.add_argument("--seed-5pt", type=int, default=DEFAULT_5PT_SEED)
    parser.add_argument(
        "--expected-rows-4pt",
        type=int,
        default=None,
        help="optional trusted count; otherwise use a matching report or count",
    )
    parser.add_argument(
        "--expected-rows-5pt",
        type=int,
        default=None,
        help="optional trusted count; otherwise use a matching report or count",
    )
    parser.add_argument(
        "--verify-alignment",
        choices=("all", "test", "none"),
        default="all",
        help=(
            "re-tokenise all rows (default), only selected test rows, or no "
            "rows; positional row/count checks always run"
        ),
    )
    parser.add_argument(
        "--tokenizer-max-particles-4pt",
        type=int,
        default=None,
        help="override the 4PT vocabulary size; otherwise infer it from report",
    )
    parser.add_argument(
        "--tokenizer-max-particles-5pt",
        type=int,
        default=None,
        help="override the 5PT vocabulary size; otherwise infer it from report",
    )
    parser.add_argument(
        "--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.test_size < 1:
        parser.error("--test-size must be at least 1")
    if (
        args.tokenizer_max_particles_4pt is not None
        and args.tokenizer_max_particles_4pt < 4
    ):
        parser.error("--tokenizer-max-particles-4pt must be at least 4")
    if (
        args.tokenizer_max_particles_5pt is not None
        and args.tokenizer_max_particles_5pt < 5
    ):
        parser.error("--tokenizer-max-particles-5pt must be at least 5")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    for option in ("expected_rows_4pt", "expected_rows_5pt"):
        value = getattr(args, option)
        if value is not None and value < args.test_size:
            parser.error(
                f"--{option.replace('_', '-')} must be at least --test-size"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    try:
        summaries = split_datasets(args)
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Created eight aligned plain CSV files:")
    for summary in summaries:
        print(
            f"{summary.n_particles}PT: {summary.train_rows:,} training rows, "
            f"{summary.test_rows:,} test rows (seed {summary.seed})"
        )
        for path in summary.paths.all():
            print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
