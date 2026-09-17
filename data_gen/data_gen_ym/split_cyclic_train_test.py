#!/usr/bin/env python3
"""Publish target- and input-disjoint splits of cyclic Yang--Mills pairs.

As in the repo's SQED release splitters, hold out singleton tokenised targets
and verify the target intersection before publishing. Also require singleton
scrambled inputs, so neither side of a test pair appears in training. Repeated
targets/inputs stay in training; no source rows or expression text are changed.
This checks token equality, not general algebraic equivalence. The Yang--Mills
splitter supplies alignment checks and transactional publication utilities;
its row-index sampling and the clean generators' canonicalisers are not used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__:
    from . import split_clean_train_test as _shared
    from .notation import CYCLIC_ORDER_REFERENCE_FILES, default_cyclic_order
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_gen.data_gen_ym import split_clean_train_test as _shared
    from data_gen.data_gen_ym.notation import CYCLIC_ORDER_REFERENCE_FILES, default_cyclic_order


DEFAULT_SPLIT_SEED = 20260401


@dataclass(frozen=True)
class _Scan:
    source_rows: int
    unique_targets: int
    unique_inputs: int
    singleton_targets: int
    eligible_targets: int
    selected: frozenset[bytes]
    source_digest: str


@dataclass(frozen=True)
class SplitPaths:
    train_raw: Path
    train_tokenised: Path
    test_raw: Path
    test_tokenised: Path
    manifest: Path

    def all(self) -> tuple[Path, ...]:
        return (
            self.train_raw, self.train_tokenised, self.test_raw,
            self.test_tokenised, self.manifest,
        )


def output_paths(
    output_dir: Path, *, n_particles: int, source_rows: int,
    test_size: int, seed: int,
) -> SplitPaths:
    stem = f"ym_cyclic_{n_particles}pt"
    train = f"{stem}_train_{source_rows - test_size}"
    test = f"{stem}_test_{test_size}"
    return SplitPaths(
        output_dir / f"{train}.csv",
        output_dir / f"{train}_tok.csv",
        output_dir / f"{test}.csv",
        output_dir / f"{test}_tok.csv",
        output_dir / f"{stem}_split_seed{seed}.json",
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def _validated_rows(pair: _shared.InputPair, tokenizer):
    for index, raw_row, token_row in _shared._aligned_rows(pair.raw, pair.tokenised):
        _shared._validate_token_row(
            raw_row, token_row, tokenizer=tokenizer,
            n_particles=pair.n_particles, index=index,
        )
        # Parse and serialize to make whitespace in token JSON irrelevant.
        target, scrambled = (
            hashlib.sha256(_json_bytes(json.loads(column))).digest()
            for column in token_row
        )
        yield index, raw_row, token_row, target, scrambled


def _scan_sources(pair: _shared.InputPair, tokenizer, test_size: int) -> _Scan:
    target_counts: Counter[bytes] = Counter()
    input_counts: Counter[bytes] = Counter()
    singleton_inputs: dict[bytes, bytes] = {}
    source_digest = hashlib.sha256()
    source_rows = 0
    for index, raw_row, token_row, target, scrambled in _validated_rows(pair, tokenizer):
        source_rows = index + 1
        source_digest.update(_json_bytes((raw_row, token_row)) + b"\n")
        target_counts[target] += 1
        input_counts[scrambled] += 1
        if target_counts[target] == 1:
            singleton_inputs[target] = scrambled
        else:
            singleton_inputs.pop(target, None)

    if test_size >= source_rows:
        raise ValueError(
            f"test size {test_size} must be smaller than source row count "
            f"{source_rows}; at least one training row is required"
        )
    eligible = sorted(
        target for target, scrambled in singleton_inputs.items()
        if input_counts[scrambled] == 1
    )
    if len(eligible) < test_size:
        raise ValueError(
            f"only {len(eligible)} eligible singleton targets with unique scrambled "
            f"inputs; need {test_size} test rows. Generate a larger pool or reduce "
            "--test-size"
        )
    selected = frozenset(random.Random(pair.seed).sample(eligible, test_size))
    return _Scan(
        source_rows, len(target_counts), len(input_counts),
        len(singleton_inputs), len(eligible), selected, source_digest.hexdigest(),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_cyclic_dataset(
    raw_path: Path,
    tokenised_path: Path,
    *,
    n_particles: int,
    test_size: int,
    output_dir: Path,
    seed: int = DEFAULT_SPLIT_SEED,
    tokenizer_max_particles: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split an existing aligned pool, preserving every row and F-block word.

    The test set has exactly ``test_size`` rows. If too few singleton targets
    with unique inputs exist, fail rather than weaken the guarantee. Every raw
    and tokenised row is checked and the four CSVs plus manifest are published
    together using the existing rollback-capable Yang--Mills split utilities.
    Input files are never overwritten. Plain and gzip source CSVs are supported.
    """
    if n_particles < 4:
        raise ValueError("cyclic Yang--Mills splitting requires N >= 4")
    pair = _shared.InputPair(
        n_particles=n_particles,
        raw=Path(raw_path).expanduser().resolve(),
        tokenised=Path(tokenised_path).expanduser().resolve(),
        seed=seed,
        tokenizer_max_particles=tokenizer_max_particles,
    )
    _shared._validate_inputs((pair,), test_size=test_size)
    tokenizer = _shared._core.ScatteringAmplitudeTokenizer(
        max_particles=tokenizer_max_particles, max_sequence_length=None,
    )
    scan = _scan_sources(pair, tokenizer, test_size)
    paths = output_paths(
        Path(output_dir).expanduser().resolve(), n_particles=n_particles,
        source_rows=scan.source_rows, test_size=test_size, seed=seed,
    )
    _shared._validate_destinations((pair,), (paths,), overwrite=overwrite)

    temporary_paths: list[Path] = []
    try:
        for destination in paths.all():
            temporary_paths.append(_shared._temporary_sibling(destination))
        train_targets: set[bytes] = set()
        test_targets: set[bytes] = set()
        train_inputs: set[bytes] = set()
        test_inputs: set[bytes] = set()
        counts = [0, 0]
        source_digest = hashlib.sha256()
        with ExitStack() as stack:
            writers = [
                csv.writer(stack.enter_context(path.open("w", newline="", encoding="utf-8")))
                for path in temporary_paths[:4]
            ]
            for writer in writers:
                writer.writerow(_shared.CSV_HEADER)
            for _, raw_row, token_row, target, scrambled in _validated_rows(pair, tokenizer):
                source_digest.update(_json_bytes((raw_row, token_row)) + b"\n")
                is_test = target in scan.selected
                partition = int(is_test)
                counts[partition] += 1
                writers[2 * partition].writerow(raw_row)
                writers[2 * partition + 1].writerow(token_row)
                (test_targets if is_test else train_targets).add(target)
                (test_inputs if is_test else train_inputs).add(scrambled)

        if source_digest.hexdigest() != scan.source_digest:
            raise ValueError("source rows changed between scanning and splitting; no split published")
        if counts != [scan.source_rows - test_size, test_size]:
            raise RuntimeError(f"split row counts are wrong: {counts}")
        target_overlap = len(train_targets & test_targets)
        input_overlap = len(train_inputs & test_inputs)
        if target_overlap or input_overlap:
            raise RuntimeError(
                f"train/test leakage detected: {target_overlap} targets, {input_overlap} inputs"
            )

        output_names = ("train_raw", "train_tokenized", "test_raw", "test_tokenized")
        outputs = {
            name: {
                "path": str(destination), "rows": counts[index // 2],
                "sha256": _file_sha256(temporary),
            }
            for index, (name, destination, temporary) in enumerate(
                zip(output_names, paths.all()[:4], temporary_paths[:4])
            )
        }
        report = {
            "schema_version": 1,
            "dataset": {
                "theory": "yang_mills", "cyclic_order": True,
                "particle_order": list(default_cyclic_order(n_particles)),
                "ordering_reference": CYCLIC_ORDER_REFERENCE_FILES.get(n_particles),
                "n_particles": n_particles, "source_rows": scan.source_rows,
                "train_rows": counts[0], "test_rows": counts[1],
            },
            "sources": {
                "raw": str(pair.raw), "tokenized": str(pair.tokenised),
                "aligned_rows_sha256": scan.source_digest,
            },
            "split": {
                "seed": seed, "method": "singleton_target_holdout",
                "leakage_key": "sha256(tokenized_simple)",
                "input_leakage_key": "sha256(tokenized_scrambled)",
                "tokenizer_max_particles": tokenizer_max_particles,
                "singleton_targets": scan.singleton_targets,
                "eligible_singleton_targets": scan.eligible_targets,
            },
            "verification": {
                "all_raw_token_rows_retokenized": True,
                "all_source_rows_preserved": True,
                "train_test_target_overlap": target_overlap,
                "train_test_input_overlap": input_overlap,
                "unique_targets": scan.unique_targets,
                "unique_inputs": scan.unique_inputs,
                "test_unique_targets": len(test_targets),
                "test_unique_inputs": len(test_inputs),
            },
            "outputs": outputs,
            "manifest_path": str(paths.manifest),
        }
        temporary_paths[-1].write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        _shared._publish_all(temporary_paths, paths.all(), overwrite=overwrite)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tokenised", "--tokenized", type=Path, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-max-particles", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = split_cyclic_dataset(
            args.raw, args.tokenised, n_particles=args.N, test_size=args.test_size,
            seed=args.seed, output_dir=args.output_dir,
            tokenizer_max_particles=args.tokenizer_max_particles,
            overwrite=args.overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
