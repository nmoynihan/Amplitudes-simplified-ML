#!/usr/bin/env python3
"""Independently verify a generated 5PT SQED train/test release.

This verifier deliberately replays the release policy from the aligned source
pool and re-tokenizes every source and release expression.  It does not call
the release builder's scanning or publication helpers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import itertools
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence, TextIO

if __package__:
    from .Tokenizer import ScatteringAmplitudeTokenizer
    from .gen_data import _validate_pair, manifest_mass_dimension
    from .filter_antisymmetry_zeros import (
        OnShellAssumptions,
        analyze_simple_expression,
    )
else:
    data_gen_dir = Path(__file__).resolve().parent
    if str(data_gen_dir) not in sys.path:
        sys.path.insert(0, str(data_gen_dir))
    from Tokenizer import ScatteringAmplitudeTokenizer
    from gen_data import _validate_pair, manifest_mass_dimension
    from filter_antisymmetry_zeros import (
        OnShellAssumptions,
        analyze_simple_expression,
    )


CSV_HEADER = ("simple", "scrambled")
STRATA = ("broad", "sqed_cover", "hard")
ASSUMPTIONS = OnShellAssumptions(
    massless_momenta=frozenset({2, 3, 4}),
    transverse_field_strengths=frozenset({2, 3, 4}),
)


@contextmanager
def _open_text(path: Path) -> Iterator[TextIO]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("r", newline="", encoding="utf-8") as handle:
            yield handle


def _sha256(path: Path, *, uncompressed: bool = False) -> str:
    digest = hashlib.sha256()
    if uncompressed and path.name.endswith(".gz"):
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    with handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_digest(*fields: str) -> bytes:
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _token_digest(tokens: Sequence[int]) -> bytes:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "big", signed=False))
    return digest.digest()


def _read_token_list(text: str, *, row: int, column: str) -> list[int]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"tokenized row {row:,} column {column!r} is invalid JSON"
        ) from exc
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise ValueError(
            f"tokenized row {row:,} column {column!r} is not an integer list"
        )
    return value


def _aligned_rows(
    raw_path: Path,
    token_path: Path,
) -> Iterator[tuple[int, list[str], list[str]]]:
    missing = object()
    with _open_text(raw_path) as raw_handle, _open_text(token_path) as token_handle:
        raw_reader = csv.reader(raw_handle)
        token_reader = csv.reader(token_handle)
        for reader, path in ((raw_reader, raw_path), (token_reader, token_path)):
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"empty CSV: {path}") from exc
            if tuple(header) != CSV_HEADER:
                raise ValueError(f"bad CSV header in {path}: {header!r}")
        for index, (raw_row, token_row) in enumerate(
            itertools.zip_longest(raw_reader, token_reader, fillvalue=missing),
            start=1,
        ):
            if raw_row is missing or token_row is missing:
                raise ValueError(
                    f"raw/tokenized row-count mismatch at data row {index:,}"
                )
            if len(raw_row) != 2 or len(token_row) != 2:
                raise ValueError(f"row {index:,} does not have exactly two columns")
            yield index, raw_row, token_row


def _validate_aligned_row(
    raw_row: list[str],
    token_row: list[str],
    *,
    row: int,
    tokenizer: ScatteringAmplitudeTokenizer,
    max_tokens: int,
) -> tuple[bytes, bytes, int, int]:
    encoded: list[list[int]] = []
    for column, expression, stored_text in zip(CSV_HEADER, raw_row, token_row):
        position = 0
        for match in tokenizer._token_re.finditer(expression):
            if expression[position : match.start()].strip():
                raise ValueError(
                    f"unsupported text at row {row:,}, column {column!r}"
                )
            position = match.end()
        if expression[position:].strip():
            raise ValueError(
                f"unsupported text at row {row:,}, column {column!r}"
            )
        labels = [
            int(label)
            for label in re.findall(r"(?:p|e|F|M)_(\d+)", expression)
        ]
        if any(label < 1 or label > 5 for label in labels):
            raise ValueError(
                f"particle label outside 1..5 at row {row:,}, column {column!r}"
            )
        stored = _read_token_list(stored_text, row=row, column=column)
        expected = tokenizer.encode_infix(expression)
        if stored != expected:
            raise ValueError(f"raw/token mismatch at row {row:,}, {column}")
        if len(expected) > max_tokens:
            raise ValueError(
                f"row {row:,}, {column} has {len(expected):,} tokens; "
                f"limit is {max_tokens:,}"
            )
        if 1 in expected:
            raise ValueError(f"row {row:,}, {column} contains <UNK>")
        encoded.append(expected)
    return (
        _token_digest(encoded[0]),
        _token_digest(encoded[1]),
        len(encoded[0]),
        len(encoded[1]),
    )


def _require_clean_target(expression: str, *, location: str) -> None:
    analysis = analyze_simple_expression(expression, assumptions=ASSUMPTIONS)
    if analysis.classification != "clean":
        raise ValueError(
            f"{location} has a {analysis.classification} zero-summand target"
        )
    dimension = manifest_mass_dimension(expression)
    if dimension != -1:
        raise ValueError(
            f"{location} has mass dimension {dimension}; expected -1 for 5PT"
        )


def _offer_numeric_sample(
    heap: list[tuple[int, str, str]],
    *,
    limit: int,
    pair_digest: bytes,
    simple: str,
    scrambled: str,
) -> None:
    if limit <= 0:
        return
    rank = int.from_bytes(
        hashlib.sha256(b"sqed5-numeric-audit\0" + pair_digest).digest()[:8],
        "big",
    )
    item = (-rank, simple, scrambled)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _manifest_profile(manifest: dict[str, object]) -> dict[str, object]:
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("manifest generation section is missing")
    settings = generation.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("manifest generation settings are missing")
    profile = settings.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("manifest profile settings are missing")
    return profile


def _source_path(
    manifest: dict[str, object],
    kind: str,
    override: Path | None,
) -> Path:
    if override is not None:
        return override.expanduser()
    generation = manifest["generation"]
    assert isinstance(generation, dict)
    artifacts = generation.get("source_artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(kind), dict):
        raise ValueError(f"manifest source artifact {kind!r} is missing")
    return Path(str(artifacts[kind]["path"]))


def _verify_file_metadata(path: Path, metadata: dict[str, object]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(metadata["bytes"]):
        raise ValueError(f"byte-size mismatch for {path}")
    if _sha256(path) != str(metadata["sha256"]):
        raise ValueError(f"SHA-256 mismatch for {path}")
    expected_uncompressed = metadata.get("uncompressed_sha256")
    if expected_uncompressed is not None and _sha256(
        path, uncompressed=True
    ) != str(expected_uncompressed):
        raise ValueError(f"uncompressed SHA-256 mismatch for {path}")


def _candidate_specs(
    manifest: dict[str, object],
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    profile = _manifest_profile(manifest)
    candidate_counts: dict[str, int] = {}
    release_counts: dict[str, int] = {}
    boundaries: list[int] = []
    total = 0
    for name in STRATA:
        entry = profile.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest profile {name!r} is missing")
        candidate_counts[name] = int(entry["candidate_rows"])
        release_counts[name] = int(entry["release_rows"])
        total += candidate_counts[name]
        boundaries.append(total)
    return boundaries, candidate_counts, release_counts


def _stratum(index_zero_based: int, boundaries: Sequence[int]) -> str:
    for name, boundary in zip(STRATA, boundaries):
        if index_zero_based < boundary:
            return name
    raise ValueError(f"candidate source has an extra row at {index_zero_based + 1:,}")


def _find_candidate_ambiguities(
    raw_path: Path,
    token_path: Path,
    *,
    expected_rows: int,
    progress_every: int,
) -> frozenset[bytes]:
    first_target: dict[bytes, bytes] = {}
    ambiguous: set[bytes] = set()
    rows = 0
    for row, _raw_row, token_row in _aligned_rows(raw_path, token_path):
        rows = row
        target = _token_digest(
            _read_token_list(token_row[0], row=row, column="simple")
        )
        scrambled = _token_digest(
            _read_token_list(token_row[1], row=row, column="scrambled")
        )
        previous = first_target.setdefault(scrambled, target)
        if previous != target:
            ambiguous.add(scrambled)
        if progress_every and row % progress_every == 0:
            print(f"preflighted {row:,}/{expected_rows:,} candidate rows")
    if rows != expected_rows:
        raise ValueError(f"candidate rows are {rows:,}; expected {expected_rows:,}")
    return frozenset(ambiguous)


def _verify_candidates(
    raw_path: Path,
    token_path: Path,
    manifest: dict[str, object],
    *,
    tokenizer: ScatteringAmplitudeTokenizer,
    max_tokens: int,
    full_zero_audit: bool,
    progress_every: int,
    excluded_ambiguous_inputs: frozenset[bytes],
) -> tuple[dict[bytes, str], frozenset[bytes], dict[str, object]]:
    boundaries, candidate_counts, release_counts = _candidate_specs(manifest)
    expected_rows = boundaries[-1]
    replay_pairs: set[bytes] = set()
    replay_inputs: dict[bytes, bytes] = {}
    accepted_pairs: dict[bytes, str] = {}
    accepted = Counter()
    accepted_target_counts: Counter[bytes] = Counter()
    accepted_target_stratum: dict[bytes, str] = {}

    all_pairs: set[bytes] = set()
    all_inputs: dict[bytes, bytes] = {}
    full_pair_duplicates = Counter()
    full_repeated_inputs = Counter()
    full_ambiguous_inputs = Counter()
    max_simple = 0
    max_scrambled = 0
    rows = 0

    for row, raw_row, token_row in _aligned_rows(raw_path, token_path):
        rows = row
        name = _stratum(row - 1, boundaries)
        target, scrambled, simple_len, scrambled_len = _validate_aligned_row(
            raw_row,
            token_row,
            row=row,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        max_simple = max(max_simple, simple_len)
        max_scrambled = max(max_scrambled, scrambled_len)
        if full_zero_audit:
            _require_clean_target(raw_row[0], location=f"candidate row {row:,}")

        pair = _field_digest(raw_row[0], raw_row[1])
        if pair in all_pairs:
            full_pair_duplicates[name] += 1
        else:
            all_pairs.add(pair)
        previous = all_inputs.get(scrambled)
        if previous is None:
            all_inputs[scrambled] = target
        elif previous == target:
            full_repeated_inputs[name] += 1
        else:
            full_ambiguous_inputs[name] += 1

        if accepted[name] < release_counts[name]:
            if pair in replay_pairs:
                continue
            if scrambled in excluded_ambiguous_inputs:
                continue
            previous = replay_inputs.get(scrambled)
            if previous is not None:
                continue
            replay_pairs.add(pair)
            replay_inputs[scrambled] = target
            accepted_pairs[pair] = name
            accepted[name] += 1
            accepted_target_counts[target] += 1
            accepted_target_stratum.setdefault(target, name)

        if progress_every and row % progress_every == 0:
            print(f"verified {row:,}/{expected_rows:,} candidate rows")

    if rows != expected_rows:
        raise ValueError(f"candidate rows are {rows:,}; expected {expected_rows:,}")
    if dict(accepted) != release_counts:
        raise ValueError(
            f"candidate replay did not reach release quotas: {dict(accepted)!r}"
        )
    if len(accepted_pairs) != sum(release_counts.values()):
        raise ValueError("candidate replay accepted-pair count is inconsistent")

    profile = _manifest_profile(manifest)
    split_seed = int(manifest["split"]["seed"])
    selected_test_targets: set[bytes] = set()
    eligible_singletons: dict[str, int] = {}
    for offset, name in enumerate(STRATA):
        eligible = sorted(
            target
            for target, count in accepted_target_counts.items()
            if count == 1 and accepted_target_stratum[target] == name
        )
        eligible_singletons[name] = len(eligible)
        entry = profile[name]
        assert isinstance(entry, dict)
        test_rows = int(entry["test_rows"])
        if len(eligible) < test_rows:
            raise ValueError(f"{name} lacks enough singleton test targets")
        rng = random.Random(split_seed + 1_000_003 * offset)
        selected_test_targets.update(rng.sample(eligible, test_rows))

    if eligible_singletons != manifest["split"]["eligible_singleton_targets_by_stratum"]:
        raise ValueError("eligible singleton-target counts disagree with manifest")

    return accepted_pairs, frozenset(selected_test_targets), {
        "rows": rows,
        "candidate_by_stratum": candidate_counts,
        "accepted_by_stratum": dict(accepted),
        "full_pair_duplicates_by_stratum": {
            name: full_pair_duplicates[name] for name in STRATA
        },
        "full_repeated_inputs_by_stratum": {
            name: full_repeated_inputs[name] for name in STRATA
        },
        "full_ambiguous_inputs_by_stratum": {
            name: full_ambiguous_inputs[name] for name in STRATA
        },
        "full_ambiguous_input_groups": len(excluded_ambiguous_inputs),
        "max_simple_tokens": max_simple,
        "max_scrambled_tokens": max_scrambled,
        "all_targets_zero_audited": full_zero_audit,
    }


def _release_paths(
    manifest: dict[str, object],
    release_dir: Path | None,
) -> dict[str, Path]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("manifest outputs section is missing")
    paths: dict[str, Path] = {}
    for name in ("train_raw", "train_tokenized", "test_raw", "test_tokenized"):
        metadata = outputs.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"manifest output {name!r} is missing")
        stored = Path(str(metadata["path"]))
        paths[name] = (
            release_dir.expanduser() / stored.name
            if release_dir is not None
            else stored
        )
    return paths


def _verify_release_split(
    split_name: str,
    raw_path: Path,
    token_path: Path,
    *,
    tokenizer: ScatteringAmplitudeTokenizer,
    max_tokens: int,
    accepted_pairs: dict[bytes, str],
    selected_test_targets: frozenset[bytes],
    numeric_sample_heap: list[tuple[int, str, str]],
    numeric_sample_limit: int,
    global_pairs: set[bytes],
    global_inputs: set[bytes],
    progress_every: int,
) -> tuple[int, Counter[str], set[bytes], int, int]:
    counts: Counter[str] = Counter()
    targets: set[bytes] = set()
    rows = 0
    max_simple = 0
    max_scrambled = 0
    for row, raw_row, token_row in _aligned_rows(raw_path, token_path):
        rows = row
        target, scrambled, simple_len, scrambled_len = _validate_aligned_row(
            raw_row,
            token_row,
            row=row,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        _require_clean_target(
            raw_row[0],
            location=f"{split_name} row {row:,}",
        )
        pair = _field_digest(raw_row[0], raw_row[1])
        if pair in global_pairs:
            raise ValueError(f"duplicate released pair at {split_name} row {row:,}")
        if scrambled in global_inputs:
            raise ValueError(
                f"duplicate released scrambled input at {split_name} row {row:,}"
            )
        stratum = accepted_pairs.get(pair)
        if stratum is None:
            raise ValueError(
                f"{split_name} row {row:,} was not selected by candidate replay"
            )
        should_be_test = target in selected_test_targets
        if should_be_test != (split_name == "test"):
            raise ValueError(
                f"{split_name} row {row:,} disagrees with deterministic split replay"
            )
        global_pairs.add(pair)
        global_inputs.add(scrambled)
        _offer_numeric_sample(
            numeric_sample_heap,
            limit=numeric_sample_limit,
            pair_digest=pair,
            simple=raw_row[0],
            scrambled=raw_row[1],
        )
        counts[stratum] += 1
        targets.add(target)
        max_simple = max(max_simple, simple_len)
        max_scrambled = max(max_scrambled, scrambled_len)
        if progress_every and row % progress_every == 0:
            print(f"verified {row:,} {split_name} rows")
    return rows, counts, targets, max_simple, max_scrambled


def verify(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = args.manifest.expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset", {}).get("n_particles") != 5:
        raise ValueError("manifest is not a 5PT dataset")
    if manifest.get("split", {}).get("leakage_key") != "sha256(tokenized_simple)":
        raise ValueError("unexpected split leakage key")

    settings = manifest["generation"]["settings"]
    max_tokens = int(settings["max_tokens"])
    max_particles = int(settings["tokenizer_max_particles"])
    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=max_particles,
        max_sequence_length=None,
    )

    raw_candidate = _source_path(manifest, "raw", args.candidate_raw)
    token_candidate = _source_path(
        manifest,
        "tokenized",
        args.candidate_tokenised,
    )
    source_artifacts = manifest["generation"]["source_artifacts"]
    _verify_file_metadata(raw_candidate, source_artifacts["raw"])
    _verify_file_metadata(token_candidate, source_artifacts["tokenized"])

    boundaries, _candidate_counts, _release_counts = _candidate_specs(manifest)
    excluded_ambiguous_inputs = _find_candidate_ambiguities(
        raw_candidate,
        token_candidate,
        expected_rows=boundaries[-1],
        progress_every=args.progress_every,
    )

    accepted_pairs, selected_test_targets, candidate_report = _verify_candidates(
        raw_candidate,
        token_candidate,
        manifest,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        full_zero_audit=args.full_zero_audit,
        progress_every=args.progress_every,
        excluded_ambiguous_inputs=excluded_ambiguous_inputs,
    )
    manifest_ambiguous_groups = manifest["verification"].get(
        "full_candidate_ambiguous_scrambled_input_groups"
    )
    if manifest_ambiguous_groups != len(excluded_ambiguous_inputs):
        raise ValueError(
            "full candidate ambiguous-input group count disagrees with manifest"
        )

    release_paths = _release_paths(manifest, args.release_dir)
    output_metadata = manifest["outputs"]
    for name, path in release_paths.items():
        _verify_file_metadata(path, output_metadata[name])
        if path.name.endswith(".gz"):
            with path.open("rb") as handle:
                if handle.read(2) != b"\x1f\x8b":
                    raise ValueError(f"invalid gzip magic bytes: {path}")

    global_pairs: set[bytes] = set()
    global_inputs: set[bytes] = set()
    numeric_sample_heap: list[tuple[int, str, str]] = []
    train = _verify_release_split(
        "train",
        release_paths["train_raw"],
        release_paths["train_tokenized"],
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        accepted_pairs=accepted_pairs,
        selected_test_targets=selected_test_targets,
        numeric_sample_heap=numeric_sample_heap,
        numeric_sample_limit=args.numeric_samples,
        global_pairs=global_pairs,
        global_inputs=global_inputs,
        progress_every=args.progress_every,
    )
    test = _verify_release_split(
        "test",
        release_paths["test_raw"],
        release_paths["test_tokenized"],
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        accepted_pairs=accepted_pairs,
        selected_test_targets=selected_test_targets,
        numeric_sample_heap=numeric_sample_heap,
        numeric_sample_limit=args.numeric_samples,
        global_pairs=global_pairs,
        global_inputs=global_inputs,
        progress_every=args.progress_every,
    )
    train_rows, train_counts, train_targets, train_simple, train_scrambled = train
    test_rows, test_counts, test_targets, test_simple, test_scrambled = test

    dataset = manifest["dataset"]
    if train_rows != int(dataset["train_rows"]):
        raise ValueError(f"train rows are {train_rows:,}, manifest says {dataset['train_rows']}")
    if test_rows != int(dataset["test_rows"]):
        raise ValueError(f"test rows are {test_rows:,}, manifest says {dataset['test_rows']}")
    if len(global_pairs) != int(dataset["total_rows"]):
        raise ValueError("released union does not contain the declared total rows")
    if train_targets.intersection(test_targets):
        raise ValueError("exact tokenized target leakage across train/test")
    if len(test_targets) != test_rows:
        raise ValueError("test targets are not all unique")
    if test_targets != set(selected_test_targets):
        raise ValueError("test targets disagree with deterministic split replay")
    if set(global_pairs) != set(accepted_pairs):
        raise ValueError("release is not the exact candidate-replay partition")

    split = manifest["split"]
    if dict(train_counts) != split["train_by_stratum"]:
        raise ValueError(f"train stratum attribution mismatch: {dict(train_counts)!r}")
    if dict(test_counts) != split["test_by_stratum"]:
        raise ValueError(f"test stratum attribution mismatch: {dict(test_counts)!r}")

    mass = float(settings["mass"])
    for sample_index, (_rank, simple, scrambled) in enumerate(
        sorted(numeric_sample_heap, reverse=True),
        start=1,
    ):
        valid, reason = _validate_pair(
            simple,
            scrambled,
            5,
            mass,
            n_checks=3,
            pol_modes=("coulomb", "covariant"),
            require_nonzero=True,
        )
        if not valid:
            raise ValueError(
                f"numerical audit sample {sample_index:,} failed: {reason}"
            )
        if args.progress_every and sample_index % 100 == 0:
            print(
                f"numerically verified {sample_index:,}/"
                f"{len(numeric_sample_heap):,} release samples"
            )

    code_hashes = manifest.get("code_sha256", {})
    repo_root = Path(__file__).resolve().parents[1]
    for relative, expected in code_hashes.items():
        path = repo_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"code hash mismatch: {relative}")
    log_metadata = manifest["generation"].get("log")
    if isinstance(log_metadata, dict):
        _verify_file_metadata(Path(log_metadata["path"]), log_metadata)

    return {
        "status": "verified",
        "candidate": candidate_report,
        "release": {
            "train_rows": train_rows,
            "test_rows": test_rows,
            "train_by_stratum": dict(train_counts),
            "test_by_stratum": dict(test_counts),
            "unique_pairs": len(global_pairs),
            "unique_scrambled_inputs": len(global_inputs),
            "test_unique_targets": len(test_targets),
            "train_test_exact_target_overlap": 0,
            "max_simple_tokens": max(train_simple, test_simple),
            "max_scrambled_tokens": max(train_scrambled, test_scrambled),
            "all_targets_zero_audited": True,
            "numerically_verified_samples": len(numeric_sample_heap),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/sqed/sqed_5pt_500k/sqed_5pt_oneshot_500000_manifest.json"),
    )
    parser.add_argument("--candidate-raw", type=Path)
    parser.add_argument("--candidate-tokenised", type=Path)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--full-zero-audit", action="store_true")
    parser.add_argument("--numeric-samples", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.numeric_samples < 0:
        parser.error("--numeric-samples must be non-negative")
    try:
        report = verify(args)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        report_path = args.report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            dir=report_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, report_path)
            os.chmod(report_path, 0o644)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
