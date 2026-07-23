#!/usr/bin/env python3
"""Generate synthetic and held-out five-point gravity simplification data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..Tokenizer import ScatteringAmplitudeTokenizer
from .core import (
    BENCHMARKS,
    PROCESS_SPECS,
    expand_expression,
    field_strength_counts_per_term,
    generate_target,
    is_benchmark_leak,
    numerically_equivalent,
    verify_paper_benchmarks,
)
from .scramble import scramble_trajectory


@dataclass(frozen=True)
class Candidate:
    simple: str
    scrambled: str
    process: str
    kind: str
    seed: int
    scramble_depth: int
    scramble_labels: str
    stage: str
    compact_terms: int
    simple_tokens: int
    scrambled_tokens: int
    relative_error: float


def _tokenizer(max_tokens: int) -> ScatteringAmplitudeTokenizer:
    return ScatteringAmplitudeTokenizer(
        max_particles=8, max_sequence_length=max_tokens
    )


def _token_lengths(
    tokenizer: ScatteringAmplitudeTokenizer, simple: str, scrambled: str
) -> tuple[int, int]:
    return len(tokenizer.encode_infix(simple)), len(tokenizer.encode_infix(scrambled))


def _make_candidate(
    *,
    process: str,
    kind: str,
    seed: int,
    min_scr: int,
    max_scr: int,
    min_terms: int,
    max_terms: int,
    max_tokens: int,
    validate: bool,
    scramble_names: Sequence[str] | None,
) -> Candidate | None:
    rng = random.Random(seed)
    spec = PROCESS_SPECS[process]
    compact = generate_target(
        process, rng=rng, min_terms=min_terms, max_terms=max_terms
    )
    if is_benchmark_leak(compact, process):
        return None
    expanded = expand_expression(compact)

    depth = rng.randint(min_scr, max_scr)
    trajectory = scramble_trajectory(
        expanded,
        spec,
        rng=rng,
        depth=depth,
        names=scramble_names,
    )
    if not trajectory:
        return None

    tokenizer = _tokenizer(max_tokens)
    if kind == "oneshot":
        target = compact
        source = trajectory[-1].expression
        stage = "compact"
        labels = [step.label for step in trajectory]
        used_depth = len(trajectory)
    elif kind == "staged":
        possible: list[tuple[str, str, str, int, list[str]]] = [
            (compact, expanded, "expanded-to-compact", 0, [])
        ]
        previous = expanded
        prefix_labels: list[str] = []
        for step in trajectory:
            prefix_labels.append(step.label)
            try:
                previous_len, current_len = _token_lengths(
                    tokenizer, previous, step.expression
                )
            except ValueError:
                previous = step.expression
                continue
            if previous_len < current_len:
                possible.append(
                    (
                        previous,
                        step.expression,
                        f"scramble-{step.depth}-to-{step.depth - 1}",
                        step.depth,
                        prefix_labels[:],
                    )
                )
            previous = step.expression
        target, source, stage, used_depth, labels = rng.choice(possible)
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    try:
        simple_tokens, scrambled_tokens = _token_lengths(tokenizer, target, source)
    except ValueError:
        return None
    if max(simple_tokens, scrambled_tokens) > max_tokens:
        return None

    relative_error = 0.0
    if validate:
        try:
            ok, relative_error = numerically_equivalent(
                target,
                source,
                process,
                seeds=(seed % 100_000 + 101,),
            )
        except Exception:
            return None
        if not ok:
            return None

    return Candidate(
        simple=target,
        scrambled=source,
        process=process,
        kind=kind,
        seed=seed,
        scramble_depth=used_depth,
        scramble_labels=",".join(labels),
        stage=stage,
        compact_terms=len(field_strength_counts_per_term(compact)),
        simple_tokens=simple_tokens,
        scrambled_tokens=scrambled_tokens,
        relative_error=relative_error,
    )


def _quota(samples: int, process: str, kind: str) -> list[tuple[str, str, int]]:
    processes = ("3s2h", "4s1h") if process == "mixed" else (process,)
    kinds = ("oneshot", "staged") if kind == "mixed" else (kind,)
    cells = [(p, k) for p in processes for k in kinds]
    base, remainder = divmod(samples, len(cells))
    return [
        (p, k, base + (1 if index < remainder else 0))
        for index, (p, k) in enumerate(cells)
    ]


def build_dataset(
    samples: int,
    *,
    process: str = "mixed",
    kind: str = "mixed",
    seed: int = 42,
    min_scr: int = 1,
    max_scr: int = 5,
    min_terms: int = 1,
    max_terms: int = 3,
    max_tokens: int = 4096,
    validate: bool = True,
    scramble_names: Sequence[str] | None = None,
    max_attempts_factor: int = 80,
) -> list[Candidate]:
    """Build exact balanced quotas, deduplicated by the training pair."""
    if process not in (*PROCESS_SPECS, "mixed"):
        raise ValueError(process)
    if kind not in ("oneshot", "staged", "mixed"):
        raise ValueError(kind)
    output: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    stream_seed = int(seed)
    for cell_index, (cell_process, cell_kind, target) in enumerate(
        _quota(samples, process, kind)
    ):
        accepted = 0
        attempts = 0
        while accepted < target and attempts < target * max_attempts_factor:
            candidate_seed = stream_seed + cell_index * 10**12 + attempts
            attempts += 1
            candidate = _make_candidate(
                process=cell_process,
                kind=cell_kind,
                seed=candidate_seed,
                min_scr=min_scr,
                max_scr=max_scr,
                min_terms=min_terms,
                max_terms=max_terms,
                max_tokens=max_tokens,
                validate=validate,
                scramble_names=scramble_names,
            )
            if candidate is None:
                continue
            key = (candidate.simple, candidate.scrambled)
            if key in seen:
                continue
            seen.add(key)
            output.append(candidate)
            accepted += 1
        if accepted != target:
            raise RuntimeError(
                f"Generated {accepted}/{target} rows for {cell_process}/{cell_kind} "
                f"after {attempts} attempts"
            )
    return output


def _worker_build(args: tuple) -> list[Candidate]:
    return build_dataset(*args[0], **args[1])


def build_dataset_parallel(
    samples: int,
    *,
    jobs: int,
    **kwargs,
) -> list[Candidate]:
    """Parallel deterministic generation with exact post-dedupe cell quotas."""
    if jobs <= 1 or samples < jobs * 4:
        return build_dataset(samples, **kwargs)
    base, remainder = divmod(samples, jobs)
    tasks = []
    root_seed = int(kwargs.get("seed", 42))
    for index in range(jobs):
        count = base + (index < remainder)
        worker_kwargs = dict(kwargs)
        worker_kwargs["seed"] = root_seed + index * 10**9
        tasks.append(((count,), worker_kwargs))
    context = mp.get_context("spawn")
    with context.Pool(jobs) as pool:
        chunks = pool.map(_worker_build, tasks)
    process = str(kwargs.get("process", "mixed"))
    kind = str(kwargs.get("kind", "mixed"))
    desired = {
        (cell_process, cell_kind): count
        for cell_process, cell_kind, count in _quota(samples, process, kind)
    }
    accepted_counts: dict[tuple[str, str], int] = {
        key: 0 for key in desired
    }
    output: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        for candidate in chunk:
            key = (candidate.simple, candidate.scrambled)
            cell = (candidate.process, candidate.kind)
            if (
                key not in seen
                and cell in desired
                and accepted_counts[cell] < desired[cell]
            ):
                seen.add(key)
                output.append(candidate)
                accepted_counts[cell] += 1

    fill_round = 0
    while accepted_counts != desired:
        fill_round += 1
        if fill_round > 100:
            raise RuntimeError(
                f"Could not fill post-dedupe quotas: {accepted_counts} != {desired}"
            )
        for cell_index, (cell, target) in enumerate(desired.items()):
            missing = target - accepted_counts[cell]
            if missing <= 0:
                continue
            fill_kwargs = dict(kwargs)
            fill_kwargs.update(
                {
                    "process": cell[0],
                    "kind": cell[1],
                    "seed": (
                        root_seed
                        + jobs * 10**9
                        + fill_round * 10**7
                        + cell_index * 10**5
                    ),
                }
            )
            for candidate in build_dataset(missing, **fill_kwargs):
                key = (candidate.simple, candidate.scrambled)
                if key in seen:
                    continue
                seen.add(key)
                output.append(candidate)
                accepted_counts[cell] += 1
                if accepted_counts[cell] == target:
                    break
    if len(output) != samples:
        raise RuntimeError(f"Post-dedupe row count {len(output)} != {samples}")
    return output


def _open_text(path: str | Path, mode: str):
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")
    return open(path, mode, newline="", encoding="utf-8")


def write_raw(candidates: Iterable[Candidate], path: str | Path) -> None:
    with _open_text(path, "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["simple", "scrambled"])
        for candidate in candidates:
            writer.writerow([candidate.simple, candidate.scrambled])


def write_metadata(candidates: Iterable[Candidate], path: str | Path) -> None:
    fields = list(Candidate.__dataclass_fields__)
    with _open_text(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def tokenise(
    candidates: Iterable[Candidate],
    path: str | Path,
    *,
    max_tokens: int = 4096,
) -> None:
    tokenizer = _tokenizer(max_tokens)
    with _open_text(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["simple", "scrambled"])
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "simple": json.dumps(tokenizer.encode_infix(candidate.simple)),
                    "scrambled": json.dumps(
                        tokenizer.encode_infix(candidate.scrambled)
                    ),
                }
            )


def build_benchmarks(
    *,
    scrambles_per_amplitude: int = 100,
    seed: int = 240804720,
    max_tokens: int = 4096,
    validate: bool = True,
    scramble_names: Sequence[str] | None = None,
) -> list[Candidate]:
    """Generate equal numbers at depths 1--5 for each held-out amplitude."""
    verify_paper_benchmarks()
    if scrambles_per_amplitude % 5:
        raise ValueError("scrambles_per_amplitude must be divisible by five")
    tokenizer = _tokenizer(max_tokens)
    output: list[Candidate] = []
    per_depth = scrambles_per_amplitude // 5
    for process_index, (process, compact) in enumerate(BENCHMARKS.items()):
        spec = PROCESS_SPECS[process]
        expanded = expand_expression(compact)
        seen: set[str] = set()
        for depth in range(1, 6):
            accepted = 0
            attempts = 0
            while accepted < per_depth and attempts < per_depth * 500:
                candidate_seed = (
                    seed + process_index * 10**9 + depth * 10**6 + attempts
                )
                attempts += 1
                rng = random.Random(candidate_seed)
                trajectory = scramble_trajectory(
                    expanded,
                    spec,
                    rng=rng,
                    depth=depth,
                    names=scramble_names,
                )
                if len(trajectory) != depth:
                    continue
                source = trajectory[-1].expression
                if source in seen:
                    continue
                try:
                    simple_tokens, scrambled_tokens = _token_lengths(
                        tokenizer, compact, source
                    )
                except ValueError:
                    continue
                if validate:
                    ok, error = numerically_equivalent(
                        compact,
                        source,
                        process,
                        seeds=(candidate_seed % 100_000 + 101,),
                    )
                    if not ok:
                        continue
                else:
                    error = 0.0
                seen.add(source)
                output.append(
                    Candidate(
                        simple=compact,
                        scrambled=source,
                        process=process,
                        kind="benchmark",
                        seed=candidate_seed,
                        scramble_depth=depth,
                        scramble_labels=",".join(
                            step.label for step in trajectory
                        ),
                        stage="held-out-paper",
                        compact_terms=2 if process == "3s2h" else 3,
                        simple_tokens=simple_tokens,
                        scrambled_tokens=scrambled_tokens,
                        relative_error=error,
                    )
                )
                accepted += 1
            if accepted != per_depth:
                raise RuntimeError(
                    f"Generated {accepted}/{per_depth} benchmark rows for "
                    f"{process} at depth {depth}"
                )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument(
        "--process", choices=(*PROCESS_SPECS, "mixed"), default="mixed"
    )
    parser.add_argument(
        "--kind", choices=("oneshot", "staged", "mixed"), default="mixed"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-scr", type=int, default=1)
    parser.add_argument("--max-scr", type=int, default=5)
    parser.add_argument("--min-terms", type=int, default=1)
    parser.add_argument("--max-terms", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--scrambles", nargs="*", default=None)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--benchmarks", action="store_true")
    parser.add_argument("--benchmark-samples", type=int, default=100)
    parser.add_argument(
        "--raw-out", default="data/gravity/gravity_5pt_100k_raw.csv.gz"
    )
    parser.add_argument(
        "--tok-out", default="data/gravity/gravity_5pt_100k_tok.csv.gz"
    )
    parser.add_argument(
        "--metadata-out", default="data/gravity/gravity_5pt_100k_metadata.csv.gz"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.benchmarks:
        candidates = build_benchmarks(
            scrambles_per_amplitude=args.benchmark_samples,
            seed=args.seed,
            max_tokens=args.max_tokens,
            validate=not args.no_validate,
            scramble_names=args.scrambles,
        )
    else:
        verify_paper_benchmarks()
        candidates = build_dataset_parallel(
            args.samples,
            jobs=args.jobs,
            process=args.process,
            kind=args.kind,
            seed=args.seed,
            min_scr=args.min_scr,
            max_scr=args.max_scr,
            min_terms=args.min_terms,
            max_terms=args.max_terms,
            max_tokens=args.max_tokens,
            validate=not args.no_validate,
            scramble_names=args.scrambles,
        )
    if len(candidates) != (
        args.benchmark_samples * 2 if args.benchmarks else args.samples
    ):
        raise RuntimeError("Incorrect final row count")
    write_raw(candidates, args.raw_out)
    tokenise(candidates, args.tok_out, max_tokens=args.max_tokens)
    write_metadata(candidates, args.metadata_out)
    print(
        f"Wrote {len(candidates)} rows to {args.raw_out}, {args.tok_out}, "
        f"and {args.metadata_out}"
    )


if __name__ == "__main__":
    main()
