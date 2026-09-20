#!/usr/bin/env python3
"""Generate synthetic training pairs and held-out ordered gravity components.

Example (100,000 training rows plus 200 test rows):
    python -m data_gen.ordered_gravity_gen --samples 100200 --test-size 200 --jobs 8

Like the cyclic Yang--Mills CLI, --samples includes the held-out rows. The test
set contains both auxiliary reconstruction pieces of each selected five-point
benchmark, as defined in data_gen_gravity/ORDERED_GRAVITY.md. Training targets
are synthetic gauge-invariant rational expressions in the same fixed leg-role
convention; they are not additional physical amplitudes. Single-F contractions
do not acquire the Yang--Mills cyclic-word restriction, and gravity poles are
not restricted to adjacent legs.

All benchmark parents, components and species-preserving relabelings are
reserved, including nonzero overall rescalings and staged descendants.
Structural exclusion is supplemented by finite numerical fingerprints; this
is not a proof against every possible algebraic equivalence. Every accepted
pair is numerically validated. Raw and tokenized train/test files have the
usual simple,scrambled columns, with ordering/reconstruction metadata and a
hashed manifest alongside them. Existing outputs require --overwrite.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_gen import gen_data as sqed
from data_gen.data_gen_gravity.core import (
    BENCHMARKS, PROCESS_SPECS, _relabel, compact_signature, eval_expression,
    expand_expression, field_strength_counts_per_term, numerically_equivalent,
)
from data_gen.data_gen_gravity.generate import (
    Candidate, _make_candidate, _open_text, _quota, tokenise, write_raw,
)
from data_gen.data_gen_gravity.kinematics import generate_kinematics
from data_gen.data_gen_gravity.ordered_benchmarks import (
    DEFAULT_ORDER, ordered_seed, reconstruction_terms, verify_reconstruction,
)
from data_gen.data_gen_gravity.scramble import normalise_names, scramble_trajectory
from data_gen.data_gen_ym.generate_clean_4pt import _publish_all
from data_gen.gravity_generation import _sha256, audit_token_files

DEFINITION_VERSION = "auxiliary-m3-m4-v1"
FINGERPRINT_SEEDS = (17, 31, 73, 107)
PARTNERS = {"3s2h": (1, 2, 3, 5, 4), "4s1h": (3, 4, 1, 2, 5)}


def parenthesize_for_semantic_tokenization(expression: str) -> str:
    """Preserve the physics AST in prefix tokens, including m4's half factors.

    The legacy expansion produces decimal coefficients, whereas the shared
    vocabulary contains integer digits. Render rational literals as explicit
    integer divisions, and dot chains as atomic factors. The tokenizer assigns
    dot the same precedence as multiplication, so these parentheses matter.
    """
    compact_signature(expression)  # Strict complete-input syntax validation.
    tree = sqed._Parser(sqed._tokenize(expression)).parse()

    def render(node) -> str:
        if isinstance(node, sqed._Num):
            coefficient = Fraction(str(node.value))
            if coefficient.denominator != 1:
                raise ValueError("A standalone fractional literal cannot be represented by the legacy tokenizer")
            return str(coefficient.numerator)
        if isinstance(node, sqed._Vec):
            return f"{node.tag}_{node.idx}"
        if isinstance(node, sqed._DotChain):
            return "(" + " · ".join(render(part) for part in node.parts) + ")"
        if isinstance(node, sqed._UnaryOp):
            return f"(-({render(node.operand)}))"
        if isinstance(node, sqed._BinOp):
            if node.op in ("*", "/"):
                # Prefix digits have no literal separator: / 1: 2: is read as
                # the single integer 12. Absorb a coefficient such as 0.5 into
                # the whole monomial, rendering X/(2*Y), never (1/2)*X/Y.
                coefficient = Fraction(1)
                numerator, denominator = [], []

                def collect(current, inverted=False):
                    nonlocal coefficient
                    if isinstance(current, sqed._Num):
                        value = Fraction(str(current.value))
                        coefficient = coefficient / value if inverted else coefficient * value
                    elif isinstance(current, sqed._UnaryOp):
                        coefficient *= -1
                        collect(current.operand, inverted)
                    elif isinstance(current, sqed._BinOp) and current.op in ("*", "/"):
                        collect(current.left, inverted)
                        collect(current.right, not inverted if current.op == "/" else inverted)
                    else:
                        (denominator if inverted else numerator).append(render(current))

                collect(node)
                if coefficient == 0:
                    return "0"
                if abs(coefficient.numerator) != 1 or not numerator:
                    numerator.insert(0, str(abs(coefficient.numerator)))
                if coefficient.denominator != 1:
                    denominator.insert(0, str(coefficient.denominator))
                top = "(" + "*".join(numerator) + ")" if len(numerator) > 1 else numerator[0]
                if denominator:
                    bottom = "(" + "*".join(denominator) + ")" if len(denominator) > 1 else denominator[0]
                    top = f"({top}/{bottom})"
                return f"(-({top}))" if coefficient < 0 else top
            operator = "^" if node.op == "**" else node.op
            return f"({render(node.left)}{operator}{render(node.right)})"
        raise ValueError(f"Unsupported gravity expression node: {type(node).__name__}")

    return render(tree)


def _processes(process: str) -> tuple[str, ...]:
    if process not in (*PROCESS_SPECS, "mixed"):
        raise ValueError(f"Unknown process: {process}")
    return tuple(PROCESS_SPECS) if process == "mixed" else (process,)


def _scaled_signature(expression: str) -> tuple:
    """Structural Laurent-monomial signature modulo an overall rational scale.

    compact_signature handles F antisymmetry and strict parsing. Here we also
    collect numeric coefficients, cancel common factors and combine like terms.
    Factored additive subexpressions remain structural atoms; the independent
    numerical family check covers their physical equivalences.
    """
    terms = defaultdict(Fraction)
    for sign, numerator, denominator in compact_signature(expression):
        coefficient = Fraction(sign)
        powers = Counter()
        for direction, factors in ((1, numerator), (-1, denominator)):
            for factor in factors:
                if factor[0] == "n":
                    number = Fraction(str(factor[1]))
                    coefficient = coefficient * number if direction == 1 else coefficient / number
                else:
                    powers[factor] += direction
        monomial = tuple(sorted((factor, power) for factor, power in powers.items() if power))
        terms[monomial] += coefficient
    ordered = sorted((monomial, coeff) for monomial, coeff in terms.items() if coeff)
    if not ordered:
        return ()
    scale = ordered[0][1]
    return tuple((monomial, coeff / scale) for monomial, coeff in ordered)


@lru_cache(maxsize=2)
def _fingerprint_points(process: str) -> tuple:
    return tuple(generate_kinematics(seed=seed, graviton_legs=PROCESS_SPECS[process].graviton_legs)
                 for seed in FINGERPRINT_SEEDS)


def _fingerprint(expression: str, process: str) -> np.ndarray:
    values = np.asarray([eval_expression(expression, kin)
                         for kin in _fingerprint_points(process)], dtype=complex)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite expression on the independent validation grid")
    return values


def _projective(values: np.ndarray) -> np.ndarray | None:
    pivot = int(np.argmax(np.abs(values)))
    if abs(values[pivot]) < 1e-12:
        return None
    return values / values[pivot]


@lru_cache(maxsize=2)
def _reserved_family(process: str) -> tuple[frozenset, np.ndarray]:
    spec = PROCESS_SPECS[process]
    signatures = {}
    # All species relabelings are reserved, even those changing the 4s1h channel.
    # This is a conservative holdout rule, not permission to sum those channels.
    for scalars in itertools.permutations(spec.scalar_legs):
        for gravitons in itertools.permutations(spec.graviton_legs):
            mapping = dict(zip(spec.scalar_legs + spec.graviton_legs, scalars + gravitons))
            for expression in (BENCHMARKS[process], *reconstruction_terms(process)):
                relabeled = _relabel(expression, mapping)
                signatures.setdefault(_scaled_signature(relabeled), relabeled)
    fingerprints = [_projective(_fingerprint(expression, process))
                    for expression in signatures.values()]
    if any(value is None for value in fingerprints):
        raise AssertionError("A reserved benchmark component vanished on the validation grid")
    return frozenset(signatures), np.asarray(fingerprints)


class OrderedHoldoutGuard:
    """Exclude entire parent/component families and exact test expressions."""

    def __init__(self, test_rows: Sequence[Candidate], max_tokens: int = 4096):
        if not test_rows:
            raise ValueError("The ordered test set must not be empty")
        self.tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=max_tokens)
        self.expressions = {
            tuple(self.tokenizer.encode_infix(parenthesize_for_semantic_tokenization(expression)))
            for row in test_rows for expression in (row.simple, row.scrambled)
        }
        self._origin_cache: dict[tuple[str, str], str | None] = {}

    def rejection_reason(self, row: Candidate) -> str | None:
        if not row.compact_origin:
            raise ValueError("Training candidate is missing its compact origin")
        if row.process not in PROCESS_SPECS:
            raise ValueError(f"Unknown gravity process: {row.process}")
        key = (row.process, row.compact_origin)
        if key not in self._origin_cache:
            signatures, reserved = _reserved_family(row.process)
            signature = _scaled_signature(row.compact_origin)
            reason = None
            if signature in signatures:
                reason = "ordered_benchmark_family"
            else:
                values = _projective(_fingerprint(row.compact_origin, row.process))
                if values is None:
                    reason = "zero_target"
                elif np.any(np.max(np.abs(reserved - values), axis=1) < 2e-8):
                    reason = "numerical_benchmark_family"
            if len(self._origin_cache) >= 8192:
                self._origin_cache.clear()
            self._origin_cache[key] = reason
        reason = self._origin_cache[key]
        if reason:
            return reason
        for expression in (row.simple, row.scrambled):
            tokens = tuple(self.tokenizer.encode_infix(parenthesize_for_semantic_tokenization(expression)))
            if tokens in self.expressions:
                return "test_expression"
        return None


def _prepare(row: Candidate, max_tokens: int) -> Candidate | None:
    """Align the numerical and tokenizer grammars and require a nonzero origin."""
    simple = parenthesize_for_semantic_tokenization(row.simple)
    scrambled = parenthesize_for_semantic_tokenization(row.scrambled)
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=max_tokens)
    try:
        target_tokens = tokenizer.encode_infix(simple)
        source_tokens = tokenizer.encode_infix(scrambled)
    except ValueError:
        return None
    if target_tokens == source_tokens:
        return None
    values = _fingerprint(row.compact_origin, row.process)
    if np.any(np.abs(values) < 1e-12):
        return None
    # This grid is independent of both construction validation and the holdout
    # fingerprint grid. It also checks the actual token stream's decoded meaning.
    ok, error = numerically_equivalent(
        simple, scrambled, row.process, seeds=(503,),
    )
    if not ok:
        return None
    for original, tokens in ((simple, target_tokens), (scrambled, source_tokens)):
        ok, token_error = numerically_equivalent(
            original, tokenizer.decode_infix(tokens), row.process, seeds=(997,),
            reference_modes=("first", "last"), gauge_shift=False,
        )
        if not ok:
            return None
        error = max(error, token_error)
    return replace(row, simple=simple, scrambled=scrambled,
                   simple_tokens=len(target_tokens), scrambled_tokens=len(source_tokens),
                   relative_error=max(row.relative_error, error))


def _training_candidate(kwargs: dict) -> Candidate | None:
    try:
        row = _make_candidate(**kwargs)
        return _prepare(row, kwargs["max_tokens"]) if row is not None else None
    except (ArithmeticError, ValueError):
        return None


def _build_test(
    test_size: int, *, process: str, seed: int, min_scr: int, max_scr: int,
    max_tokens: int, scramble_names: Sequence[str] | None, max_attempts_factor: int,
) -> list[Candidate]:
    components = [(name, component, expression)
                  for name in _processes(process)
                  for component, expression in enumerate(reconstruction_terms(name))]
    base, remainder = divmod(test_size, len(components))
    output = []
    seen = set()
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=max_tokens)
    for index, (name, component, compact) in enumerate(components):
        quota = base + (index < remainder)
        expanded = expand_expression(compact)
        for depth in range(min_scr, max_scr + 1):
            depth_base, depth_extra = divmod(quota, max_scr - min_scr + 1)
            wanted = depth_base + (depth - min_scr < depth_extra)
            accepted = 0
            for attempt in range(wanted * max_attempts_factor):
                if accepted == wanted:
                    break
                candidate_seed = seed + index * 10**12 + depth * 10**8 + attempt
                trajectory = scramble_trajectory(
                    expanded, PROCESS_SPECS[name], rng=random.Random(candidate_seed),
                    depth=depth, names=scramble_names,
                )
                if len(trajectory) != depth:
                    continue
                row = Candidate(
                    simple=compact, scrambled=trajectory[-1].expression, process=name,
                    kind="benchmark", seed=candidate_seed, scramble_depth=depth,
                    scramble_labels=",".join(step.label for step in trajectory),
                    stage=f"ordered-component-{component}",
                    compact_terms=len(field_strength_counts_per_term(compact)),
                    simple_tokens=0, scrambled_tokens=0, relative_error=0.0,
                    compact_origin=compact,
                )
                try:
                    row = _prepare(row, max_tokens)
                except (ArithmeticError, ValueError):
                    continue
                if row is None:
                    continue
                token_key = tuple(tokenizer.encode_infix(row.scrambled))
                if token_key in seen:
                    continue
                seen.add(token_key)
                output.append(row)
                accepted += 1
            if accepted != wanted:
                raise RuntimeError(f"Could not fill test quota for {name}/component-{component}/depth-{depth}: "
                                   f"{accepted}/{wanted}; increase --max-candidates-factor or --max-tokens")
    return output


def build_datasets(
    *, samples: int = 100_200, test_size: int = 200, process: str = "mixed",
    kind: str = "mixed", seed: int = 42, split_seed: int = 240804720, jobs: int = 1,
    max_tokens: int = 4096, min_scr: int = 1, max_scr: int = 5,
    min_terms: int = 1, max_terms: int = 3, max_attempts_factor: int = 80,
    scramble_names: Sequence[str] | None = None, batch_size: int = 128,
) -> tuple[list[Candidate], list[Candidate], dict]:
    """Return exact train/test quotas; samples includes the benchmark test rows."""
    processes = _processes(process)
    if kind not in ("oneshot", "staged", "mixed"):
        raise ValueError("kind must be oneshot, staged or mixed")
    if test_size < 2 * len(processes) or samples <= test_size:
        raise ValueError("--samples must exceed --test-size; test size must include both components per process")
    if min(jobs, max_tokens, min_scr, min_terms, max_attempts_factor, batch_size) < 1:
        raise ValueError("Jobs, token cap, scramble/term limits and attempt/batch sizes must be positive")
    if max_scr < min_scr or max_terms < min_terms:
        raise ValueError("Maximum scramble/term limits must be at least their minima")
    active_scrambles = normalise_names(scramble_names)
    if not active_scrambles:
        raise ValueError("At least one effective scrambler is required")
    reconstruction = verify_reconstruction((17, 31))
    test = _build_test(
        test_size, process=process, seed=split_seed, min_scr=min_scr, max_scr=max_scr,
        max_tokens=max_tokens, scramble_names=active_scrambles,
        max_attempts_factor=max_attempts_factor,
    )
    guard = OrderedHoldoutGuard(test, max_tokens)
    training: list[Candidate] = []
    rejected = Counter()
    seen = set()
    tokenizer = guard.tokenizer
    pool = mp.get_context("spawn").Pool(jobs) if jobs > 1 else None
    try:
        for cell_index, (name, mode, wanted) in enumerate(_quota(samples - test_size, process, kind)):
            accepted = attempts = 0
            while accepted < wanted and attempts < wanted * max_attempts_factor:
                count = min(batch_size, wanted - accepted, wanted * max_attempts_factor - attempts)
                tasks = [
                    dict(process=name, kind=mode, seed=seed + cell_index * 10**12 + attempts + offset,
                         min_scr=min_scr, max_scr=max_scr, min_terms=min_terms, max_terms=max_terms,
                         max_tokens=max_tokens, validate=True, scramble_names=active_scrambles)
                    for offset in range(count)
                ]
                candidates = pool.map(_training_candidate, tasks) if pool else map(_training_candidate, tasks)
                attempts += count
                for row in candidates:
                    if row is None:
                        rejected["invalid_zero_or_token_limit"] += 1
                        continue
                    reason = guard.rejection_reason(row)
                    if reason:
                        rejected[reason] += 1
                        continue
                    key = (tuple(tokenizer.encode_infix(row.simple)), tuple(tokenizer.encode_infix(row.scrambled)))
                    if key in seen:
                        rejected["duplicate_token_pair"] += 1
                        continue
                    seen.add(key)
                    training.append(row)
                    accepted += 1
            if accepted != wanted:
                raise RuntimeError(f"Could not fill training quota for {name}/{mode}: {accepted}/{wanted}")
    finally:
        if pool:
            pool.close()
            pool.join()
    report = {
        "schema_version": 1, "generator": "data_gen.ordered_gravity_gen",
        "definition_version": DEFINITION_VERSION, "reference_order": list(DEFAULT_ORDER),
        "samples_includes_test": True, "max_tokens": max_tokens, "tokenizer_max_particles": 8,
        "training": {"rows": len(training), "seed": seed,
                     "object_kind": "synthetic_fixed_role_expression",
                     "counts": dict(Counter(f"{r.process}/{r.kind}" for r in training))},
        "test": {"rows": len(test), "seed": split_seed,
                 "unique_component_targets": 2 * len(processes),
                 "object_kind": "auxiliary_ordered_benchmark_seed",
                 "counts": dict(Counter(f"{r.process}/{r.stage}/depth-{r.scramble_depth}" for r in test))},
        "settings": {"process": process, "kind": kind, "jobs": jobs, "batch_size": batch_size,
                     "min_scr": min_scr, "max_scr": max_scr, "min_terms": min_terms,
                     "max_terms": max_terms, "scrambles": list(active_scrambles),
                     "max_candidates_factor": max_attempts_factor},
        "holdout": {"reserved": "both full parents, both components and all species-preserving relabelings",
                    "rational_rescaling_excluded": True, "staged_origin_checked": True,
                    "fingerprint_seeds": list(FINGERPRINT_SEEDS),
                    "limitation": "structural and finite numerical checks, not arbitrary algebraic inequivalence"},
        "rejected": dict(rejected), "reconstruction": reconstruction,
    }
    return training, test, report


def _metadata(row: Candidate) -> dict:
    benchmark = row.kind == "benchmark"
    component = int(row.stage.rsplit("-", 1)[1]) if benchmark else None
    order = DEFAULT_ORDER if component != 1 else PARTNERS[row.process]
    partner = PARTNERS[row.process] if component == 0 else DEFAULT_ORDER
    record = asdict(row)
    record.update(
        object_kind="auxiliary_ordered_benchmark_seed" if benchmark else "synthetic_fixed_role_expression",
        definition_version=DEFINITION_VERSION,
        reference_order=json.dumps(order), scalar_legs=json.dumps(PROCESS_SPECS[row.process].scalar_legs),
        graviton_legs=json.dumps(PROCESS_SPECS[row.process].graviton_legs),
        benchmark_family=row.process if benchmark else "",
        reconstruction_component=component if benchmark else "",
        reconstruction_partner_order=json.dumps(partner) if benchmark else "",
        reconstruction_coefficient=1 if benchmark else "",
        parent_channel="14|23" if benchmark and row.process == "4s1h" else "",
        paper_to_repository_factor=(2 if row.process == "4s1h" else 1) if benchmark else "",
    )
    return record


def _write_metadata(rows: Sequence[Candidate], path: Path) -> None:
    with _open_text(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_metadata(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(_metadata(row))


def output_paths(output_dir: Path) -> dict[str, Path]:
    directory = output_dir.expanduser().resolve()
    return {
        **{f"{split}_{kind}": directory / f"ordered_gravity_{split}_{kind}.csv.gz"
           for split in ("train", "test") for kind in ("raw", "tok", "metadata")},
        "manifest": directory / "ordered_gravity_manifest.json",
    }


def _check_destinations(paths: dict[str, Path], overwrite: bool) -> None:
    for path in paths.values():
        if path.is_symlink():
            raise ValueError(f"Output must not be a symbolic link: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"Output is not a regular file: {path}")
        if any(parent.exists() and not parent.is_dir() for parent in path.parents):
            raise ValueError(f"Output parent is not a directory: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists (use --overwrite): {path}")


def write_datasets(
    training: Sequence[Candidate], test: Sequence[Candidate], report: dict,
    output_dir: Path, *, overwrite: bool = False,
) -> dict[str, Path]:
    paths = output_paths(output_dir)
    _check_destinations(paths, overwrite)
    staged = {}
    try:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
            os.close(fd)
            staged[key] = Path(name)
        for split, rows in (("train", training), ("test", test)):
            write_raw(rows, staged[f"{split}_raw"])
            tokenise(rows, staged[f"{split}_tok"], max_tokens=report["max_tokens"])
            _write_metadata(rows, staged[f"{split}_metadata"])
        audit = audit_token_files(staged["train_tok"], staged["test_tok"], max_tokens=report["max_tokens"])
        if audit["training_rows"] != len(training) or audit["test_rows"] != len(test):
            raise RuntimeError("Serialized train/test counts do not match generation")
        manifest = dict(report, audit=audit, files={
            key: {"path": str(paths[key]), "sha256": _sha256(path),
                  "rows": len(training) if key.startswith("train_") else len(test)}
            for key, path in staged.items() if key != "manifest"
        })
        staged["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Reuse Yang--Mills' rollback-capable publication of related files.
        _publish_all(tuple(staged.values()), tuple(paths.values()), overwrite=overwrite)
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("N", nargs="?", type=int, default=5, help="Only five-point processes are defined")
    parser.add_argument("--samples", type=int, default=100_200, help="Total rows, INCLUDING the test set")
    parser.add_argument("--test-size", type=int, default=200, help="Total held-out component scrambles")
    parser.add_argument("--process", choices=(*PROCESS_SPECS, "mixed"), default="mixed")
    parser.add_argument("--kind", "--dataset-kind", choices=("oneshot", "staged", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", "--test-seed", type=int, default=240804720,
                        help="Seed for reserved benchmark component scrambles")
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-scr", type=int, default=1)
    parser.add_argument("--max-scr", type=int, default=5)
    parser.add_argument("--min-terms", type=int, default=1)
    parser.add_argument("--max-terms", type=int, default=3)
    parser.add_argument("--max-candidates-factor", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scrambles", nargs="+", default=None)
    parser.add_argument("--output-dir", "--split-output-dir", type=Path,
                        default=REPO_ROOT / "data/gravity/ordered")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.N != 5:
            raise ValueError("Only N=5 is defined for the ordered scalar--gravity seeds")
        paths = output_paths(args.output_dir)
        _check_destinations(paths, args.overwrite)
        print(f"Generating {args.samples - args.test_size:,} synthetic training rows and "
              f"{args.test_size:,} held-out ordered component rows...", flush=True)
        training, test, report = build_datasets(
            samples=args.samples, test_size=args.test_size, process=args.process, kind=args.kind,
            seed=args.seed, split_seed=args.split_seed, jobs=args.jobs, max_tokens=args.max_tokens,
            min_scr=args.min_scr, max_scr=args.max_scr, min_terms=args.min_terms, max_terms=args.max_terms,
            max_attempts_factor=args.max_candidates_factor, scramble_names=args.scrambles,
            batch_size=args.batch_size,
        )
        write_datasets(training, test, report, args.output_dir, overwrite=args.overwrite)
        print(f"Wrote {len(training):,} training and {len(test):,} test rows; expression overlap: 0.")
        print(f"Training tokens: {paths['train_tok']}")
        print(f"Test tokens: {paths['test_tok']}")
        print(f"Manifest: {paths['manifest']}")
    except (ArithmeticError, csv.Error, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
