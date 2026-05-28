#!/usr/bin/env python3
"""Answer-agnostic staged simplification data generator.

This entrypoint generates local simplification pairs rather than one-shot
``fully scrambled -> canonical`` pairs.  For each accepted base expression it
builds a scramble trajectory and emits adjacent reverse steps:

    trajectory[k] -> trajectory[k-1]

It also emits the final canonicalisation step ``expanded -> simple`` by
default, so a model trained on the resulting data can roll all the way back to
the compact gauge-invariant form.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gen_data as gd
import gen_data_v2 as gv2


DEFAULT_LENGTH_BUCKETS = gv2.DEFAULT_LENGTH_BUCKETS


@dataclass(frozen=True)
class StageCandidate:
    simple: str
    scrambled: str
    canonical_simple: str
    trajectory_id: str
    stage_kind: str
    stage_from: str
    stage_to: str
    stage_bucket: str
    applied_scramble: str
    cumulative_scrambles: tuple[str, ...]
    signature: tuple[str, ...]
    simple_len: int
    scrambled_len: int
    length_reduction_bucket: str
    term_count: str
    block_profile: str
    endpoint_profile: str
    denominator_profile: str
    coefficient_profile: str
    scramble_profile: str
    simple_length_bucket: str
    scrambled_length_bucket: str


@dataclass(frozen=True)
class StageBatchJob:
    n_particles: int
    target_count: int
    max_attempts: int
    seed: int
    min_scr: int
    max_scr: int
    unit_probability: float
    old_style_probability: float
    spurious_repeat_probability: float
    scalar_power_probability: float
    min_terms: int
    max_terms: int
    validate: bool
    mass: float
    full_expand_scrambled: bool
    max_tokens: int | None
    tokenizer_max_particles: int
    length_buckets: tuple[int, ...]
    validation_pol_modes: tuple[str, ...]
    scramble_names: tuple[str, ...]
    include_canonical_stage: bool
    batch_index: int


def length_reduction_bucket(delta: int) -> str:
    if delta <= 0:
        return "<=0"
    if delta <= 16:
        return "1-16"
    if delta <= 64:
        return "17-64"
    if delta <= 256:
        return "65-256"
    if delta <= 512:
        return "257-512"
    return ">512"


def tracked_scramble_trajectory(
    expr: str,
    n_gamma: int,
    n_particles: int,
    *,
    min_scr: int,
    max_scr: int,
    full_expand: bool,
    scramble_names: Sequence[str] | None,
    max_len: int = gd.DEFAULT_MAX_SCRAMBLED_LEN,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = gd.normalise_scramble_names(scramble_names)
    out = gd.full_expand_expression(expr) if full_expand else expr
    path = [out]
    applied: list[str] = []
    n_steps = random.randint(max(0, min_scr), max(max_scr, min_scr)) if max_scr > 0 and names else 0
    for _ in range(n_steps):
        name = random.choice(names)
        cand = gd._SCRAMBLER_BY_NAME[name](out, n_gamma, n_particles)
        if full_expand:
            cand = gd.full_expand_expression(cand)
        if cand != out and len(cand) <= max_len:
            out = cand
            path.append(out)
            applied.append(name)
    return tuple(path), tuple(applied)


def _token_len(tokenizer, expr: str) -> int:
    return len(tokenizer.encode_infix(expr))


def _make_stage_candidate(
    *,
    target_expr: str,
    source_expr: str,
    canonical_simple: str,
    trajectory_id: str,
    stage_kind: str,
    stage_from: str,
    stage_to: str,
    applied_scramble: str,
    cumulative_scrambles: tuple[str, ...],
    n_particles: int,
    tokenizer,
    max_tokens: int | None,
    length_buckets: Sequence[int],
) -> StageCandidate | None:
    target_expr = target_expr.replace("**", "^")
    source_expr = source_expr.replace("**", "^")
    try:
        simple_len = _token_len(tokenizer, target_expr)
        scrambled_len = _token_len(tokenizer, source_expr)
    except ValueError:
        return None
    if max_tokens is not None and (simple_len > max_tokens or scrambled_len > max_tokens):
        return None
    if simple_len >= scrambled_len:
        return None

    parts = gv2.expression_signature_parts(
        canonical_simple,
        source_expr,
        n_particles=n_particles,
        simple_len=simple_len,
        scrambled_len=scrambled_len,
        length_buckets=length_buckets,
        applied_scrambles=cumulative_scrambles,
    )
    reduction_bucket = length_reduction_bucket(scrambled_len - simple_len)
    stage_bucket = f"{stage_kind}:{stage_from}->{stage_to}"
    signature = (
        stage_bucket,
        applied_scramble,
        parts["term_count"],
        parts["block_profile"],
        parts["endpoint_profile"],
        parts["denominator_profile"],
        parts["coefficient_profile"],
        parts["scramble_profile"],
        parts["scrambled_length_bucket"],
        parts["simple_length_bucket"],
        reduction_bucket,
    )
    return StageCandidate(
        simple=target_expr,
        scrambled=source_expr,
        canonical_simple=canonical_simple,
        trajectory_id=trajectory_id,
        stage_kind=stage_kind,
        stage_from=stage_from,
        stage_to=stage_to,
        stage_bucket=stage_bucket,
        applied_scramble=applied_scramble,
        cumulative_scrambles=cumulative_scrambles,
        signature=signature,
        simple_len=simple_len,
        scrambled_len=scrambled_len,
        length_reduction_bucket=reduction_bucket,
        **parts,
    )


def generate_stage_candidates(
    n_particles: int,
    *,
    min_scr: int,
    max_scr: int,
    unit_probability: float,
    old_style_probability: float,
    denom_repeat_probability: float,
    scalar_power_probability: float,
    min_terms: int,
    max_terms: int,
    validate: bool,
    mass: float,
    full_expand_scrambled: bool,
    max_tokens: int | None,
    tokenizer,
    length_buckets: Sequence[int],
    validation_pol_modes: Sequence[str],
    scramble_names: Sequence[str] | None,
    include_canonical_stage: bool,
    trajectory_id: str,
) -> list[StageCandidate]:
    built = gd._build_base_expression(
        n_particles,
        unit_probability=unit_probability,
        old_style_probability=old_style_probability,
        denom_repeat_probability=denom_repeat_probability,
        scalar_power_probability=scalar_power_probability,
        use_denominators=gd.DEFAULT_USE_DENOMINATORS,
        min_terms=min_terms,
        max_terms=max_terms,
    )
    if built is None:
        return []
    canonical_simple, expanded = built
    expanded_start = gd.full_expand_expression(expanded) if full_expand_scrambled else expanded

    if validate:
        ok, _details = gd._validate_pair(
            canonical_simple,
            expanded_start,
            n_particles,
            mass,
            pol_modes=validation_pol_modes,
        )
        if not ok:
            return []

    trajectory, applied = tracked_scramble_trajectory(
        expanded,
        n_particles - 2,
        n_particles,
        min_scr=min_scr,
        max_scr=max_scr,
        full_expand=full_expand_scrambled,
        scramble_names=scramble_names,
    )
    if validate and trajectory:
        ok, _details = gd._validate_pair(
            expanded_start,
            trajectory[-1],
            n_particles,
            mass,
            pol_modes=validation_pol_modes,
        )
        if not ok:
            return []

    out: list[StageCandidate] = []
    if include_canonical_stage:
        candidate = _make_stage_candidate(
            target_expr=canonical_simple,
            source_expr=expanded_start,
            canonical_simple=canonical_simple,
            trajectory_id=trajectory_id,
            stage_kind="canonical",
            stage_from="expanded",
            stage_to="simple",
            applied_scramble="canonical",
            cumulative_scrambles=(),
            n_particles=n_particles,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            length_buckets=length_buckets,
        )
        if candidate is not None:
            if not validate or gd._validate_pair(
                candidate.scrambled,
                candidate.simple,
                n_particles,
                mass,
                pol_modes=validation_pol_modes,
            )[0]:
                out.append(candidate)

    for i in range(len(trajectory) - 1, 0, -1):
        source_expr = trajectory[i]
        target_expr = trajectory[i - 1]
        cumulative = applied[:i]
        step_label = applied[i - 1] if i - 1 < len(applied) else "unknown"
        candidate = _make_stage_candidate(
            target_expr=target_expr,
            source_expr=source_expr,
            canonical_simple=canonical_simple,
            trajectory_id=trajectory_id,
            stage_kind="scramble_step",
            stage_from=str(i),
            stage_to=str(i - 1),
            applied_scramble=step_label,
            cumulative_scrambles=cumulative,
            n_particles=n_particles,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            length_buckets=length_buckets,
        )
        if candidate is None:
            continue
        if validate:
            ok, _details = gd._validate_pair(
                candidate.scrambled,
                candidate.simple,
                n_particles,
                mass,
                pol_modes=validation_pol_modes,
            )
            if not ok:
                continue
        out.append(candidate)
    return out


def dedupe_candidates(candidates: list[StageCandidate]) -> tuple[list[StageCandidate], int]:
    seen: set[tuple[str, str]] = set()
    out: list[StageCandidate] = []
    for candidate in candidates:
        key = (candidate.simple, candidate.scrambled)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out, len(candidates) - len(out)


def balanced_select(candidates: list[StageCandidate], target: int) -> list[StageCandidate]:
    stage_groups: dict[str, list[StageCandidate]] = defaultdict(list)
    for candidate in candidates:
        stage_groups[candidate.stage_bucket].append(candidate)
    for stage_candidates in stage_groups.values():
        random.shuffle(stage_candidates)

    stage_keys = list(stage_groups)
    random.shuffle(stage_keys)
    selected: list[StageCandidate] = []
    per_stage_target = max(1, math.ceil(target / max(1, len(stage_keys))))
    leftovers: list[StageCandidate] = []

    for stage_key in stage_keys:
        by_signature: dict[tuple[str, ...], list[StageCandidate]] = defaultdict(list)
        for candidate in stage_groups[stage_key]:
            by_signature[candidate.signature].append(candidate)
        for bucket in by_signature.values():
            random.shuffle(bucket)
        sig_keys = list(by_signature)
        random.shuffle(sig_keys)
        stage_selected: list[StageCandidate] = []
        while len(stage_selected) < per_stage_target and sig_keys:
            next_keys: list[tuple[str, ...]] = []
            for sig_key in sig_keys:
                bucket = by_signature[sig_key]
                if bucket and len(stage_selected) < per_stage_target:
                    stage_selected.append(bucket.pop())
                if bucket:
                    next_keys.append(sig_key)
            sig_keys = next_keys
        selected.extend(stage_selected)
        for bucket in by_signature.values():
            leftovers.extend(bucket)

    if len(selected) < target:
        random.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    elif len(selected) > target:
        random.shuffle(selected)
        selected = selected[:target]
    return selected


def write_raw_csv(candidates: Sequence[StageCandidate], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["simple", "scrambled"])
        for candidate in candidates:
            writer.writerow([candidate.simple, candidate.scrambled])


def metadata_path_for(raw_out: str) -> str:
    path = Path(raw_out)
    return str(path.with_name(f"{path.stem}_stages.csv"))


def write_metadata_csv(candidates: Sequence[StageCandidate], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "trajectory_id",
        "stage_kind",
        "stage_from",
        "stage_to",
        "stage_bucket",
        "applied_scramble",
        "cumulative_scramble_depth",
        "cumulative_scrambles",
        "simple_len",
        "scrambled_len",
        "length_reduction",
        "length_reduction_bucket",
        "simple_length_bucket",
        "scrambled_length_bucket",
        "term_count",
        "block_profile",
        "endpoint_profile",
        "denominator_profile",
        "coefficient_profile",
        "scramble_profile",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, candidate in enumerate(candidates):
            writer.writerow(
                {
                    "row_id": row_id,
                    "trajectory_id": candidate.trajectory_id,
                    "stage_kind": candidate.stage_kind,
                    "stage_from": candidate.stage_from,
                    "stage_to": candidate.stage_to,
                    "stage_bucket": candidate.stage_bucket,
                    "applied_scramble": candidate.applied_scramble,
                    "cumulative_scramble_depth": len(candidate.cumulative_scrambles),
                    "cumulative_scrambles": ",".join(candidate.cumulative_scrambles),
                    "simple_len": candidate.simple_len,
                    "scrambled_len": candidate.scrambled_len,
                    "length_reduction": candidate.scrambled_len - candidate.simple_len,
                    "length_reduction_bucket": candidate.length_reduction_bucket,
                    "simple_length_bucket": candidate.simple_length_bucket,
                    "scrambled_length_bucket": candidate.scrambled_length_bucket,
                    "term_count": candidate.term_count,
                    "block_profile": candidate.block_profile,
                    "endpoint_profile": candidate.endpoint_profile,
                    "denominator_profile": candidate.denominator_profile,
                    "coefficient_profile": candidate.coefficient_profile,
                    "scramble_profile": candidate.scramble_profile,
                }
            )


def _counter_lines(title: str, counter: Counter[str]) -> list[str]:
    lines = [title]
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {key}: {count}")
    return lines


def write_coverage_report(
    path: str,
    *,
    selected: list[StageCandidate],
    candidates: list[StageCandidate],
    requested: int,
    attempts: int,
    max_attempts: int,
    removed_duplicates: int,
) -> None:
    stratum_counts = Counter(candidate.signature for candidate in selected)
    observed_stratum_counts = Counter(candidate.signature for candidate in candidates)
    observed_strata = set(observed_stratum_counts)
    selected_stage_counts = Counter(candidate.stage_bucket for candidate in selected)
    observed_stage_counts = Counter(candidate.stage_bucket for candidate in candidates)
    target_per_stage = max(1, math.ceil(requested / max(1, len(observed_stage_counts))))
    target_per_stratum = max(1, math.ceil(requested / max(1, len(observed_strata))))
    underfilled_stages = [
        (stage, selected_stage_counts.get(stage, 0), observed_stage_counts[stage])
        for stage in sorted(observed_stage_counts)
        if selected_stage_counts.get(stage, 0) < min(target_per_stage, observed_stage_counts[stage])
    ]
    underfilled_strata = [
        (signature, stratum_counts.get(signature, 0))
        for signature in sorted(observed_strata)
        if stratum_counts.get(signature, 0) < min(target_per_stratum, observed_stratum_counts[signature])
    ]

    scramble_counts: Counter[str] = Counter(candidate.applied_scramble for candidate in selected)
    simple_lengths = Counter(candidate.simple_length_bucket for candidate in selected)
    scrambled_lengths = Counter(candidate.scrambled_length_bucket for candidate in selected)
    reductions = Counter(candidate.length_reduction_bucket for candidate in selected)
    block_counts = Counter(candidate.block_profile for candidate in selected)
    endpoint_counts = Counter(candidate.endpoint_profile for candidate in selected)
    denominator_counts = Counter(candidate.denominator_profile for candidate in selected)

    lines = [
        "# gen_data_staged coverage report",
        f"requested={requested}",
        f"selected={len(selected)}",
        f"candidate_pool={len(candidates)}",
        f"attempts={attempts}",
        f"max_attempts={max_attempts}",
        f"removed_duplicates={removed_duplicates}",
        f"observed_stage_buckets={len(observed_stage_counts)}",
        f"observed_strata={len(observed_strata)}",
        f"selected_strata={len(stratum_counts)}",
        f"target_per_observed_stage={target_per_stage}",
        f"target_per_observed_stratum={target_per_stratum}",
        "",
        *_counter_lines("stage_buckets", selected_stage_counts),
        "",
        *_counter_lines("applied_scramble_families", scramble_counts),
        "",
        *_counter_lines("target_length_buckets", simple_lengths),
        "",
        *_counter_lines("source_length_buckets", scrambled_lengths),
        "",
        *_counter_lines("length_reduction_buckets", reductions),
        "",
        *_counter_lines("block_profiles", block_counts),
        "",
        *_counter_lines("endpoint_profiles", endpoint_counts),
        "",
        *_counter_lines("denominator_profiles", denominator_counts),
        "",
        "underfilled_stage_buckets",
    ]
    for stage, selected_count, observed_count in underfilled_stages:
        lines.append(f"  {stage}: selected={selected_count} observed={observed_count} target={target_per_stage}")
    lines.extend(["", "stratum_fill_counts"])
    for signature, count in sorted(stratum_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {count}: {' || '.join(signature)}")
    lines.extend(["", "underfilled_observed_strata"])
    for signature, count in underfilled_strata[:200]:
        lines.append(f"  {count}/{target_per_stratum}: {' || '.join(signature)}")
    if len(underfilled_strata) > 200:
        lines.append(f"  ... {len(underfilled_strata) - 200} more")
    lines.append("")
    lines.append("redistribution_summary")
    if len(selected) >= requested:
        lines.append("  selected target count; any underfilled strata were redistributed to available candidates")
    else:
        lines.append("  selected fewer than requested; candidate pool exhausted before target count")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worker_generate_candidates(job: StageBatchJob) -> tuple[list[StageCandidate], int]:
    random.seed(job.seed)
    tokenizer = gv2.make_tokenizer(job.tokenizer_max_particles, job.max_tokens)
    candidates: list[StageCandidate] = []
    attempts = 0
    while len(candidates) < job.target_count and attempts < job.max_attempts:
        attempts += 1
        trajectory_id = f"b{job.batch_index}_a{attempts}"
        candidates.extend(
            generate_stage_candidates(
                job.n_particles,
                min_scr=job.min_scr,
                max_scr=job.max_scr,
                unit_probability=job.unit_probability,
                old_style_probability=job.old_style_probability,
                denom_repeat_probability=job.spurious_repeat_probability,
                scalar_power_probability=job.scalar_power_probability,
                min_terms=job.min_terms,
                max_terms=job.max_terms,
                validate=job.validate,
                mass=job.mass,
                full_expand_scrambled=job.full_expand_scrambled,
                max_tokens=job.max_tokens,
                tokenizer=tokenizer,
                length_buckets=job.length_buckets,
                validation_pol_modes=job.validation_pol_modes,
                scramble_names=job.scramble_names,
                include_canonical_stage=job.include_canonical_stage,
                trajectory_id=trajectory_id,
            )
        )
    return candidates, attempts


def generate_balanced(args) -> tuple[list[StageCandidate], list[StageCandidate], int, int]:
    length_buckets = gv2.parse_length_buckets(args.length_buckets)
    max_tokens = None if args.max_tokens <= 0 else args.max_tokens
    validation_pol_modes = tuple(args.validation_pol_modes)
    scramble_names = gd.normalise_scramble_names(args.scrambles)
    target_pool = max(args.samples, int(math.ceil(args.samples * args.coverage_oversample)))
    max_attempts = max(1, int(args.samples * args.max_attempts_factor * max(1.0, args.coverage_oversample)))
    full_expand_scrambled = not args.grouped_scrambled if gd.DEFAULT_FULL_EXPAND_SCRAMBLED else False
    jobs = gd._resolve_jobs(args.jobs)
    batch_size = max(1, int(args.batch_size))
    batch_counts = gd._batch_sizes(target_pool, batch_size)
    base_seed = args.seed if args.seed is not None else random.randrange(1, 2**31 - 1)

    batch_jobs: list[StageBatchJob] = []
    for index, count in enumerate(batch_counts):
        batch_attempts = max(count, int(math.ceil(max_attempts * count / max(1, target_pool))))
        batch_jobs.append(
            StageBatchJob(
                n_particles=args.N,
                target_count=count,
                max_attempts=batch_attempts,
                seed=base_seed + 1000003 * index,
                min_scr=args.min_scr,
                max_scr=args.max_scr,
                unit_probability=args.unit_probability,
                old_style_probability=args.old_style_probability,
                spurious_repeat_probability=args.spurious_repeat_probability,
                scalar_power_probability=args.scalar_power_probability,
                min_terms=args.min_terms,
                max_terms=args.max_terms,
                validate=not args.no_validate,
                mass=args.mass,
                full_expand_scrambled=full_expand_scrambled,
                max_tokens=max_tokens,
                tokenizer_max_particles=args.tokenizer_max_particles,
                length_buckets=length_buckets,
                validation_pol_modes=validation_pol_modes,
                scramble_names=scramble_names,
                include_canonical_stage=args.include_canonical_stage,
                batch_index=index,
            )
        )

    candidates: list[StageCandidate] = []
    attempts = 0
    if jobs == 1 or len(batch_jobs) <= 1:
        iterator = (_worker_generate_candidates(job) for job in batch_jobs)
        for batch_candidates, batch_attempts in gd._progress(
            iterator,
            total=len(batch_jobs),
            enabled=not args.no_progress,
            desc="generating-staged",
        ):
            candidates.extend(batch_candidates)
            attempts += batch_attempts
    else:
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with context.Pool(processes=min(jobs, len(batch_jobs))) as pool:
            iterator = pool.imap_unordered(_worker_generate_candidates, batch_jobs)
            for batch_candidates, batch_attempts in gd._progress(
                iterator,
                total=len(batch_jobs),
                enabled=not args.no_progress,
                desc="generating-staged",
            ):
                candidates.extend(batch_candidates)
                attempts += batch_attempts

    candidates, removed = dedupe_candidates(candidates)
    if args.coverage_mode == "random":
        random.shuffle(candidates)
        selected = candidates[: args.samples]
    else:
        selected = balanced_select(candidates, args.samples)
    return selected, candidates, attempts, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate answer-agnostic staged simplification data.")
    parser.add_argument("N", nargs="?", type=int, default=gd.DEFAULT_N_PARTICLES)
    parser.add_argument("--samples", type=int, default=gd.DEFAULT_SAMPLES)
    parser.add_argument("--coverage-mode", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--coverage-oversample", type=float, default=2.0)
    parser.add_argument("--length-buckets", default=DEFAULT_LENGTH_BUCKETS)
    parser.add_argument("--max-attempts-factor", type=int, default=gd.DEFAULT_MAX_ATTEMPTS_FACTOR)
    parser.add_argument("--max-scr", type=int, default=gd.DEFAULT_MAX_SCR)
    parser.add_argument("--min-scr", type=int, default=gd.DEFAULT_MIN_SCR)
    parser.add_argument("--min-terms", type=int, default=gd.DEFAULT_MIN_TERMS)
    parser.add_argument("--max-terms", type=int, default=gd.DEFAULT_MAX_TERMS)
    parser.add_argument("--seed", type=int, default=gd.DEFAULT_SEED)
    parser.add_argument("--unit-probability", type=float, default=0.8)
    parser.add_argument("--nonunit-probability", type=float, default=None)
    parser.add_argument("--old-style-probability", type=float, default=0.0)
    parser.add_argument("--spurious-repeat-probability", type=float, default=gd.DENOM_REPEAT_PROBABILITY)
    parser.add_argument("--scalar-power-probability", type=float, default=gd.SCALAR_POWER_PROBABILITY)
    parser.add_argument("--mass", type=float, default=gd.DEFAULT_MASS)
    parser.add_argument("--raw-out", type=str, default=None)
    parser.add_argument("--tok-out", type=str, default=None)
    parser.add_argument("--metadata-out", type=str, default=None)
    parser.add_argument("--log-out", type=str, default=None)
    parser.add_argument("--coverage-report-out", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=gd.DEFAULT_MAX_TOKENS)
    parser.add_argument("--tokenizer-max-particles", type=int, default=gd.DEFAULT_TOKENIZER_MAX_PARTICLES)
    parser.add_argument(
        "--validation-pol-modes",
        nargs="+",
        choices=["coulomb", "covariant"],
        default=list(gd.DEFAULT_VALIDATION_POL_MODES),
    )
    parser.add_argument(
        "--scrambles",
        nargs="*",
        default=None,
        choices=["all", "none", *list(gd._SCRAMBLER_BY_NAME)],
    )
    parser.add_argument("--include-canonical-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-tokenise", action="store_true")
    parser.add_argument("--grouped-scrambled", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--jobs", type=str, default="1", help="Number of worker processes, or 'auto'.")
    parser.add_argument("--batch-size", type=int, default=gd.DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.nonunit_probability is not None:
        args.unit_probability = max(0.0, min(1.0, 1.0 - args.nonunit_probability))

    nsamps = args.samples // 1000
    raw_out = args.raw_out or f"data/gi_{args.N}pt_staged_{nsamps}k.csv"
    tok_out = args.tok_out or f"data/gi_{args.N}pt_staged_{nsamps}k_tok.csv"
    metadata_out = args.metadata_out or metadata_path_for(raw_out)
    log_out = args.log_out or f"gen_data_{args.N}pt_staged_{nsamps}k.log"
    coverage_report = args.coverage_report_out or str(Path(log_out).with_suffix(".coverage.txt"))

    t0 = time.perf_counter()
    selected, candidates, attempts, removed = generate_balanced(args)
    t1 = time.perf_counter()

    write_raw_csv(selected, raw_out)
    write_metadata_csv(selected, metadata_out)
    if gd.DEFAULT_TOKENISE and not args.no_tokenise:
        max_tokens = None if args.max_tokens <= 0 else args.max_tokens
        gd.tokenise_csv(
            raw_out,
            tok_out,
            max_particles=args.tokenizer_max_particles,
            max_sequence_length=max_tokens,
        )
    t2 = time.perf_counter()

    max_attempts = max(1, int(args.samples * args.max_attempts_factor * max(1.0, args.coverage_oversample)))
    Path(log_out).write_text(
        "\n".join(
            [
                f"# gen_data_staged log N={args.N} requested={args.samples} selected={len(selected)}",
                f"coverage_mode={args.coverage_mode}",
                f"coverage_oversample={args.coverage_oversample}",
                f"jobs={args.jobs}",
                f"batch_size={args.batch_size}",
                f"terms=[{args.min_terms},{args.max_terms}] scr=[{args.min_scr},{args.max_scr}]",
                f"include_canonical_stage={args.include_canonical_stage}",
                f"unit_probability={args.unit_probability}",
                f"old_style_probability={args.old_style_probability}",
                f"spurious_repeat_probability={args.spurious_repeat_probability}",
                f"scalar_power_probability={args.scalar_power_probability}",
                f"max_tokens={args.max_tokens}",
                f"length_buckets={args.length_buckets}",
                f"scrambles={','.join(gd.normalise_scramble_names(args.scrambles))}",
                f"candidate_pool={len(candidates)} attempts={attempts} removed_duplicates={removed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_coverage_report(
        coverage_report,
        selected=selected,
        candidates=candidates,
        requested=args.samples,
        attempts=attempts,
        max_attempts=max_attempts,
        removed_duplicates=removed,
    )

    print(f"{len(selected)} staged pairs -> {raw_out}")
    print(f"  metadata   : {metadata_out}")
    if gd.DEFAULT_TOKENISE and not args.no_tokenise:
        print(f"  tokenized  : {tok_out}")
    print(f"  generation : {t1 - t0:.2f}s")
    print(f"  write/tok  : {t2 - t1:.2f}s")
    print(f"  log        : {log_out}")
    print(f"  coverage   : {coverage_report}")
    if len(selected) < args.samples:
        print(f"WARNING: selected {len(selected)} / requested {args.samples}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
