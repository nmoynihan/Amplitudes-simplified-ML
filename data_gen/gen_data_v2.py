#!/usr/bin/env python3
"""Answer-agnostic balanced data generator.

This v2 entrypoint reuses the expression construction, scrambling,
validation, simplification, CSV writing, and tokenisation utilities from
``gen_data.py``.  Its main difference is selection: it generates valid
candidates, assigns each candidate an answer-agnostic structural signature,
then selects examples by round-robin across observed signatures.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gen_data as gd


DEFAULT_LENGTH_BUCKETS = "512,768,1024,1536,2048"


@dataclass(frozen=True)
class Candidate:
    simple: str
    scrambled: str
    signature: tuple[str, ...]
    simple_len: int
    scrambled_len: int
    term_count: str
    block_profile: str
    endpoint_profile: str
    denominator_profile: str
    coefficient_profile: str
    scramble_profile: str
    simple_length_bucket: str
    scrambled_length_bucket: str
    applied_scrambles: tuple[str, ...]


@dataclass(frozen=True)
class CandidateBatchJob:
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


def parse_length_buckets(raw: str) -> tuple[int, ...]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or values[0] <= 0:
        raise ValueError("length buckets must be positive comma-separated integers")
    return tuple(values)


def length_bucket(value: int, buckets: Sequence[int]) -> str:
    for limit in buckets:
        if value <= limit:
            return f"<={limit}"
    return f">{buckets[-1]}"


def strip_coeff(body: str) -> tuple[str, bool]:
    body = body.strip()
    match = re.match(r"^(\d+)\*(.+)$", body)
    if not match:
        return body, True
    coeff = int(match.group(1))
    return match.group(2).strip(), coeff == 1


def endpoint_kind(idx: int, n_particles: int) -> str:
    return "scalar" if idx in gd.scalar_legs(n_particles) else "photon"


def endpoint_category(left: int, right: int, n_particles: int) -> str:
    left_kind = endpoint_kind(left, n_particles)
    right_kind = endpoint_kind(right, n_particles)
    if left_kind == right_kind == "scalar":
        return "scalar-scalar-same" if left == right else "scalar-scalar-mixed"
    if left_kind == right_kind == "photon":
        return "photon-photon-same" if left == right else "photon-photon-mixed"
    return f"{left_kind}-{right_kind}"


def split_terms(expr: str) -> list[tuple[str, str]]:
    return gd._split_signed_terms(expr)


def denominator_factors(term_body: str) -> list[str]:
    if "/" not in term_body:
        return []
    _num, den = term_body.split("/", 1)
    den = gd._strip_matched_outer_parens(den)
    factors: list[str] = []
    for part in gd._split_top_level(den, "*"):
        part = part.strip()
        if not part:
            continue
        power_match = re.fullmatch(r"\((p_\d+\s*·\s*p_\d+)\)\^(\d+)", part)
        if power_match:
            factors.extend([gd._canon_pp(power_match.group(1))] * int(power_match.group(2)))
            continue
        factors.append(gd._canon_pp(part) if gd._RE_pp.fullmatch(part) else part)
    return factors


def numerator_factors(term_body: str) -> list[str]:
    if "/" in term_body:
        num, _den = term_body.split("/", 1)
    else:
        num = term_body
    num = gd._strip_matched_outer_parens(num)
    return [gd._strip_matched_outer_parens(part.strip())
            for part in gd._split_top_level(num, "*") if part.strip()]


def factor_profiles(term_body: str, n_particles: int) -> tuple[list[str], list[str], bool]:
    block_types: list[str] = []
    endpoint_types: list[str] = []
    has_scalar_power = False

    for factor in numerator_factors(term_body):
        power_match = re.fullmatch(r"\((.+)\)\^(\d+)", factor)
        if power_match:
            has_scalar_power = True
            factor = power_match.group(1)

        chain = gd._RE_pFchainp.fullmatch(factor)
        if chain:
            left = int(chain.group(1))
            right = int(chain.group(3))
            photons = re.findall(r"F_\d+", chain.group(2))
            if len(photons) == 1:
                block_types.append("singleF")
            elif len(photons) == 2:
                block_types.append("doubleF")
            elif len(photons) == 3:
                block_types.append("tripleF")
            else:
                block_types.append(f"chain{len(photons)}F")
            endpoint_types.append(endpoint_category(left, right, n_particles))
            continue

        trace = gd._RE_TrN.fullmatch(factor)
        if trace:
            trace_photons = re.findall(r"F_\d+", trace.group(1))
            block_types.append(f"tr{len(trace_photons)}")
            continue

        if gd._RE_pp.fullmatch(factor):
            continue

        block_types.append("other")

    return block_types, endpoint_types, has_scalar_power


def expression_signature_parts(
    simple: str,
    scrambled: str,
    *,
    n_particles: int,
    simple_len: int,
    scrambled_len: int,
    length_buckets: Sequence[int],
    applied_scrambles: Sequence[str],
) -> dict[str, str]:
    terms = split_terms(simple)
    term_profiles: list[str] = []
    endpoint_counter: Counter[str] = Counter()
    denominator_counts: list[int] = []
    repeated_denominators = False
    scalar_power = False
    supported_poles = True
    all_unit_coeffs = True

    for _sign, raw_body in terms:
        body, is_unit = strip_coeff(raw_body)
        all_unit_coeffs = all_unit_coeffs and is_unit
        block_types, endpoint_types, term_scalar_power = factor_profiles(body, n_particles)
        scalar_power = scalar_power or term_scalar_power
        endpoint_counter.update(endpoint_types)
        term_profiles.append("+".join(sorted(block_types)) if block_types else "empty")

        den = denominator_factors(body)
        denominator_counts.append(len(den))
        den_counts = Counter(den)
        repeated_denominators = repeated_denominators or any(count > 1 for count in den_counts.values())
        supported_poles = supported_poles and gd._has_supported_physical_poles(body)

    block_counter = Counter(term_profiles)
    block_profile = "|".join(f"{key}x{block_counter[key]}" for key in sorted(block_counter))
    endpoint_profile = "|".join(f"{key}x{endpoint_counter[key]}" for key in sorted(endpoint_counter)) or "none"
    denominator_profile = (
        f"counts={'+'.join(map(str, sorted(denominator_counts)))};"
        f"repeat={'yes' if repeated_denominators else 'no'};"
        f"scalar_power={'yes' if scalar_power else 'no'};"
        f"support={'full' if supported_poles else 'partial'}"
    )
    scramble_counts = Counter(applied_scrambles)
    scramble_labels = "+".join(f"{name}x{scramble_counts[name]}" for name in sorted(scramble_counts)) or "none"
    scramble_depth = len(applied_scrambles)
    if scramble_depth <= 1:
        depth_bucket = str(scramble_depth)
    elif scramble_depth <= 3:
        depth_bucket = "2-3"
    else:
        depth_bucket = "4+"

    return {
        "term_count": str(len(terms)),
        "block_profile": block_profile,
        "endpoint_profile": endpoint_profile,
        "denominator_profile": denominator_profile,
        "coefficient_profile": "unit" if all_unit_coeffs else "nonunit",
        "scramble_profile": f"depth={depth_bucket};labels={scramble_labels}",
        "simple_length_bucket": length_bucket(simple_len, length_buckets),
        "scrambled_length_bucket": length_bucket(scrambled_len, length_buckets),
    }


def tracked_scramble(
    expr: str,
    n_gamma: int,
    n_particles: int,
    *,
    min_scr: int,
    max_scr: int,
    full_expand: bool,
    scramble_names: Sequence[str] | None,
) -> tuple[str, tuple[str, ...]]:
    names = gd.normalise_scramble_names(scramble_names)
    out = gd.full_expand_expression(expr) if full_expand else expr
    n_steps = random.randint(max(0, min_scr), max(max_scr, min_scr)) if max_scr > 0 and names else 0
    applied: list[str] = []
    for _ in range(n_steps):
        name = random.choice(names)
        cand = gd._SCRAMBLER_BY_NAME[name](out, n_gamma, n_particles)
        if full_expand:
            cand = gd.full_expand_expression(cand)
        if len(cand) <= gd.DEFAULT_MAX_SCRAMBLED_LEN:
            if cand != out:
                applied.append(name)
            out = cand
    return (gd.full_expand_expression(out) if full_expand else out), tuple(applied)


def make_tokenizer(max_particles: int, max_tokens: int | None):
    from Tokenizer import ScatteringAmplitudeTokenizer

    return ScatteringAmplitudeTokenizer(
        max_particles=max_particles,
        max_sequence_length=max_tokens,
    )


def generate_candidate(
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
) -> Candidate | None:
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
        return None
    simple, expanded = built

    if validate:
        ok, _details = gd._validate_pair(simple, expanded, n_particles, mass, pol_modes=validation_pol_modes)
        if not ok:
            return None

    scrambled, applied_scrambles = tracked_scramble(
        expanded,
        n_particles - 2,
        n_particles,
        min_scr=min_scr,
        max_scr=max_scr,
        full_expand=full_expand_scrambled,
        scramble_names=scramble_names,
    )

    if validate:
        ok, _details = gd._validate_pair(expanded, scrambled, n_particles, mass, pol_modes=validation_pol_modes)
        if not ok:
            return None

    try:
        simplified = gd.simplify_to_lowest_terms(scrambled)
        if simplified.strip() == "0":
            raise ValueError("simplified scrambled expression is bare 0")
    except Exception:
        simplified = scrambled.replace("**", "^")

    if validate:
        ok, _details = gd._validate_pair(simple, simplified, n_particles, mass, pol_modes=validation_pol_modes)
        if not ok:
            return None

    try:
        simple_len = len(tokenizer.encode_infix(simple))
        scrambled_len = len(tokenizer.encode_infix(simplified))
    except ValueError:
        return None

    if max_tokens is not None and (simple_len > max_tokens or scrambled_len > max_tokens):
        return None

    parts = expression_signature_parts(
        simple,
        simplified,
        n_particles=n_particles,
        simple_len=simple_len,
        scrambled_len=scrambled_len,
        length_buckets=length_buckets,
        applied_scrambles=applied_scrambles,
    )
    signature = (
        parts["term_count"],
        parts["block_profile"],
        parts["endpoint_profile"],
        parts["denominator_profile"],
        parts["coefficient_profile"],
        parts["scramble_profile"],
        parts["scrambled_length_bucket"],
    )
    return Candidate(
        simple=simple,
        scrambled=simplified,
        signature=signature,
        simple_len=simple_len,
        scrambled_len=scrambled_len,
        applied_scrambles=applied_scrambles,
        **parts,
    )


def balanced_select(candidates: list[Candidate], target: int) -> list[Candidate]:
    groups: dict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.signature].append(candidate)
    for bucket in groups.values():
        random.shuffle(bucket)

    selected: list[Candidate] = []
    keys = list(groups)
    random.shuffle(keys)
    while len(selected) < target and keys:
        next_keys: list[tuple[str, ...]] = []
        for key in keys:
            bucket = groups[key]
            if bucket and len(selected) < target:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def dedupe_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for candidate in candidates:
        key = (candidate.simple, candidate.scrambled)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out, len(candidates) - len(out)


def write_coverage_report(
    path: str,
    *,
    selected: list[Candidate],
    candidates: list[Candidate],
    requested: int,
    attempts: int,
    max_attempts: int,
    removed_duplicates: int,
) -> None:
    stratum_counts = Counter(candidate.signature for candidate in selected)
    observed_strata = {candidate.signature for candidate in candidates}
    target_per_stratum = max(1, math.ceil(requested / max(1, len(observed_strata))))
    underfilled = [
        (signature, stratum_counts.get(signature, 0))
        for signature in sorted(observed_strata)
        if stratum_counts.get(signature, 0) < target_per_stratum
    ]

    def counter_lines(title: str, counter: Counter[str]) -> list[str]:
        lines = [title]
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  {key}: {count}")
        return lines

    simple_lengths = Counter(candidate.simple_length_bucket for candidate in selected)
    scrambled_lengths = Counter(candidate.scrambled_length_bucket for candidate in selected)
    scramble_counts: Counter[str] = Counter()
    for candidate in selected:
        scramble_counts.update(candidate.applied_scrambles or ("none",))
    block_counts = Counter(candidate.block_profile for candidate in selected)

    lines = [
        "# gen_data_v2 coverage report",
        f"requested={requested}",
        f"selected={len(selected)}",
        f"candidate_pool={len(candidates)}",
        f"attempts={attempts}",
        f"max_attempts={max_attempts}",
        f"removed_duplicates={removed_duplicates}",
        f"observed_strata={len(observed_strata)}",
        f"selected_strata={len(stratum_counts)}",
        f"target_per_observed_stratum={target_per_stratum}",
        "",
        *counter_lines("simple_length_buckets", simple_lengths),
        "",
        *counter_lines("scrambled_length_buckets", scrambled_lengths),
        "",
        *counter_lines("scramble_families", scramble_counts),
        "",
        *counter_lines("block_profiles", block_counts),
        "",
        "stratum_fill_counts",
    ]
    for signature, count in sorted(stratum_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {count}: {' || '.join(signature)}")
    lines.extend(["", "underfilled_observed_strata"])
    for signature, count in underfilled[:200]:
        lines.append(f"  {count}/{target_per_stratum}: {' || '.join(signature)}")
    if len(underfilled) > 200:
        lines.append(f"  ... {len(underfilled) - 200} more")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidate_csv(candidates: Sequence[Candidate], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["simple", "scrambled"])
        for candidate in candidates:
            writer.writerow([candidate.simple, candidate.scrambled])


def generate_balanced(args) -> tuple[list[Candidate], list[Candidate], int, int]:
    length_buckets = parse_length_buckets(args.length_buckets)
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

    batch_jobs: list[CandidateBatchJob] = []
    for index, count in enumerate(batch_counts):
        batch_attempts = max(
            count,
            int(math.ceil(max_attempts * count / max(1, target_pool))),
        )
        batch_jobs.append(
            CandidateBatchJob(
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
            )
        )

    candidates: list[Candidate] = []
    attempts = 0
    if jobs == 1 or len(batch_jobs) <= 1:
        iterator = (_worker_generate_candidates(job) for job in batch_jobs)
        for batch_candidates, batch_attempts in gd._progress(
            iterator,
            total=len(batch_jobs),
            enabled=not args.no_progress,
            desc="generating-v2",
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
                desc="generating-v2",
            ):
                candidates.extend(batch_candidates)
                attempts += batch_attempts

    candidates, removed = dedupe_candidates(candidates)
    if args.coverage_mode == "random":
        random.shuffle(candidates)
        selected = candidates[: args.samples]
    else:
        selected = balanced_select(candidates, args.samples)
        if len(selected) < args.samples:
            selected_keys = {(item.simple, item.scrambled) for item in selected}
            leftovers = [
                item for item in candidates
                if (item.simple, item.scrambled) not in selected_keys
            ]
            random.shuffle(leftovers)
            selected.extend(leftovers[: args.samples - len(selected)])
    return selected, candidates, attempts, removed


def _worker_generate_candidates(job: CandidateBatchJob) -> tuple[list[Candidate], int]:
    random.seed(job.seed)
    tokenizer = make_tokenizer(job.tokenizer_max_particles, job.max_tokens)
    candidates: list[Candidate] = []
    attempts = 0
    while len(candidates) < job.target_count and attempts < job.max_attempts:
        attempts += 1
        candidate = generate_candidate(
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
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates, attempts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate answer-agnostic balanced scalar-QED-like data.")
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
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-tokenise", action="store_true")
    parser.add_argument("--grouped-scrambled", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Accepted for CLI compatibility; v2 has no progress bar.")
    parser.add_argument("--jobs", type=str, default="1", help="Number of worker processes, or 'auto'.")
    parser.add_argument("--batch-size", type=int, default=gd.DEFAULT_BATCH_SIZE, help="Target accepted candidates per worker batch.")
    args = parser.parse_args()

    if args.nonunit_probability is not None:
        args.unit_probability = max(0.0, min(1.0, 1.0 - args.nonunit_probability))

    nsamps = args.samples // 1000
    raw_out = args.raw_out or f"data/gi_{args.N}pt_v2_{nsamps}k.csv"
    tok_out = args.tok_out or f"data/gi_{args.N}pt_v2_{nsamps}k_tok.csv"
    log_out = args.log_out or f"gen_data_{args.N}pt_v2_{nsamps}k.log"
    coverage_report = args.coverage_report_out or str(Path(log_out).with_suffix(".coverage.txt"))

    t0 = time.perf_counter()
    selected, candidates, attempts, removed = generate_balanced(args)
    t1 = time.perf_counter()

    Path(raw_out).parent.mkdir(parents=True, exist_ok=True)
    write_candidate_csv(selected, raw_out)
    if gd.DEFAULT_TOKENISE and not args.no_tokenise:
        max_tokens = None if args.max_tokens <= 0 else args.max_tokens
        gd.tokenise_csv(
            raw_out,
            tok_out,
            max_particles=args.tokenizer_max_particles,
            max_sequence_length=max_tokens,
        )
    t2 = time.perf_counter()

    Path(log_out).write_text(
        "\n".join(
            [
                f"# gen_data_v2 log N={args.N} requested={args.samples} selected={len(selected)}",
                f"coverage_mode={args.coverage_mode}",
                f"coverage_oversample={args.coverage_oversample}",
                f"jobs={args.jobs}",
                f"batch_size={args.batch_size}",
                f"terms=[{args.min_terms},{args.max_terms}] scr=[{args.min_scr},{args.max_scr}]",
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
        max_attempts=max(1, int(args.samples * args.max_attempts_factor * max(1.0, args.coverage_oversample))),
        removed_duplicates=removed,
    )

    print(f"{len(selected)} pairs -> {raw_out}")
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
