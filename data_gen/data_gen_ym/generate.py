"""generate — extracted from gen_data.py (scaffold, verbatim)."""

from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from notation import *
from algebra import *
from kinematics import generate_kinematics, mdot
from numerics import *
from scramble import *
from expr_model import *

DEFAULT_N_PARTICLES = 4


DEFAULT_SAMPLES = 5000


DEFAULT_SEED = 42


DEFAULT_MASS = 2.0


NSAMPS = DEFAULT_SAMPLES // 1000


DEFAULT_RAW_OUT_TEMPLATE = "gi_{N}pt_{NSAMPS}k.csv"


DEFAULT_TOK_OUT_TEMPLATE = "gi_{N}pt_tok_{NSAMPS}k.csv"


DEFAULT_LOG_OUT_TEMPLATE = "gen_data_{N}pt_{NSAMPS}k.log"


DEFAULT_VALIDATE = True


DEFAULT_TOKENISE = True


DEFAULT_FULL_EXPAND_SCRAMBLED = True


DEFAULT_OVERSAMPLE_FACTOR = 1.2


DEFAULT_DATASET_KIND = "oneshot"


DEFAULT_MAX_TOKENS = 2048


DEFAULT_BATCH_SIZE = 1000


DEFAULT_JOBS = "auto"  # integer as string, or "auto"


DEFAULT_PROGRESS = True


DEFAULT_TOKENIZER_MAX_PARTICLES = 8


@dataclass(frozen=True)
class BatchJob:
    dataset_kind: str
    N: int
    num_samples: int
    max_scr: int
    min_scr: int
    seed: int | None
    unit_probability: float
    old_style_probability: float
    denom_repeat_probability: float
    scalar_power_probability: float
    use_denominators: bool
    validate: bool
    M: float
    min_terms: int
    max_terms: int
    max_attempts_factor: int
    full_expand_scrambled: bool
    max_tokens: int | None
    tokenizer_max_particles: int
    validation_pol_modes: tuple[str, ...]
    scramble_names: tuple[str, ...] | None


def _make_tokenizer(tokenizer_max_particles: int, max_tokens: int | None):
    if max_tokens is None:
        return None
    from Tokenizer import ScatteringAmplitudeTokenizer

    return ScatteringAmplitudeTokenizer(
        max_particles=tokenizer_max_particles,
        max_sequence_length=max_tokens,
    )


def _within_token_budget(
    simple_expr: str,
    scrambled_expr: str,
    tokenizer,
    max_tokens: int | None,
) -> bool:
    if tokenizer is None or max_tokens is None:
        return True
    return (
        len(tokenizer.encode_infix(simple_expr)) <= max_tokens
        and len(tokenizer.encode_infix(scrambled_expr)) <= max_tokens
    )


def build_dataset(
    N: int,
    num_samples: int,
    *,
    max_scr: int = DEFAULT_MAX_SCR,
    min_scr: int = DEFAULT_MIN_SCR,
    seed: int | None = None,
    unit_probability: float = UNIT_PROBABILITY,
    old_style_probability: float = OLD_STYLE_PROBABILITY,
    denom_repeat_probability: float = DENOM_REPEAT_PROBABILITY,
    scalar_power_probability: float = SCALAR_POWER_PROBABILITY,
    use_denominators: bool = DEFAULT_USE_DENOMINATORS,
    validate: bool = DEFAULT_VALIDATE,
    M: float = DEFAULT_MASS,
    min_terms: int = DEFAULT_MIN_TERMS,
    max_terms: int = DEFAULT_MAX_TERMS,
    log_path: str | None = None,
    max_attempts_factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR,
    full_expand_scrambled: bool = DEFAULT_FULL_EXPAND_SCRAMBLED,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    tokenizer_max_particles: int = DEFAULT_TOKENIZER_MAX_PARTICLES,
    validation_pol_modes: Sequence[str] = DEFAULT_VALIDATION_POL_MODES,
    scramble_names: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Build a dataset of (simple, scrambled) pairs."""
    min_terms = max(1, int(min_terms))
    max_terms = max(min_terms, int(max_terms))
    if seed is not None:
        random.seed(seed)

    data: list[tuple[str, str]] = []
    stats = {
        "attempts": 0,
        "parity_fail": 0,
        "scramble_fail": 0,
        "simplify_fail": 0,
        "pole_fail": 0,
        "dimension_fail": 0,
        "token_fail": 0,
    }
    max_attempts = max(1, num_samples * max_attempts_factor)
    tokenizer = _make_tokenizer(tokenizer_max_particles, max_tokens)
    scramble_names = normalise_scramble_names(scramble_names)

    if log_path:
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(
                f"# gen_data log N={N} target={num_samples} "
                f"terms=[{min_terms},{max_terms}] scr=[{min_scr},{max_scr}] "
                f"unit_probability={unit_probability} "
                f"old_style_probability={old_style_probability} "
                f"spurious_repeat_probability={denom_repeat_probability} "
                f"scalar_power_probability={scalar_power_probability} "
                f"full_expand_scrambled={full_expand_scrambled} seed={seed} "
                f"max_tokens={max_tokens} pol_modes={','.join(validation_pol_modes)} "
                f"scrambles={','.join(scramble_names) if scramble_names else 'none'}\n"
            )

    while len(data) < num_samples and stats["attempts"] < max_attempts:
        stats["attempts"] += 1
        built = _build_base_expression(
            N,
            unit_probability=unit_probability,
            old_style_probability=old_style_probability,
            denom_repeat_probability=denom_repeat_probability,
            scalar_power_probability=scalar_power_probability,
            use_denominators=use_denominators,
            min_terms=min_terms,
            max_terms=max_terms,
        )
        if built is None:
            stats["dimension_fail"] += 1
            continue
        simple_expr, expanded_expr = built

        if validate:
            ok, _ = _validate_pair(simple_expr, expanded_expr, N, M, pol_modes=validation_pol_modes)
            if not ok:
                stats["parity_fail"] += 1
                continue

        scrambled = scramble(
            expanded_expr,
            N,
            min_scr=min_scr,
            max_scr=max_scr,
            full_expand=full_expand_scrambled,
            scramble_names=scramble_names,
        )

        if validate:
            ok, _ = _validate_pair(expanded_expr, scrambled, N, M, pol_modes=validation_pol_modes)
            if not ok:
                stats["scramble_fail"] += 1
                continue

        try:
            simplified_scrambled = simplify_to_lowest_terms(scrambled)
            if simplified_scrambled.strip() == "0":
                raise ValueError("simplified scrambled expression is bare 0")
        except Exception:
            stats["simplify_fail"] += 1
            simplified_scrambled = scrambled.replace("**", "^")

        if validate:
            ok, _ = _validate_pair(simple_expr, simplified_scrambled, N, M, pol_modes=validation_pol_modes)
            if not ok:
                stats["simplify_fail"] += 1
                continue

        try:
            if not _within_token_budget(simple_expr, simplified_scrambled, tokenizer, max_tokens):
                stats["token_fail"] += 1
                continue
        except ValueError:
            stats["token_fail"] += 1
            continue

        data.append((simple_expr, simplified_scrambled))

    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(
                "# SUMMARY "
                f"accepted={len(data)} parity_fail={stats['parity_fail']} "
                f"scramble_fail={stats['scramble_fail']} pole_fail={stats['pole_fail']} "
                f"simplify_fail={stats['simplify_fail']} "
                f"dimension_fail={stats['dimension_fail']} token_fail={stats['token_fail']} "
                f"attempts={stats['attempts']}\n"
            )
    return data


def _batch_sizes(total: int, batch_size: int) -> list[int]:
    total = max(0, int(total))
    batch_size = max(1, int(batch_size))
    out: list[int] = []
    remaining = total
    while remaining > 0:
        take = min(batch_size, remaining)
        out.append(take)
        remaining -= take
    return out


def _progress(iterable, *, total: int, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="batch")
    except Exception:
        return iterable


def _worker_build_dataset(job: BatchJob) -> list[tuple[str, str]]:
    builder = build_dataset  # step mode dropped in YM rewrite
    return builder(
        job.N,
        job.num_samples,
        max_scr=job.max_scr,
        min_scr=job.min_scr,
        seed=job.seed,
        unit_probability=job.unit_probability,
        old_style_probability=job.old_style_probability,
        denom_repeat_probability=job.denom_repeat_probability,
        scalar_power_probability=job.scalar_power_probability,
        use_denominators=job.use_denominators,
        validate=job.validate,
        M=job.M,
        min_terms=job.min_terms,
        max_terms=job.max_terms,
        log_path=None,
        max_attempts_factor=job.max_attempts_factor,
        full_expand_scrambled=job.full_expand_scrambled,
        max_tokens=job.max_tokens,
        tokenizer_max_particles=job.tokenizer_max_particles,
        validation_pol_modes=job.validation_pol_modes,
        scramble_names=job.scramble_names,
    )


def build_dataset_batched(
    N: int,
    num_samples: int,
    *,
    dataset_kind: str = DEFAULT_DATASET_KIND,
    max_scr: int = DEFAULT_MAX_SCR,
    min_scr: int = DEFAULT_MIN_SCR,
    seed: int | None = None,
    unit_probability: float = UNIT_PROBABILITY,
    old_style_probability: float = OLD_STYLE_PROBABILITY,
    denom_repeat_probability: float = DENOM_REPEAT_PROBABILITY,
    scalar_power_probability: float = SCALAR_POWER_PROBABILITY,
    use_denominators: bool = DEFAULT_USE_DENOMINATORS,
    validate: bool = DEFAULT_VALIDATE,
    M: float = DEFAULT_MASS,
    min_terms: int = DEFAULT_MIN_TERMS,
    max_terms: int = DEFAULT_MAX_TERMS,
    log_path: str | None = None,
    max_attempts_factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR,
    full_expand_scrambled: bool = DEFAULT_FULL_EXPAND_SCRAMBLED,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    tokenizer_max_particles: int = DEFAULT_TOKENIZER_MAX_PARTICLES,
    validation_pol_modes: Sequence[str] = DEFAULT_VALIDATION_POL_MODES,
    scramble_names: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    jobs: int | str = DEFAULT_JOBS,
    progress: bool = DEFAULT_PROGRESS,
) -> list[tuple[str, str]]:
    """Build a dataset in independent batches, optionally using multiple CPUs."""
    if dataset_kind not in {"oneshot", "step"}:
        raise ValueError("dataset_kind must be 'oneshot' or 'step'")
    batch_counts = _batch_sizes(num_samples, batch_size)
    if not batch_counts:
        return []

    jobs = _resolve_jobs(jobs) if isinstance(jobs, str) else max(1, int(jobs))
    base_seed = seed if seed is not None else random.randrange(1, 2**31 - 1)
    seeds = [base_seed + 1000003 * i for i in range(len(batch_counts))]
    normalised_scrambles = normalise_scramble_names(scramble_names)
    validation_pol_modes = tuple(validation_pol_modes)
    job_specs = [
        BatchJob(
            dataset_kind=dataset_kind,
            N=N,
            num_samples=count,
            max_scr=max_scr,
            min_scr=min_scr,
            seed=seeds[i],
            unit_probability=unit_probability,
            old_style_probability=old_style_probability,
            denom_repeat_probability=denom_repeat_probability,
            scalar_power_probability=scalar_power_probability,
            use_denominators=use_denominators,
            validate=validate,
            M=M,
            min_terms=min_terms,
            max_terms=max_terms,
            max_attempts_factor=max_attempts_factor,
            full_expand_scrambled=full_expand_scrambled,
            max_tokens=max_tokens,
            tokenizer_max_particles=tokenizer_max_particles,
            validation_pol_modes=validation_pol_modes,
            scramble_names=normalised_scrambles,
        )
        for i, count in enumerate(batch_counts)
    ]

    if log_path:
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(
                f"# gen_data batched log N={N} target={num_samples} "
                f"dataset_kind={dataset_kind} "
                f"batches={len(job_specs)} batch_size={batch_size} jobs={jobs} "
                f"terms=[{min_terms},{max_terms}] scr=[{min_scr},{max_scr}] "
                f"unit_probability={unit_probability} "
                f"old_style_probability={old_style_probability} "
                f"spurious_repeat_probability={denom_repeat_probability} "
                f"scalar_power_probability={scalar_power_probability} "
                f"full_expand_scrambled={full_expand_scrambled} seed={seed} base_seed={base_seed} "
                f"max_tokens={max_tokens} tokenizer_max_particles={tokenizer_max_particles} "
                f"pol_modes={','.join(validation_pol_modes)} "
                f"scrambles={','.join(normalised_scrambles) if normalised_scrambles else 'none'}\n"
            )

    pairs: list[tuple[str, str]] = []
    if jobs == 1:
        iterator = (_worker_build_dataset(job) for job in job_specs)
        for batch_pairs in _progress(iterator, total=len(job_specs), enabled=progress, desc="generating"):
            pairs.extend(batch_pairs)
    else:
        with mp.Pool(processes=jobs) as pool:
            iterator = pool.imap_unordered(_worker_build_dataset, job_specs)
            for batch_pairs in _progress(iterator, total=len(job_specs), enabled=progress, desc="generating"):
                pairs.extend(batch_pairs)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"# SUMMARY accepted={len(pairs)} requested={num_samples} "
                f"batches={len(job_specs)} jobs={jobs}\n"
            )
    return pairs


def _resolve_jobs(value: str) -> int:
    if str(value).lower() == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(value))


def dedupe_pairs(
    pairs: list[tuple[str, str]],
    *,
    keep: str = "first",
) -> tuple[list[tuple[str, str]], int]:
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")
    if keep == "first":
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for item in pairs:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out, len(pairs) - len(out)
    last_idx = {item: i for i, item in enumerate(pairs)}
    out = [item for i, item in enumerate(pairs) if last_idx[item] == i]
    return out, len(pairs) - len(out)


def write_csv(pairs: Iterable[tuple[str, str]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["simple", "scrambled"])
        for simple, scrambled in pairs:
            writer.writerow([simple, scrambled])


def tokenise_csv(
    inp: str,
    out: str,
    *,
    max_particles: int = DEFAULT_TOKENIZER_MAX_PARTICLES,
    max_sequence_length: int | None = DEFAULT_MAX_TOKENS,
) -> None:
    from Tokenizer import ScatteringAmplitudeTokenizer

    tok = ScatteringAmplitudeTokenizer(
        max_particles=max_particles,
        max_sequence_length=max_sequence_length,
    )
    with open(inp, newline="", encoding="utf-8") as fin, open(
        out, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["simple", "scrambled"])
        writer.writeheader()
        for row in reader:
            writer.writerow(
                {
                    "simple": json.dumps(tok.encode_infix(row["simple"])),
                    "scrambled": json.dumps(tok.encode_infix(row["scrambled"])),
                }
            )

__all__ = [
    'DEFAULT_N_PARTICLES',
    'DEFAULT_SAMPLES',
    'DEFAULT_SEED',
    'DEFAULT_MASS',
    'NSAMPS',
    'DEFAULT_RAW_OUT_TEMPLATE',
    'DEFAULT_TOK_OUT_TEMPLATE',
    'DEFAULT_LOG_OUT_TEMPLATE',
    'DEFAULT_VALIDATE',
    'DEFAULT_TOKENISE',
    'DEFAULT_FULL_EXPAND_SCRAMBLED',
    'DEFAULT_OVERSAMPLE_FACTOR',
    'DEFAULT_DATASET_KIND',
    'DEFAULT_MAX_TOKENS',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_JOBS',
    'DEFAULT_PROGRESS',
    'DEFAULT_TOKENIZER_MAX_PARTICLES',
    'BatchJob',
    '_make_tokenizer',
    '_within_token_budget',
    'build_dataset',
    '_batch_sizes',
    '_progress',
    '_worker_build_dataset',
    'build_dataset_batched',
    '_resolve_jobs',
    'dedupe_pairs',
    'write_csv',
    'tokenise_csv',
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scalar-QED-like amplitude data.")
    parser.add_argument("N", nargs="?", type=int, default=DEFAULT_N_PARTICLES, help="Number of external legs.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--max-scr", type=int, default=DEFAULT_MAX_SCR)
    parser.add_argument("--min-scr", type=int, default=DEFAULT_MIN_SCR)
    parser.add_argument("--min-terms", type=int, default=DEFAULT_MIN_TERMS)
    parser.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS)
    parser.add_argument(
        "--dataset-kind",
        choices=["oneshot", "step"],
        default=DEFAULT_DATASET_KIND,
        help="Generate direct scrambled->simple pairs or one-step simplification pairs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--unit-probability",
        type=float,
        default=UNIT_PROBABILITY,
        help=(
            "Probability that a generated expression uses only unit coefficients "
            "(+1 or -1) for all top-level terms. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--old-style-probability",
        type=float,
        default=OLD_STYLE_PROBABILITY,
        help=(
            "Probability of old-source-like samples: two terms, ±1 top-level "
            "coefficients, and 4pt block weights biased toward single-F chains. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--denom-repeat-probability",
        "--spurious-repeat-probability",
        type=float,
        default=DENOM_REPEAT_PROBABILITY,
        help=(
            "Probability of adding a repeated denominator pole when the extra copy "
            "is spurious because the pole appears in an adjacent p·F-chain "
            "expansion. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--scalar-power-probability",
        type=float,
        default=SCALAR_POWER_PROBABILITY,
        help=(
            "Probability, when adding optional scalar p·p numerator factors, "
            "of repeating an existing physical-pole scalar factor. This creates "
            "manifest numerator powers like (p_i · p_j)^2. Repeated denominators "
            "are generated separately from hidden p·F-chain factors, not from "
            "these trivial scalar cancellations. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of accepted pairs per generation batch. Default: %(default)s",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default=DEFAULT_JOBS,
        help="Number of worker processes, or 'auto'. Default: %(default)s",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS)
    parser.add_argument("--raw-out", type=str, default=None)
    parser.add_argument("--tok-out", type=str, default=None)
    parser.add_argument("--log-out", type=str, default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokenized length per expression. Use 0 to disable filtering.",
    )
    parser.add_argument("--tokenizer-max-particles", type=int, default=DEFAULT_TOKENIZER_MAX_PARTICLES)
    parser.add_argument(
        "--validation-pol-modes",
        nargs="+",
        choices=["coulomb", "covariant"],
        default=list(DEFAULT_VALIDATION_POL_MODES),
        help="Polarization modes used for numerical validation.",
    )
    parser.add_argument(
        "--scrambles",
        nargs="*",
        default=None,
        choices=["all", "none", *list(_SCRAMBLER_BY_NAME)],
        help=(
            "Enabled scramble labels. Omit for the default set. "
            "Use --scrambles none with --min-scr 0 --max-scr 0 for expanded->simple only."
        ),
    )
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-tokenise", action="store_true")
    parser.add_argument(
        "--grouped-scrambled",
        action="store_true",
        help="Keep the old grouped scrambled style instead of fully expanding scrambled expressions.",
    )
    args = parser.parse_args()

    nsamps = args.samples // 1000
    raw_out = args.raw_out or DEFAULT_RAW_OUT_TEMPLATE.format(N=args.N, NSAMPS=nsamps)
    tok_out = args.tok_out or DEFAULT_TOK_OUT_TEMPLATE.format(N=args.N, NSAMPS=nsamps)
    log_out = args.log_out or DEFAULT_LOG_OUT_TEMPLATE.format(N=args.N, NSAMPS=nsamps)

    t0 = time.perf_counter()
    oversample = int(round(args.samples * DEFAULT_OVERSAMPLE_FACTOR))
    max_tokens = None if args.max_tokens <= 0 else args.max_tokens
    pairs = build_dataset_batched(
        args.N,
        oversample,
        dataset_kind=args.dataset_kind,
        max_scr=args.max_scr,
        min_scr=args.min_scr,
        seed=args.seed,
        unit_probability=args.unit_probability,
        old_style_probability=args.old_style_probability,
        denom_repeat_probability=args.denom_repeat_probability,
        scalar_power_probability=args.scalar_power_probability,
        use_denominators=DEFAULT_USE_DENOMINATORS,
        validate=not args.no_validate,
        M=args.mass,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
        log_path=log_out,
        full_expand_scrambled=(not args.grouped_scrambled) if DEFAULT_FULL_EXPAND_SCRAMBLED else False,
        max_tokens=max_tokens,
        tokenizer_max_particles=args.tokenizer_max_particles,
        validation_pol_modes=tuple(args.validation_pol_modes),
        scramble_names=args.scrambles,
        batch_size=args.batch_size,
        jobs=_resolve_jobs(args.jobs),
        progress=DEFAULT_PROGRESS and not args.no_progress,
    )
    t1 = time.perf_counter()

    before = len(pairs)
    pairs, removed = dedupe_pairs(pairs)
    pairs = pairs[: args.samples]
    write_csv(pairs, raw_out)
    if DEFAULT_TOKENISE and not args.no_tokenise:
        tokenise_csv(
            raw_out,
            tok_out,
            max_particles=args.tokenizer_max_particles,
            max_sequence_length=max_tokens,
        )
    t2 = time.perf_counter()

    print(f"{len(pairs)} pairs -> {raw_out}")
    print(f"  generation : {t1 - t0:.2f}s")
    print(f"  dedupe     : removed {removed} ({before} -> {len(pairs)})")
    print(f"  write/tok  : {t2 - t1:.2f}s")
    print(f"  log        : {log_out}")
