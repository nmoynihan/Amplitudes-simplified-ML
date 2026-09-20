"""Explicit ordered seeds for the two existing scalar--gravity benchmarks.

These are auxiliary, gauge-invariant building blocks with specified permutation
sums, not color-ordered gravity amplitudes or a general amplitude generator.
See ORDERED_GRAVITY.md for their definitions, proofs, and normalization.

Run: python -m data_gen.data_gen_gravity.ordered_benchmarks --output report.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import BENCHMARKS, PROCESS_SPECS, X, eval_expression, paper_spinor_value, s
from .kinematics import generate_kinematics, with_references

DEFAULT_ORDER = (1, 2, 3, 4, 5)
SCALAR_CHANNEL = frozenset((frozenset((1, 4)), frozenset((2, 3))))


def _validate_order(process: str, order: Sequence[int]) -> tuple[int, ...]:
    if process not in PROCESS_SPECS:
        raise ValueError(f"Unknown gravity process: {process}")
    order = tuple(order)
    if (len(order) != 5 or any(type(leg) is not int for leg in order)
            or set(order) != set(DEFAULT_ORDER)):
        raise ValueError("Order must be a permutation of the integer labels 1..5")
    spec = PROCESS_SPECS[process]
    scalar_count = len(spec.scalar_legs)
    if (set(order[:scalar_count]) != set(spec.scalar_legs)
            or set(order[scalar_count:]) != set(spec.graviton_legs)):
        raise ValueError("Order must preserve scalar and graviton species")
    if process == "4s1h":
        a, b, c, d, _ = order
        channel = frozenset((frozenset((a, d)), frozenset((b, c))))
        if channel != SCALAR_CHANNEL:
            raise ValueError("Order must preserve the benchmark's scalar channel {14|23}")
    return order


def _product(factors: Sequence[str]) -> str:
    return "*".join(f"({factor})" for factor in factors)


def ordered_seed(process: str, order: Sequence[int] = DEFAULT_ORDER) -> str:
    """Return the specified fixed-order seed in repository normalization.

    For 3s2h the slots are (a,b,c;h,k); for 4s1h they are (a,b,c,d;h),
    with the scalar channel (a,d)|(b,c) held fixed. Relabelings act on all
    momenta, field strengths, and denominators together.
    """
    a, b, c, d, e = _validate_order(process, order)
    if process == "3s2h":
        h, k = d, e
        numerator = _product((X(h, a, b), X(h, a, k), X(k, a, c), X(k, b, c)))
        denominator = _product((s(a, h), s(a, k), s(b, h), s(b, k), s(c, k), s(h, k)))
        return f"-({numerator})/({denominator})"
    h = e
    shared = _product((X(h, a, d), X(h, b, c)))
    shared_denominator = _product((s(a, d), s(b, c), s(a, h), s(c, h)))
    second = _product((X(h, a, d), X(h, c, d)))
    second_denominator = _product((s(b, c), s(a, h), s(c, h), s(d, h)))
    return f"({shared})/(2*({shared_denominator})) + ({second})/({second_denominator})"


def reconstruction_terms(
    process: str, order: Sequence[int] = DEFAULT_ORDER,
) -> tuple[str, str]:
    """Two ordered pieces whose sum is exactly the corresponding fixture."""
    a, b, c, d, e = _validate_order(process, order)
    partner = (a, b, c, e, d) if process == "3s2h" else (c, d, a, b, e)
    return ordered_seed(process, order), ordered_seed(process, partner)


def reconstruct_benchmark(process: str, order: Sequence[int] = DEFAULT_ORDER) -> str:
    return " + ".join(f"({term})" for term in reconstruction_terms(process, order))


def permutation_orders(process: str) -> tuple[tuple[int, ...], ...]:
    """Species-preserving orders; also preserve {14|23} for the four scalars."""
    _validate_order(process, DEFAULT_ORDER)
    spec = PROCESS_SPECS[process]
    orders = []
    for scalars in itertools.permutations(spec.scalar_legs):
        for gravitons in itertools.permutations(spec.graviton_legs):
            order = scalars + gravitons
            if process == "4s1h":
                a, b, c, d, _ = order
                if frozenset((frozenset((a, d)), frozenset((b, c)))) != SCALAR_CHANNEL:
                    continue
            orders.append(order)
    return tuple(orders)


def full_permutation_sum(process: str) -> str:
    """Weighted sum over all allowed orders, valid in the benchmark sector.

    The 3s2h all-order identity uses on-shell, same-helicity graviton kinematics;
    the 4s1h identity uses on-shell momenta and transverse polarization. The
    minimal two-piece identity above requires only antisymmetry of F.
    """
    terms = [ordered_seed(process, order) for order in permutation_orders(process)]
    weight_denominator = 6 if process == "3s2h" else 4
    total = " + ".join(f"({term})" for term in terms)
    return f"({total})/{weight_denominator}"


def verify_reconstruction(seeds: Sequence[int] = tuple(range(30))) -> dict:
    """Cross-check both permutation sums against fixtures and paper spinors."""
    if not seeds:
        raise ValueError("At least one kinematic seed is required")
    report = {}
    for process, spec in PROCESS_SPECS.items():
        minimal = reconstruct_benchmark(process)
        full = full_permutation_sum(process)
        seed_expression = ordered_seed(process)
        worst = {"two_piece_vs_fixture": 0.0, "all_orders_vs_fixture": 0.0,
                 "two_piece_vs_paper": 0.0, "seed_gauge_invariance": 0.0}
        configurations = 0
        for seed in seeds:
            base = generate_kinematics(seed=seed, graviton_legs=spec.graviton_legs)
            base_seed_value = eval_expression(seed_expression, base)
            for mode in ("cyclic", "first", "last", "random", "shifted"):
                kin = with_references(
                    base, spec.graviton_legs,
                    reference_mode="cyclic" if mode == "shifted" else mode,
                    seed=seed + 911,
                    gauge_shifts=({leg: 0.73 + 0.27j for leg in spec.graviton_legs}
                                  if mode == "shifted" else None),
                )
                fixture = eval_expression(BENCHMARKS[process], kin)
                two_piece = eval_expression(minimal, kin)
                comparisons = {
                    "two_piece_vs_fixture": (two_piece, fixture),
                    "all_orders_vs_fixture": (eval_expression(full, kin), fixture),
                    # paper_spinor_value already includes the repository's factor 2
                    # for 4s1h. This is not a new normalization adjustment.
                    "two_piece_vs_paper": (two_piece, paper_spinor_value(process, kin)),
                    "seed_gauge_invariance": (eval_expression(seed_expression, kin), base_seed_value),
                }
                for name, (left, right) in comparisons.items():
                    scale = max(abs(left), abs(right))
                    error = abs(left - right) / scale if scale else 0.0
                    if not np.isfinite(error) or abs(left - right) > 1e-10 + 5e-8 * scale:
                        raise AssertionError(f"{process}, seed {seed}, gauge {mode}, {name}: {error}")
                    worst[name] = max(worst[name], float(error))
                configurations += 1
        report[process] = {
            "ordered_seed": seed_expression,
            "reconstruction_orders": ([DEFAULT_ORDER, (1, 2, 3, 5, 4)]
                                      if process == "3s2h"
                                      else [DEFAULT_ORDER, (3, 4, 1, 2, 5)]),
            "allowed_order_count": len(permutation_orders(process)),
            "all_order_weight": "1/6" if process == "3s2h" else "1/4",
            # Multiply the repository value by this factor to obtain the paper value.
            "repository_to_paper_factor": 1 if process == "3s2h" else 0.5,
            "kinematic_seeds": list(seeds),
            "configurations": configurations,
            "maximum_relative_errors": worst,
        }
    return report


def check_saved_targets(path: Path) -> dict[str, int]:
    """Check which existing test rows use the two reconstructed targets.

    This checks the targets, not the numerical validity of every scrambled input.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    by_expression = {expression: process for process, expression in BENCHMARKS.items()}
    counts = dict.fromkeys(BENCHMARKS, 0)
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            process = by_expression.get(row.get("simple"))
            if process is None:
                raise ValueError(f"Unexpected benchmark target at {path}, row {index}")
            counts[process] += 1
    if not all(counts.values()):
        raise ValueError("Saved benchmark file must contain both reconstructed targets")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30, help="number of numerical kinematic points")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-raw", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    report = {
        "definition": "auxiliary ordered benchmark seeds, not color-ordered amplitudes",
        "checks": verify_reconstruction(tuple(range(args.seeds))),
    }
    if args.benchmark_raw:
        report["saved_target_counts"] = check_saved_targets(args.benchmark_raw)
        report["saved_scrambled_inputs_revalidated"] = False
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
