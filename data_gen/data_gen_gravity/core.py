"""Gravity expression model, compact target generator, and numerical checks."""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .. import gen_data as sqed
from .kinematics import SpinorKinematics, generate_kinematics, mdot, with_references

DOT = "·"


def p(i: int) -> str:
    return f"p_{i}"


def e(i: int) -> str:
    return f"e_{i}"


def F(i: int) -> str:
    return f"F_{i}"


def dot(a: str, b: str) -> str:
    return f"{a} {DOT} {b}"


def X(i: int, a: int, b: int) -> str:
    """A single, deliberately unmerged ``p_a·F_i·p_b`` contraction."""
    return f"{p(a)} {DOT} {F(i)} {DOT} {p(b)}"


def s(a: int, b: int) -> str:
    a, b = sorted((a, b))
    return dot(p(a), p(b))


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    scalar_legs: tuple[int, ...]
    graviton_legs: tuple[int, ...]
    target_dimension: int

    @property
    def field_strengths_per_term(self) -> int:
        return 2 * len(self.graviton_legs)


PROCESS_SPECS: dict[str, ProcessSpec] = {
    "3s2h": ProcessSpec("3s2h", (1, 2, 3), (4, 5), 0),
    "4s1h": ProcessSpec("4s1h", (1, 2, 3, 4), (5,), -2),
}


BENCHMARK_3S2H = (
    "-(p_1 · F_4 · p_2)*(p_1 · F_4 · p_5)*(p_1 · F_5 · p_3)"
    "*(p_2 · F_5 · p_3)/((p_1 · p_4)*(p_1 · p_5)*(p_2 · p_4)"
    "*(p_2 · p_5)*(p_3 · p_5)*(p_4 · p_5))"
    " - (p_1 · F_4 · p_3)*(p_2 · F_4 · p_3)*(p_1 · F_5 · p_2)"
    "*(p_1 · F_5 · p_4)/((p_1 · p_4)*(p_1 · p_5)*(p_2 · p_4)"
    "*(p_2 · p_5)*(p_3 · p_4)*(p_4 · p_5))"
)

BENCHMARK_4S1H = (
    "(p_1 · F_5 · p_4)*(p_2 · F_5 · p_3)"
    "/((p_1 · p_4)*(p_2 · p_3)*(p_1 · p_5)*(p_3 · p_5))"
    " + (p_1 · F_5 · p_4)*(p_3 · F_5 · p_4)"
    "/((p_2 · p_3)*(p_1 · p_5)*(p_3 · p_5)*(p_4 · p_5))"
    " - (p_1 · F_5 · p_2)*(p_2 · F_5 · p_3)"
    "/((p_1 · p_4)*(p_1 · p_5)*(p_2 · p_5)*(p_3 · p_5))"
)

BENCHMARKS: dict[str, str] = {
    "3s2h": BENCHMARK_3S2H,
    "4s1h": BENCHMARK_4S1H,
}


def expand_expression(expr: str, *, full: bool = True) -> str:
    expanded = sqed.expand_simple_expression(expr)
    return sqed.full_expand_expression(expanded) if full else expanded


def count_expanded_terms(expr: str) -> int:
    """Count additive leaves after fully expanding field-strength blocks."""
    expanded = expand_expression(expr, full=True)
    tree = sqed._Parser(sqed._tokenize(expanded)).parse()

    def visit(node) -> int:
        if isinstance(node, sqed._BinOp) and node.op in ("+", "-"):
            return visit(node.left) + visit(node.right)
        return 1

    return visit(tree)


def field_strength_counts(expr: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for raw in re.findall(r"F_(\d+)", expr):
        leg = int(raw)
        counts[leg] = counts.get(leg, 0) + 1
    return counts


def field_strength_counts_per_term(expr: str) -> list[dict[int, int]]:
    """Return F multiplicities for each top-level additive term."""
    tree = sqed._Parser(sqed._tokenize(expr)).parse()

    def additive_terms(node) -> list:
        if isinstance(node, sqed._BinOp) and node.op in ("+", "-"):
            return additive_terms(node.left) + additive_terms(node.right)
        return [node]

    def count(node, result: dict[int, int]) -> None:
        if isinstance(node, sqed._Vec) and node.tag == "F":
            result[node.idx] = result.get(node.idx, 0) + 1
        elif isinstance(node, sqed._DotChain):
            for part in node.parts:
                if isinstance(part, sqed._Vec) and part.tag == "F":
                    result[part.idx] = result.get(part.idx, 0) + 1
        elif isinstance(node, sqed._UnaryOp):
            count(node.operand, result)
        elif isinstance(node, sqed._BinOp):
            count(node.left, result)
            count(node.right, result)

    output: list[dict[int, int]] = []
    for term in additive_terms(tree):
        item: dict[int, int] = {}
        count(term, item)
        output.append(item)
    return output


def expression_mass_dimension(expr: str) -> int:
    """Dimension of a homogeneous compact expression.

    ``p·F·p`` has dimension three, ``p·p`` dimension two.  All top-level
    terms are required to agree.
    """
    tree = sqed._Parser(sqed._tokenize(expr)).parse()

    def dim(node) -> int:
        if isinstance(node, sqed._Num):
            return 0
        if isinstance(node, sqed._UnaryOp):
            return dim(node.operand)
        if isinstance(node, sqed._DotChain):
            parts = [x for x in node.parts if isinstance(x, sqed._Vec)]
            if any(x.tag == "F" for x in parts):
                return sum(1 for x in parts if x.tag in ("p", "F"))
            return 2 if len(parts) == 2 else max(0, len(parts))
        if isinstance(node, sqed._BinOp):
            if node.op in ("+", "-"):
                left, right = dim(node.left), dim(node.right)
                if left != right:
                    raise ValueError(f"Non-homogeneous dimensions: {left} and {right}")
                return left
            if node.op == "*":
                return dim(node.left) + dim(node.right)
            if node.op == "/":
                return dim(node.left) - dim(node.right)
            if node.op == "**":
                if not isinstance(node.right, sqed._Num):
                    raise ValueError("Non-numeric power")
                return int(dim(node.left) * node.right.value)
        return 0

    return dim(tree)


def _eval_tree(tree, kin: SpinorKinematics) -> complex:
    P = {i: kin.momenta[i - 1] for i in range(1, 6)}
    E = dict(kin.polarisations)

    def evaluate(node):
        if isinstance(node, sqed._Num):
            return complex(node.value)
        if isinstance(node, sqed._UnaryOp):
            value = evaluate(node.operand)
            return -value if node.op == "-" else value
        if isinstance(node, sqed._BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            return {
                "+": lambda: left + right,
                "-": lambda: left - right,
                "*": lambda: left * right,
                "/": lambda: left / right,
                "**": lambda: left**right,
            }[node.op]()
        if isinstance(node, tuple) and node[0] == sqed._DOT_TAG:
            lhs, rhs = node[1], node[2]
            va = P[lhs[2]] if lhs[1] == "p" else E[lhs[2]]
            vb = P[rhs[2]] if rhs[1] == "p" else E[rhs[2]]
            return mdot(va, vb)
        return 0j

    return complex(evaluate(sqed._expand_ast(tree)))


def eval_expression(expr: str, kin: SpinorKinematics) -> complex:
    """Evaluate either compact F notation or its dot-product expansion."""
    tree = sqed._Parser(sqed._tokenize(expr)).parse()
    return _eval_tree(tree, kin)


def numerically_equivalent(
    left: str,
    right: str,
    process: str | ProcessSpec,
    *,
    seeds: Sequence[int] = (101, 307),
    reference_modes: Sequence[str] = ("first", "last", "random"),
    gauge_shift: bool = True,
    rtol: float = 2e-8,
    atol: float = 2e-9,
) -> tuple[bool, float]:
    spec = PROCESS_SPECS[process] if isinstance(process, str) else process
    left_tree = sqed._Parser(sqed._tokenize(left)).parse()
    right_tree = sqed._Parser(sqed._tokenize(right)).parse()
    worst = 0.0
    for seed in seeds:
        base = generate_kinematics(
            seed=seed, graviton_legs=spec.graviton_legs, reference_mode="cyclic"
        )
        gauges = [
            with_references(
                base,
                spec.graviton_legs,
                reference_mode=mode,
                seed=seed + 11,
            )
            for mode in reference_modes
        ]
        if gauge_shift:
            shifts = {
                leg: complex(0.19 * (leg + 1), -0.07 * leg)
                for leg in spec.graviton_legs
            }
            gauges.append(
                with_references(
                    base,
                    spec.graviton_legs,
                    reference_mode="cyclic",
                    gauge_shifts=shifts,
                )
            )
        for kin in gauges:
            a, b = _eval_tree(left_tree, kin), _eval_tree(right_tree, kin)
            scale = max(1.0, abs(a), abs(b))
            error = abs(a - b) / scale
            worst = max(worst, float(error))
            if not np.isfinite(error) or abs(a - b) > atol + rtol * scale:
                return False, worst
    return True, worst


def validate_expression_pair(
    simple: str,
    scrambled: str,
    process: str | ProcessSpec,
    *,
    seeds: Sequence[int] = (101, 307),
) -> tuple[bool, str]:
    spec = PROCESS_SPECS[process] if isinstance(process, str) else process
    expected = {leg: 2 for leg in spec.graviton_legs}
    if any(counts != expected for counts in field_strength_counts_per_term(simple)):
        return False, "field-strength multiplicity"
    try:
        if expression_mass_dimension(simple) != spec.target_dimension:
            return False, "mass dimension"
    except ValueError:
        return False, "non-homogeneous dimension"
    try:
        ok, error = numerically_equivalent(simple, scrambled, spec, seeds=seeds)
    except (KeyError, ValueError, ZeroDivisionError, FloatingPointError):
        return False, "numerical evaluation"
    return (True, f"relative error {error:.3e}") if ok else (False, f"mismatch {error:.3e}")


def _physical_poles() -> list[tuple[int, int]]:
    return list(itertools.combinations(range(1, 6), 2))


def _x_endpoints(graviton: int, rng: random.Random) -> tuple[int, int]:
    choices = [i for i in range(1, 6) if i != graviton]
    a, b = rng.sample(choices, 2)
    return (a, b) if a < b else (b, a)


def _term(spec: ProcessSpec, rng: random.Random) -> str:
    factors: list[str] = []
    for graviton in spec.graviton_legs:
        for _ in range(2):
            a, b = _x_endpoints(graviton, rng)
            factors.append(f"({X(graviton, a, b)})")

    scalar_dot_count = 1 if rng.random() < 0.22 else 0
    numerator_poles: set[tuple[int, int]] = set()
    for _ in range(scalar_dot_count):
        pair = rng.choice(_physical_poles())
        numerator_poles.add(pair)
        factors.append(f"({s(*pair)})")

    numerator_dimension = 3 * spec.field_strengths_per_term + 2 * scalar_dot_count
    denominator_count = (numerator_dimension - spec.target_dimension) // 2
    pool = [pair for pair in _physical_poles() if pair not in numerator_poles]
    rng.shuffle(pool)
    required: list[tuple[int, int]] = []
    for graviton in spec.graviton_legs:
        candidates = [pair for pair in pool if graviton in pair and pair not in required]
        if candidates:
            required.append(rng.choice(candidates))
    remaining = [pair for pair in pool if pair not in required]
    denominator_pairs = required + remaining[: denominator_count - len(required)]
    if len(denominator_pairs) != denominator_count:
        raise RuntimeError("Not enough distinct physical five-point poles")

    numerator = "*".join(factors)
    denominator = "*".join(f"({s(*pair)})" for pair in denominator_pairs)
    return f"({numerator})/({denominator})"


def _normalise_text(expr: str) -> str:
    return re.sub(r"\s+", "", expr).replace("**", "^")


def compact_signature(expr: str) -> tuple:
    """Canonical compact signature, insensitive to product/term ordering.

    This is deliberately structural rather than a string comparison so the
    benchmark blacklist also catches equivalent formatting and factor order.
    """
    tree = sqed._Parser(sqed._tokenize(expr)).parse()

    def signed_terms(node, sign: int = 1) -> list[tuple[int, object]]:
        if isinstance(node, sqed._UnaryOp):
            return signed_terms(node.operand, -sign)
        if isinstance(node, sqed._BinOp) and node.op == "+":
            return signed_terms(node.left, sign) + signed_terms(node.right, sign)
        if isinstance(node, sqed._BinOp) and node.op == "-":
            return signed_terms(node.left, sign) + signed_terms(node.right, -sign)
        return [(sign, node)]

    def factor_signature(node) -> tuple[int, tuple]:
        if isinstance(node, sqed._DotChain):
            parts = [part for part in node.parts if isinstance(part, sqed._Vec)]
            if (
                len(parts) == 3
                and parts[0].tag == "p"
                and parts[1].tag == "F"
                and parts[2].tag == "p"
            ):
                a, b = parts[0].idx, parts[2].idx
                sign = 1
                if a > b:
                    a, b = b, a
                    sign = -1
                return sign, ("X", parts[1].idx, a, b)
            if len(parts) == 2 and all(part.tag == "p" for part in parts):
                a, b = sorted((parts[0].idx, parts[1].idx))
                return 1, ("s", a, b)
        if isinstance(node, sqed._Num):
            value = node.value
            sign = -1 if value < 0 else 1
            return sign, ("n", abs(value))
        return 1, ("raw", _normalise_text(sqed._ast_to_infix(node)))

    def rational_factors(node, inverted: bool = False) -> tuple[int, list, list]:
        if isinstance(node, sqed._UnaryOp):
            sign, numerator, denominator = rational_factors(node.operand, inverted)
            return -sign, numerator, denominator
        if isinstance(node, sqed._BinOp) and node.op == "*":
            ls, ln, ld = rational_factors(node.left, inverted)
            rs, rn, rd = rational_factors(node.right, inverted)
            return ls * rs, ln + rn, ld + rd
        if isinstance(node, sqed._BinOp) and node.op == "/":
            ls, ln, ld = rational_factors(node.left, inverted)
            rs, rn, rd = rational_factors(node.right, not inverted)
            return ls * rs, ln + rn, ld + rd
        factor_sign, factor = factor_signature(node)
        if inverted:
            return factor_sign, [], [factor]
        return factor_sign, [factor], []

    canonical_terms = []
    for outer_sign, node in signed_terms(tree):
        inner_sign, numerator, denominator = rational_factors(node)
        canonical_terms.append(
            (
                outer_sign * inner_sign,
                tuple(sorted(numerator)),
                tuple(sorted(denominator)),
            )
        )
    return tuple(sorted(canonical_terms))


def _relabel(expr: str, mapping: Mapping[int, int]) -> str:
    placeholders = {i: f"LEG{i}X" for i in range(1, 6)}
    for i, placeholder in placeholders.items():
        expr = re.sub(rf"([peF])_{i}\b", rf"\1_{placeholder}", expr)
    for i, placeholder in placeholders.items():
        expr = expr.replace(f"_{placeholder}", f"_{mapping[i]}")
    return expr


def benchmark_relabelings(process: str) -> set[tuple]:
    spec = PROCESS_SPECS[process]
    results: set[tuple] = set()
    for scalar_perm in itertools.permutations(spec.scalar_legs):
        for graviton_perm in itertools.permutations(spec.graviton_legs):
            mapping = dict(zip(spec.scalar_legs, scalar_perm))
            mapping.update(zip(spec.graviton_legs, graviton_perm))
            results.add(compact_signature(_relabel(BENCHMARKS[process], mapping)))
    return results


_BENCHMARK_BLACKLIST = {
    process: benchmark_relabelings(process) for process in PROCESS_SPECS
}


def is_benchmark_leak(expr: str, process: str) -> bool:
    try:
        return compact_signature(expr) in _BENCHMARK_BLACKLIST[process]
    except Exception:
        return _normalise_text(expr) in _BENCHMARK_BLACKLIST[process]


def generate_target(
    process: str,
    *,
    rng: random.Random | None = None,
    min_terms: int = 1,
    max_terms: int = 3,
) -> str:
    """Generate a homogeneous 1--3 term compact gravity target."""
    if process == "mixed":
        process = (rng or random).choice(tuple(PROCESS_SPECS))
    spec = PROCESS_SPECS[process]
    rng = rng or random.Random()
    for _ in range(200):
        terms: list[str] = []
        while len(terms) < rng.randint(min_terms, max_terms):
            candidate = _term(spec, rng)
            if candidate not in terms:
                terms.append(candidate)
        pieces: list[str] = []
        for index, term in enumerate(terms):
            sign = rng.choice((-1, 1))
            if index == 0:
                pieces.append(term if sign > 0 else f"-{term}")
            else:
                pieces.append((" + " if sign > 0 else " - ") + term)
        expr = "".join(pieces)
        if not is_benchmark_leak(expr, process):
            return expr
    raise RuntimeError("Could not generate a non-benchmark target")


def paper_spinor_value(process: str, kin: SpinorKinematics) -> complex:
    """Original Eqs. (4.7)/(4.8); the latter is returned as ``2 M``."""
    a, q = kin.angle, kin.square
    if process == "3s2h":
        prefactor = a(1, 2) * a(1, 3) * a(2, 3) / (
            a(2, 4) * a(2, 5) * a(4, 5)
        )
        return prefactor * (
            q(1, 4) * q(3, 5) / (a(1, 4) * a(3, 5))
            - q(1, 5) * q(3, 4) / (a(1, 5) * a(3, 4))
        )
    if process == "4s1h":
        prefactor = q(2, 5) * q(4, 5) / (
            a(1, 5) * a(3, 5) * q(1, 4) * q(2, 3)
        )
        bracket = (
            1
            + a(1, 4) * a(3, 4) * q(1, 4)
            / (a(2, 3) * a(4, 5) * q(2, 5))
            - a(1, 2) * a(2, 3) * q(2, 3)
            / (a(1, 4) * a(2, 5) * q(4, 5))
        )
        return 2 * prefactor * bracket
    raise ValueError(process)


def verify_paper_benchmarks(
    seeds: Sequence[int] = (17, 31, 73),
    *,
    rtol: float = 5e-8,
) -> dict[str, float]:
    """Verify the field-strength fixtures against the paper's spinor forms."""
    errors: dict[str, float] = {}
    for process, spec in PROCESS_SPECS.items():
        worst = 0.0
        for seed in seeds:
            kin = generate_kinematics(
                seed=seed,
                graviton_legs=spec.graviton_legs,
                reference_mode="cyclic",
            )
            field_value = eval_expression(BENCHMARKS[process], kin)
            spinor_value = paper_spinor_value(process, kin)
            scale = max(1.0, abs(field_value), abs(spinor_value))
            error = abs(field_value - spinor_value) / scale
            worst = max(worst, float(error))
            if error > rtol:
                raise AssertionError(
                    f"{process} field-strength fixture disagrees with paper: {error:.3e}"
                )
        errors[process] = worst
    return errors
