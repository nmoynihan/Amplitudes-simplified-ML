"""Gravity expression model, compact target generator, and numerical checks."""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from collections import Counter, defaultdict
from fractions import Fraction
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


class _ExactParser(sqed._Parser):
    """Use the shared AST without passing rational literals through a float."""

    def _primary(self):
        token = self.peek()
        if token and re.fullmatch(r"\d+(?:\.\d+)?", token):
            self.pop()
            node = sqed._Num.__new__(sqed._Num)
            node.value = Fraction(token)
            return node
        return super()._primary()

    def _power(self):
        node = self._primary()
        if self.peek() == "**":
            self.pop()
            node = sqed._BinOp("**", node, self._factor())
        return node


def _integer_power(node) -> int:
    value = _constant_value(node)
    if value is None or value.denominator != 1:
        raise ValueError("Gravity powers require an integer exponent")
    exponent = value.numerator
    if abs(exponent) > 64:
        raise ValueError("Gravity power exceeds the supported bound of 64")
    return exponent


def _constant_value(node) -> Fraction | None:
    """Exactly fold a numerical-only subtree, otherwise return None."""
    if isinstance(node, sqed._Num):
        return Fraction(str(node.value))
    if isinstance(node, sqed._UnaryOp):
        value = _constant_value(node.operand)
        return -value if value is not None and node.op == "-" else value
    if isinstance(node, sqed._BinOp):
        left, right = _constant_value(node.left), _constant_value(node.right)
        if left is None or right is None:
            return None
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        if node.op == "**":
            if right.denominator != 1 or abs(right.numerator) > 64:
                raise ValueError("Gravity powers require a bounded integer exponent")
            return left ** right.numerator
    return None


def _validate_scalar_ast(node) -> None:
    if isinstance(node, sqed._Num):
        return
    if isinstance(node, sqed._UnaryOp):
        if node.op != "-":
            raise ValueError(f"Unsupported unary operator: {node.op}")
        _validate_scalar_ast(node.operand)
        return
    if isinstance(node, sqed._BinOp):
        if node.op not in {"+", "-", "*", "/", "**"}:
            raise ValueError(f"Unsupported operator: {node.op}")
        _validate_scalar_ast(node.left)
        _validate_scalar_ast(node.right)
        if node.op == "**":
            _integer_power(node.right)
        if node.op == "/" and _constant_value(node.right) == 0:
            raise ValueError("Division by zero")
        return
    if isinstance(node, sqed._DotChain):
        trace = bool(node.parts) and node.parts[-1] is sqed._DotChain._TR
        parts = node.parts[:-1] if trace else node.parts
        if not parts or any(not isinstance(part, sqed._Vec) for part in parts):
            raise ValueError("Malformed gravity contraction")
        if any(part.tag not in {"p", "e", "F"} or not 1 <= part.idx <= 5 for part in parts):
            raise ValueError("Unsupported gravity vector")
        if trace:
            if len(parts) < 2 or any(part.tag != "F" for part in parts):
                raise ValueError("A trace requires at least two field strengths")
        elif any(part.tag == "F" for part in parts):
            if not (len(parts) >= 3 and parts[0].tag == parts[-1].tag == "p"
                    and all(part.tag == "F" for part in parts[1:-1])):
                raise ValueError("A mixed chain requires momentum endpoints and ordered F factors")
        elif len(parts) != 2 or any(part.tag not in {"p", "e"} for part in parts):
            raise ValueError("A scalar dot requires exactly two p/e vectors")
        return
    raise ValueError(f"Unsupported gravity scalar node: {type(node).__name__}")


def parse_expression(expr: str):
    """Strict five-point scalar parser with exact rational number leaves."""
    tokens = sqed._strict_tokenize(expr)
    sqed._validate_strict_token_sequence(tokens)
    depth = 0
    for token in tokens:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unmatched closing parenthesis")
    if depth:
        raise ValueError("Unmatched opening parenthesis")
    parser = _ExactParser(tokens)
    tree = parser.parse()
    if parser.i != len(tokens):
        raise ValueError("Gravity expression was not fully consumed")
    _validate_scalar_ast(tree)
    return tree


def _degree_and_dimension(node) -> tuple[dict[int, int], int]:
    if isinstance(node, sqed._Num):
        return {}, 0
    if isinstance(node, sqed._UnaryOp):
        return _degree_and_dimension(node.operand)
    if isinstance(node, sqed._DotChain):
        parts = [part for part in node.parts if isinstance(part, sqed._Vec)]
        degree = Counter(part.idx for part in parts if part.tag in {"e", "F"})
        return dict(degree), sum(part.tag in {"p", "F"} for part in parts)
    if isinstance(node, sqed._BinOp):
        left, ld = _degree_and_dimension(node.left)
        if node.op == "**":
            power = _integer_power(node.right)
            return {leg: count * power for leg, count in left.items() if count * power}, ld * power
        right, rd = _degree_and_dimension(node.right)
        if node.op in {"+", "-"}:
            if left != right:
                raise ValueError(f"Non-homogeneous polarization degrees: {left} and {right}")
            if ld != rd:
                raise ValueError(f"Non-homogeneous dimensions: {ld} and {rd}")
            return left, ld
        sign = -1 if node.op == "/" else 1
        degree = {leg: left.get(leg, 0) + sign * right.get(leg, 0) for leg in left.keys() | right.keys()}
        return {leg: count for leg, count in degree.items() if count}, ld + sign * rd
    raise ValueError(f"Unsupported gravity degree node: {type(node).__name__}")


def polarization_degree(expr: str) -> dict[int, int]:
    """Homogeneous polarization degree, including e, F and integer powers."""
    return _degree_and_dimension(parse_expression(expr))[0]


def field_strength_counts_per_term(expr: str) -> list[dict[int, int]]:
    """Polarization degrees per additive term, checking nested homogeneity.

    This historical name is retained for callers. Expanded e vectors carry the
    same polarization degree as compact F factors; a power multiplies degree.
    """
    tree = parse_expression(expr)
    _degree_and_dimension(tree)

    def terms(node):
        if isinstance(node, sqed._BinOp) and node.op in {"+", "-"}:
            return terms(node.left) + terms(node.right)
        return [_degree_and_dimension(node)[0]]

    return terms(tree)


def expression_mass_dimension(expr: str) -> int:
    """Homogeneous stripped dimension: p and F have dimension one, e zero."""
    return _degree_and_dimension(parse_expression(expr))[1]


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
        raise ValueError(f"Unsupported evaluated gravity node: {type(node).__name__}")

    return complex(evaluate(sqed._expand_ast(tree)))


def eval_expression(expr: str, kin: SpinorKinematics) -> complex:
    """Evaluate either compact F notation or its dot-product expansion."""
    tree = parse_expression(expr)
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
    left_tree = parse_expression(left)
    right_tree = parse_expression(right)
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
            if not (np.isfinite(abs(a)) and np.isfinite(abs(b))):
                return False, float("inf")
            difference = abs(a - b)
            scale = max(abs(a), abs(b))
            error = difference / scale if scale else difference
            worst = max(worst, float(error))
            if not sqed.numeric_values_close(
                a,
                b,
                tol_abs=atol,
                tol_rel=rtol,
            ):
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
    for label, expression in (("target", simple), ("source", scrambled)):
        try:
            tree = parse_expression(expression)
        except (ArithmeticError, ValueError) as exc:
            return False, f"{label} parse error: {exc}"
        try:
            degree, dimension = _degree_and_dimension(tree)
        except ValueError as exc:
            return False, f"{label} non-homogeneous: {exc}"
        if degree != expected:
            return False, f"{label} polarization degree"
        if dimension != spec.target_dimension:
            return False, f"{label} mass dimension"
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
    """Exact structural Laurent polynomial, preserving matrix order.

    Scalar products commute. Mixed chains reverse with (-1)^number_of_F;
    traces are cyclic and obey the same signed reversal. No other matrix
    permutation is allowed. Unsupported scalars fail before canonicalization.
    """
    tree = parse_expression(expr)

    def atom(node) -> tuple[int, tuple]:
        parts = node.parts
        trace = parts[-1] is sqed._DotChain._TR
        if trace:
            labels = tuple(part.idx for part in parts[:-1])
            rotations = [labels[i:] + labels[:i] for i in range(len(labels))]
            reversed_labels = labels[::-1]
            reversed_rotations = [reversed_labels[i:] + reversed_labels[:i] for i in range(len(labels))]
            forward, backward = min(rotations), min(reversed_rotations)
            reverse_sign = (-1) ** len(labels)
            if reverse_sign == -1 and forward == backward:
                return 0, ("trace", forward)
            return (1, ("trace", forward)) if forward <= backward else (reverse_sign, ("trace", backward))
        if any(part.tag == "F" for part in parts):
            labels = tuple(part.idx for part in parts[1:-1])
            a, b = parts[0].idx, parts[-1].idx
            forward, backward = (a, labels, b), (b, labels[::-1], a)
            sign = (-1) ** len(labels)
            if forward == backward and sign == -1:
                return 0, ("chain", forward)
            chosen, coefficient = (forward, 1) if forward <= backward else (backward, sign)
            if len(labels) == 1:
                return coefficient, ("X", chosen[1][0], chosen[0], chosen[2])
            return coefficient, ("chain", *chosen)
        endpoints = tuple(sorted((part.tag, part.idx) for part in parts))
        if all(tag == "p" for tag, _ in endpoints):
            return 1, ("s", endpoints[0][1], endpoints[1][1])
        return 1, ("dot", *endpoints)

    def multiply(left, right):
        if len(left) * len(right) > 16384:
            raise ValueError("Structural expansion exceeds 16384 terms")
        result = defaultdict(Fraction)
        for lp, lc in left.items():
            for rp, rc in right.items():
                powers = Counter(dict(lp))
                powers.update(dict(rp))
                key = tuple(sorted((factor, power) for factor, power in powers.items() if power))
                result[key] += lc * rc
        return {key: coefficient for key, coefficient in result.items() if coefficient}

    def invert(poly):
        if not poly:
            raise ValueError("Division by an identically zero expression")
        if len(poly) == 1:
            powers, coefficient = next(iter(poly.items()))
            return {tuple((factor, -power) for factor, power in powers): 1 / coefficient}
        ordered = tuple(sorted(poly.items()))
        scale = ordered[0][1]
        normalized = tuple((powers, coefficient / scale) for powers, coefficient in ordered)
        return {((("sum", normalized), -1),): 1 / scale}

    def polynomial(node):
        constant = _constant_value(node)
        if constant is not None:
            return {(): constant} if constant else {}
        if isinstance(node, sqed._DotChain):
            coefficient, factor = atom(node)
            return {((factor, 1),): Fraction(coefficient)} if coefficient else {}
        if isinstance(node, sqed._UnaryOp):
            return {powers: -coefficient for powers, coefficient in polynomial(node.operand).items()}
        if isinstance(node, sqed._BinOp):
            left = polynomial(node.left)
            if node.op == "**":
                power = _integer_power(node.right)
                if power < 0:
                    left, power = invert(left), -power
                result = {(): Fraction(1)}
                for _ in range(power):
                    result = multiply(result, left)
                return result
            right = polynomial(node.right)
            if node.op in {"+", "-"}:
                result = defaultdict(Fraction, left)
                for powers, coefficient in right.items():
                    result[powers] += coefficient if node.op == "+" else -coefficient
                return {powers: coefficient for powers, coefficient in result.items() if coefficient}
            return multiply(left, invert(right) if node.op == "/" else right)
        raise ValueError(f"Unsupported compact-signature node: {type(node).__name__}")

    output = []
    for powers, coefficient in polynomial(tree).items():
        numerator, denominator = [], []
        if abs(coefficient.numerator) != 1:
            numerator.append(("n", abs(coefficient.numerator)))
        if coefficient.denominator != 1:
            denominator.append(("n", coefficient.denominator))
        for factor, power in powers:
            (numerator if power > 0 else denominator).extend([factor] * abs(power))
        output.append((1 if coefficient > 0 else -1, tuple(sorted(numerator)), tuple(sorted(denominator))))
    return tuple(sorted(output))


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
        signature = compact_signature(expr)
    except Exception as exc:
        raise ValueError(
            f"Cannot check benchmark exclusion for {process}: invalid expression"
        ) from exc
    return signature in _BENCHMARK_BLACKLIST[process]


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
    atol: float = 1e-10,
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
            difference = abs(field_value - spinor_value)
            scale = max(abs(field_value), abs(spinor_value))
            error = difference / scale if scale else difference
            worst = max(worst, float(error))
            if not sqed.numeric_values_close(
                field_value,
                spinor_value,
                tol_abs=atol,
                tol_rel=rtol,
            ):
                raise AssertionError(
                    f"{process} field-strength fixture disagrees with paper: {error:.3e}"
                )
        errors[process] = worst
    return errors
