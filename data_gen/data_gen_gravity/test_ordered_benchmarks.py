"""Independent algebraic and physical checks for the ordered benchmark seeds."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import re
import unittest

import sympy as sp

from ..Tokenizer import ScatteringAmplitudeTokenizer
from .core import (
    BENCHMARKS,
    PROCESS_SPECS,
    eval_expression,
    expression_mass_dimension,
    field_strength_counts_per_term,
    paper_spinor_value,
)
from .kinematics import generate_kinematics, with_references
from .ordered_benchmarks import (
    full_permutation_sum,
    ordered_seed,
    permutation_orders,
    reconstruction_terms,
    reconstruct_benchmark,
    verify_reconstruction,
)
from .verify_ordered_chy import verify_chy


def _symbolic_compact(expression: str) -> sp.Expr:
    """Parse contractions independently, imposing only F antisymmetry and d symmetry."""
    def field(match: re.Match[str]) -> str:
        a, h, b = map(int, match.groups())
        if a == b:
            return "0"
        sign = "-" if a > b else ""
        a, b = sorted((a, b))
        return f"({sign}x_{h}_{a}_{b})"

    def invariant(match: re.Match[str]) -> str:
        a, b = sorted(map(int, match.groups()))
        return f"d_{a}_{b}"

    converted = re.sub(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)", field, expression)
    converted = re.sub(r"p_(\d+)\s*·\s*p_(\d+)", invariant, converted)
    if re.search(r"[peF]_\d|·", converted):
        raise AssertionError(f"Unsupported compact expression: {expression}")
    return sp.sympify(converted, rational=True)


def _relabel_fixture(expression: str, order: tuple[int, ...]) -> str:
    return re.sub(
        r"([peF])_(\d+)\b",
        lambda match: f"{match[1]}_{order[int(match[2]) - 1]}",
        expression,
    )


def _allowed_orders(process: str) -> list[tuple[int, ...]]:
    if process == "3s2h":
        return [a + h for a in itertools.permutations((1, 2, 3))
                for h in itertools.permutations((4, 5))]
    channel = {frozenset((1, 4)), frozenset((2, 3))}
    return [a + (5,) for a in itertools.permutations((1, 2, 3, 4))
            if {frozenset((a[0], a[3])), frozenset((a[1], a[2]))} == channel]


def _exact_3s2h_substitutions(expression: sp.Expr) -> dict[sp.Symbol, sp.Expr]:
    """A nonsingular rational conserved massless spinor point, with ++ helicities."""
    lam = {1: (1, 2), 2: (1, 3), 3: (1, 5), 4: (1, 0), 5: (0, 1)}
    tilde = {1: (1, 2), 2: (3, 7), 3: (5, 13), 4: (-9, -22), 5: (-36, -90)}

    def bracket(spinors: dict, a: int, b: int) -> int:
        x, y = spinors[a], spinors[b]
        return x[0] * y[1] - x[1] * y[0]

    result = {}
    for symbol in expression.free_symbols:
        parts = str(symbol).split("_")
        if parts[0] == "d":
            a, b = map(int, parts[1:])
            result[symbol] = sp.Rational(1, 2) * bracket(lam, a, b) * bracket(tilde, a, b)
        else:
            h, a, b = map(int, parts[1:])
            result[symbol] = (-sp.sqrt(2) / 4 * bracket(lam, a, b)
                              * bracket(tilde, a, h) * bracket(tilde, b, h))
    return result


class OrderedBenchmarkTests(unittest.TestCase):
    def test_reconstruction_is_exact_before_on_shell_constraints(self) -> None:
        for process in PROCESS_SPECS:
            orders = _allowed_orders(process)
            self.assertEqual(len(orders), 12 if process == "3s2h" else 8)
            for order in orders:
                with self.subTest(process=process, order=order):
                    expected = _symbolic_compact(_relabel_fixture(BENCHMARKS[process], order))
                    terms = reconstruction_terms(process, order=order)
                    self.assertEqual(len(terms), 2)
                    self.assertEqual(terms[0], ordered_seed(process, order=order))
                    summed = sum(map(_symbolic_compact, terms))
                    self.assertEqual(sp.cancel(summed - expected), 0)
                    reconstructed = _symbolic_compact(reconstruct_benchmark(process, order=order))
                    self.assertEqual(sp.cancel(reconstructed - expected), 0)

    def test_two_orderings_match_independent_paper_formula(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            for seed in (17, 31, 73):
                with self.subTest(process=process, seed=seed):
                    kin = generate_kinematics(seed=seed, graviton_legs=spec.graviton_legs)
                    expected = paper_spinor_value(process, kin)
                    actual = sum(eval_expression(term, kin) for term in reconstruction_terms(process))
                    self.assertLessEqual(abs(actual - expected), 1e-9 * max(1, abs(expected)))

    def test_report_factors_convert_repository_values_to_paper_values(self) -> None:
        reports = {
            "reconstruction": verify_reconstruction((17,)),
            "chy": verify_chy((17,))["checks"],
        }
        for filename in ("ordered_benchmark_verification.json", "ordered_chy_verification.json"):
            reports[filename] = json.loads(Path(__file__).with_name(filename).read_text())["checks"]
        for process, spec in PROCESS_SPECS.items():
            kin = generate_kinematics(seed=17, graviton_legs=spec.graviton_legs)
            repository_value = eval_expression(BENCHMARKS[process], kin)
            if process == "3s2h":
                paper_value = paper_spinor_value(process, kin)
            else:
                # Eq. (4.8) itself, without paper_spinor_value's repository factor 2.
                a, q = kin.angle, kin.square
                paper_value = q(2, 5) * q(4, 5) / (a(1, 5) * a(3, 5) * q(1, 4) * q(2, 3)) * (
                    1 + a(1, 4) * a(3, 4) * q(1, 4) / (a(2, 3) * a(4, 5) * q(2, 5))
                    - a(1, 2) * a(2, 3) * q(2, 3) / (a(1, 4) * a(2, 5) * q(4, 5))
                )
            self.assertGreater(abs(paper_value), 1e-6)
            for source, checks in reports.items():
                with self.subTest(process=process, source=source):
                    factor = checks[process]["repository_to_paper_factor"]
                    self.assertEqual(factor, 1 if process == "3s2h" else 0.5)
                    self.assertLessEqual(
                        abs(factor * repository_value - paper_value),
                        1e-9 * max(1, abs(paper_value)),
                    )

    def test_each_seed_is_gauge_invariant(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            kin = generate_kinematics(seed=91, graviton_legs=spec.graviton_legs)
            points = [with_references(kin, spec.graviton_legs, reference_mode=mode)
                      for mode in ("first", "last", "random")]
            points.append(with_references(
                kin, spec.graviton_legs, reference_mode="cyclic",
                gauge_shifts={h: 0.3 - 0.2j for h in spec.graviton_legs},
            ))
            for term in reconstruction_terms(process):
                expected = eval_expression(term, kin)
                for point in points:
                    actual = eval_expression(term, point)
                    self.assertLessEqual(abs(actual - expected), 1e-9 * max(1, abs(expected)))

    def test_3s2h_exact_rational_point_is_nontrivial(self) -> None:
        terms = [_symbolic_compact(term) for term in reconstruction_terms("3s2h")]
        values = [sp.simplify(term.subs(_exact_3s2h_substitutions(term))) for term in terms]
        self.assertEqual(values, [-sp.Integer(72), sp.Rational(252, 5)])
        self.assertEqual(sum(values), -sp.Rational(108, 5))

    def test_3s2h_scalar_symmetry_exact_on_dense_positive_helicity_patch(self) -> None:
        # A generic SL(2) frame and little-group gauge; momentum conservation
        # determines the final two tilde spinors. No numerical sampling enters.
        x, y, z, a, b, c, d, e, f = sp.symbols("x y z a b c d e f")
        lam = {1: (1, x), 2: (1, y), 3: (1, z), 4: (1, 0), 5: (0, 1)}
        tilde = {1: (a, b), 2: (c, d), 3: (e, f),
                 4: (-a - c - e, -b - d - f),
                 5: (-x*a - y*c - z*e, -x*b - y*d - z*f)}

        def bracket(spinors: dict, i: int, j: int) -> sp.Expr:
            u, v = spinors[i], spinors[j]
            return u[0]*v[1] - u[1]*v[0]

        def seed(i: int, j: int, k: int, h: int, l: int) -> sp.Expr:
            A = lambda r, s: bracket(lam, r, s)
            Q = lambda r, s: bracket(tilde, r, s)
            return (A(i,j)*A(i,k)*A(j,k)*Q(i,h)*Q(k,l)
                    / (A(i,h)*A(j,h)*A(j,l)*A(k,l)*A(h,l)))

        def amplitude(i: int, j: int, k: int) -> sp.Expr:
            return seed(i,j,k,4,5) + seed(i,j,k,5,4)

        base = amplitude(1,2,3)
        # These adjacent transpositions generate the full scalar S3.
        for order in ((2,1,3), (1,3,2)):
            self.assertEqual(sp.factor(base - amplitude(*order)), 0)
        self.assertEqual(sp.factor(seed(1,2,3,4,5) - seed(3,2,1,5,4)), 0)

    def test_4s1h_channel_symmetry_exact_for_transverse_polarization(self) -> None:
        # General invariant solution of masslessness, momentum conservation,
        # and e5.p5=0. Unlike the 3s2h proof, no helicity identity is required.
        a, b, c, v, w, z1, z2, z3 = sp.symbols("a b c v w z1 z2 z3")
        r = {1: a, 2: b, 3: c, 4: -a-b-c}
        z = {1: z1, 2: z2, 3: z3, 4: -z1-z2-z3}
        dots = {(1,2): w, (1,3): -a-w-v-b-c, (1,4): v+b+c,
                (2,3): v, (2,4): -b-w-v, (3,4): a+b+w}

        def evaluate(expression: str) -> sp.Expr:
            parsed = _symbolic_compact(expression)
            substitutions = {}
            for symbol in parsed.free_symbols:
                parts = str(symbol).split("_")
                if parts[0] == "d":
                    i, j = map(int, parts[1:])
                    substitutions[symbol] = r[i] if j == 5 else dots[(i,j)]
                else:
                    h, i, j = map(int, parts[1:])
                    self.assertEqual(h, 5)
                    substitutions[symbol] = r[i]*z[j] - z[i]*r[j]
            return parsed.subs(substitutions, simultaneous=True)

        base = evaluate(BENCHMARKS["4s1h"])
        # Swap within either scalar pair, or exchange the two pairs.
        for order in ((4,2,3,1,5), (1,3,2,4,5), (3,4,1,2,5)):
            candidate = evaluate(_relabel_fixture(BENCHMARKS["4s1h"], order))
            self.assertEqual(sp.factor(candidate - base), 0)

    def test_weighted_full_group_sum_matches_paper(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            self.assertEqual(set(permutation_orders(process)), set(_allowed_orders(process)))
            for seed in (17, 31):
                kin = generate_kinematics(seed=seed, graviton_legs=spec.graviton_legs)
                expected = paper_spinor_value(process, kin)
                actual = eval_expression(full_permutation_sum(process), kin)
                self.assertLessEqual(abs(actual - expected), 1e-9 * max(1, abs(expected)))

    def test_4s1h_seed_is_not_half_the_fixture(self) -> None:
        kin = generate_kinematics(seed=17, graviton_legs=(5,))
        first, second = [eval_expression(term, kin) for term in reconstruction_terms("4s1h")]
        self.assertGreater(abs(first - second), 1e-3 * max(abs(first), abs(second)))

    def test_seed_dimensions_multiplicities_and_tokenization(self) -> None:
        tokenizer = ScatteringAmplitudeTokenizer(max_particles=5, max_sequence_length=4096)
        for process, spec in PROCESS_SPECS.items():
            for term in reconstruction_terms(process):
                with self.subTest(process=process, term=term):
                    self.assertEqual(expression_mass_dimension(term), spec.target_dimension)
                    expected = {h: 2 for h in spec.graviton_legs}
                    self.assertTrue(all(item == expected for item in field_strength_counts_per_term(term)))
                    self.assertNotIn(tokenizer.vocab["<UNK>"], tokenizer.encode_infix(term))

    def test_invalid_orders_and_flavor_channels_are_rejected(self) -> None:
        for operation in (ordered_seed, reconstruction_terms, reconstruct_benchmark):
            for process in PROCESS_SPECS:
                for order in ((1, 2, 3, 4), (1, 2, 3, 4, 4), (0, 2, 3, 4, 5), (5, 2, 3, 4, 1)):
                    with self.subTest(operation=operation.__name__, process=process, order=order):
                        with self.assertRaises(ValueError):
                            operation(process, order=order)
            # This permutation sends the fixed 14|23 channel to 13|24.
            with self.assertRaises(ValueError):
                operation("4s1h", order=(1, 2, 4, 3, 5))
            with self.assertRaises(ValueError):
                operation("unknown")


if __name__ == "__main__":
    unittest.main()
