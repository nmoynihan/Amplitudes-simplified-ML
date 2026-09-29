"""Exact representation and unchanged-vocabulary release gates for gravity v2."""
from __future__ import annotations

import unittest
from fractions import Fraction

import sympy as sp

from data_gen import gen_data as sqed
from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_gen.data_gen_gravity.core import (
    compact_signature, eval_expression, expression_mass_dimension,
    field_strength_counts_per_term, parse_expression, polarization_degree,
)
from data_gen.data_gen_gravity.kinematics import generate_kinematics
from data_gen.ordered_gravity_gen import parenthesize_for_semantic_tokenization


def exact_expansion(expression):
    """Independent commutative algebra on general p/e dot-product symbols."""
    def visit(node):
        if isinstance(node, sqed._Num):
            value = Fraction(str(node.value))
            return sp.Rational(value.numerator, value.denominator)
        if isinstance(node, sqed._UnaryOp):
            return -visit(node.operand)
        if isinstance(node, sqed._BinOp):
            left, right = visit(node.left), visit(node.right)
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right, "/": lambda: left / right,
                    "**": lambda: left ** right}[node.op]()
        if isinstance(node, tuple) and node[0] == sqed._DOT_TAG:
            pair = sorted((f"{part[1]}{part[2]}" for part in node[1:]))
            return sp.Symbol("_".join(pair))
        raise ValueError(type(node))
    return sp.cancel(visit(sqed._expand_ast(parse_expression(expression))))


class ExactRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=4096)

    def round_trip(self, expression):
        rendered = parenthesize_for_semantic_tokenization(expression)
        ids = self.tokenizer.encode_infix(rendered)
        decoded = self.tokenizer.decode_infix(ids)
        self.assertEqual(compact_signature(expression), compact_signature(decoded))
        self.assertEqual(exact_expansion(expression), exact_expansion(decoded))
        return rendered

    def test_exact_fraction_leaves_never_pass_through_float(self):
        node = parse_expression("123456789123456789123456789.125")
        self.assertIsInstance(node.value, Fraction)
        self.assertEqual(node.value, Fraction("123456789123456789123456789.125"))
        for coefficient in ("1/3", "1/6", "-7/12", "11/24", "-12", "123", "0.5"):
            with self.subTest(coefficient=coefficient):
                self.round_trip(f"({coefficient})*(p_1 · F_4 · p_2)/(p_3 · p_5)")

    def test_numeric_only_subtrees_fold_exactly(self):
        self.assertEqual(self.round_trip("2*3"), "6")
        self.assertEqual(self.round_trip("2+3*4"), "14")
        self.round_trip("((1/3)+(1/6))*(p_1 · p_2)/(p_3 · p_4)")
        self.round_trip("((11/24)/(7/12))*(p_1 · p_2)")

    def test_ambiguous_numeric_prefix_cases_are_rejected(self):
        for expression in ("2/3", "(p_1 · p_2 + 2)/3", "1/3 + (p_1 · p_2)"):
            with self.subTest(expression=expression), self.assertRaisesRegex(ValueError, "unsafe_numeric"):
                parenthesize_for_semantic_tokenization(expression)

    def test_trace_sentinel_mixed_chains_and_powers_round_trip(self):
        examples = (
            "Tr(F_4 · F_5)^2/(24*(p_4 · p_5)^2)",
            "-7*(p_1 · F_4 · F_5 · p_2)^2/(12*(p_3 · p_4)^4)",
            "((p_1 · F_4 · p_2)*(p_3 · F_5 · p_1))^2/((p_1 · p_2)^2*(p_4 · p_5)^4)",
        )
        for expression in examples:
            with self.subTest(expression=expression):
                rendered = self.round_trip(expression)
                self.assertNotIn("^", rendered)
        self.assertIn("Tr(", self.round_trip(examples[0]))

    def test_family_degrees_dimensions_independent_of_constructor(self):
        x4, x5, t, q = "(p_1 · F_4 · p_2)", "(p_2 · F_5 · p_3)", "Tr(F_4 · F_5)", "(p_1 · F_4 · F_5 · p_2)"
        cases = ((f"{x4}^2*{x5}^2", 6), (f"{t}*{x4}*{x5}", 4),
                 (f"{t}^2", 2), (f"{q}*{x4}*{x5}", 5), (f"{q}^2", 4), (f"{t}*{q}", 3))
        for numerator, count in cases:
            expression = f"({numerator})/(p_1 · p_3)^{count}"
            with self.subTest(expression=expression):
                self.assertEqual(polarization_degree(expression), {4: 2, 5: 2})
                self.assertEqual(expression_mass_dimension(expression), 0)
                self.assertEqual(field_strength_counts_per_term(expression), [{4: 2, 5: 2}])
                extra = f"({expression})*(p_2 · p_3)/(p_2 · p_5)"
                self.assertEqual(expression_mass_dimension(extra), 0)
        self.assertEqual(expression_mass_dimension("(e_4 · e_5)"), 0)
        self.assertEqual(expression_mass_dimension("(e_4 · p_1)"), 1)
        self.assertEqual(polarization_degree(f"{x4}^3/{x4}"), {4: 2})

    def test_additive_homogeneity_in_nested_products_and_divisions(self):
        for expression in ("(p_1 · F_4 · p_2)+(p_1 · F_5 · p_2)",
                           "((p_1 · F_4 · p_2)+1)*(p_2 · F_4 · p_3)",
                           "(e_4 · e_5)+(p_4 · p_5)"):
            with self.subTest(expression=expression), self.assertRaisesRegex(ValueError, "Non-homogeneous"):
                polarization_degree(expression)

    def test_matrix_order_and_signed_reversal(self):
        cases = (("(p_1 · F_4 · p_2)", "-(p_2 · F_4 · p_1)"),
                 ("(p_1 · F_4 · F_5 · p_2)", "(p_2 · F_5 · F_4 · p_1)"),
                 ("(p_1 · F_4 · F_5 · F_4 · p_2)", "-(p_2 · F_4 · F_5 · F_4 · p_1)"),
                 ("Tr(F_1 · F_4 · F_5)", "Tr(F_4 · F_5 · F_1)"),
                 ("Tr(F_1 · F_4 · F_5)", "-Tr(F_5 · F_4 · F_1)"))
        for left, right in cases:
            self.assertEqual(compact_signature(left), compact_signature(right))
            self.assertEqual(exact_expansion(left), exact_expansion(right))
        self.assertNotEqual(compact_signature("(p_1 · F_4 · F_5 · p_2)"),
                            compact_signature("(p_1 · F_5 · F_4 · p_2)"))
        self.assertEqual(compact_signature("Tr(F_4 · F_5 · F_4)"), ())

    def test_rational_collection_and_repeated_factor_cancellation(self):
        a = "(p_1 · F_4 · p_2)"
        cases = ((f"{a}+{a}", f"2*{a}"),
                 (f"{a}/3+{a}/6", f"{a}/2"),
                 (f"(-7/12)*{a}+(11/24)*{a}", f"-{a}/8"),
                 (f"{a}^4/{a}^2", f"{a}*{a}"),
                 (f"{a}^-2", f"1/({a}*{a})"))
        for left, right in cases:
            self.assertEqual(compact_signature(left), compact_signature(right))
        self.assertEqual(compact_signature(f"{a}-{a}"), ())

    def test_trace_identity_exactly_for_general_polarizations(self):
        a, b, c, d = "(e_4 · p_5)", "(e_5 · p_4)", "(e_4 · e_5)", "(p_4 · p_5)"
        left = f"{c}*{c}/6-{c}*{a}*{b}/(3*{d})+{a}*{a}*{b}*{b}/(6*{d}*{d})"
        right = f"Tr(F_4 · F_5)*Tr(F_4 · F_5)/(24*{d}*{d})"
        self.assertEqual(sp.cancel(exact_expansion(left) - exact_expansion(right)), 0)
        self.round_trip(left)
        self.round_trip(right)

    def test_strict_parser_rejects_unsupported_or_unconsumed_nodes(self):
        kin = generate_kinematics(seed=991, graviton_legs=(4, 5))
        invalid = ("", "foo", "p_1", "F_4 · F_5", "e_4 · F_5 · p_1", "p_1 · p_2 · p_3",
                   "Tr()", "Tr(p_1)", "Tr(F_4)", "1 2", "p_1 · p_2 garbage",
                   "(p_1 · p_2", "(p_1 · p_2)^0.5", "(p_1 · p_2)/(1-1)", "p_9 · p_1")
        for expression in invalid:
            with self.subTest(expression=expression):
                for function in (compact_signature, parse_expression, lambda x: eval_expression(x, kin)):
                    with self.assertRaises((ValueError, ZeroDivisionError)):
                        function(expression)


if __name__ == "__main__":
    unittest.main()
