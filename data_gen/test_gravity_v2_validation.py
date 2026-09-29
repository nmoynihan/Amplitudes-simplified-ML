"""Scientific regression gates for the versioned exact v2 validator."""
from fractions import Fraction
import unittest
import sympy as sp
from data_gen.data_gen_gravity.core import eval_expression
from data_gen.data_gen_gravity.v2_validation import (
    FrozenFamilyGuard, NumericalContext, PairValidator, ParseError,
    canonical_family, check_process, exact_equivalent, parse_exact,
    parse_laurent, validate_token_roundtrip,
)
D = "(p_4 · p_5)"
T = "Tr(F_4 · F_5)"
X = "(p_1 · F_4 · p_2)"
Y = "(p_1 · F_5 · p_3)"

class TestStrictExactParser(unittest.TestCase):
    def test_exact_coefficients(self):
        for value in (Fraction(1, 3), Fraction(1, 6), Fraction(-7, 12), Fraction(11, 24), Fraction(-123), Fraction(17)):
            text = f"({value.numerator}*{X})/{value.denominator}"
            self.assertEqual(list(parse_laurent(text).values()), [value])
            self.assertFalse(parse_exact(text).atoms(sp.Float))
        self.assertEqual(parse_laurent(f"(0.125*{X})"), parse_laurent(f"{X}/8"))

    def test_full_consumption_and_malformed_rejection(self):
        for text in ("", "junk", "p_1", "F_4", "p_1 p_2", "p_1·p_2)junk", "(p_1·p_2", "Tr()", "Tr(F_4)", "Tr(F_4·p_5)", "p_1··p_2", "p_1·F_4·F_5", "p_1·p_2 garbage", "p_1·p_6", "1/0", "1/(2-2)", "(p_1·p_2)^(1/2)", "p_1·p_2 +"):
            with self.subTest(text=text):
                with self.assertRaises((ParseError, ZeroDivisionError)): parse_laurent(text)

    def test_numeric_boundary_cannot_silently_change(self):
        for before, after in (("2/3", "23"), ("2*3", "23"), ("(2*3)*(p_1·p_2)", "23*(p_1·p_2)")):
            self.assertFalse(validate_token_roundtrip(before, after)["ok"])
        self.assertTrue(validate_token_roundtrip(f"(2*{X})/(3*{D})", "2*p_1·F_4·p_2/(3*p_4·p_5)")["ok"])

    def test_sum_denominator_only_sympy_parser(self):
        with self.assertRaises(ParseError): parse_laurent("1/((p_1·p_2)+(p_1·p_3))")
        self.assertEqual(parse_exact("1/((p_1·p_2)+(p_1·p_3))"), 1/(sp.Symbol("d_1_2")+sp.Symbol("d_1_3")))

class TestScientificAlgebra(unittest.TestCase):
    def test_trace_square_identity_general_polarization(self):
        a, b, c = "(e_4·p_5)", "(e_5·p_4)", "(e_4·e_5)"
        left = f"{c}*{c}/6 - {c}*{a}*{b}/(3*{D}) + {a}*{a}*{b}*{b}/(6*{D}*{D})"
        right = f"{T}*{T}/(24*{D}*{D})"
        self.assertTrue(exact_equivalent(left, right))
        result = PairValidator(seed=400001).validate(right, left, "3s2h")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["samples"], 15)
        self.assertEqual(result["general_polarization_samples"], 3)

    def test_field_strength_conventions(self):
        text = "(p_1·F_4·F_5·p_2)"
        expanded = "(p_1·p_4)*(e_4·p_5)*(e_5·p_2) - (p_1·p_4)*(e_4·e_5)*(p_5·p_2) - (p_1·e_4)*(p_4·p_5)*(e_5·p_2) + (p_1·e_4)*(p_4·e_5)*(p_5·p_2)"
        self.assertTrue(exact_equivalent(text, expanded))
        self.assertTrue(exact_equivalent(X, "(p_1·p_4)*(e_4·p_2) - (p_1·e_4)*(p_4·p_2)"))
        ctx = NumericalContext("3s2h", seed=501)
        for compact in (T, X, text):
            self.assertAlmostEqual(abs(ctx.evaluate(compact)[0]-eval_expression(compact, ctx.points[0])), 0, places=9)

    def test_order_reversal_and_cyclic_signs(self):
        self.assertEqual(parse_laurent(X), parse_laurent("-(p_2·F_4·p_1)"))
        self.assertEqual(parse_laurent("p_1·F_4·F_5·p_2"), parse_laurent("p_2·F_5·F_4·p_1"))
        self.assertNotEqual(parse_laurent("p_1·F_4·F_5·p_2"), parse_laurent("p_1·F_5·F_4·p_2"))
        self.assertEqual(parse_laurent(T), parse_laurent("Tr(F_5·F_4)"))
        self.assertEqual(parse_laurent("Tr(F_1·F_2·F_3)"), parse_laurent("-Tr(F_3·F_2·F_1)"))
        self.assertTrue(exact_equivalent("Tr(F_1·F_2·F_3)", "-Tr(F_3·F_2·F_1)"))
        self.assertEqual(parse_laurent("Tr(F_4·F_4·F_5)"), {})

    def test_all_six_numerator_families(self):
        q = "(p_1·F_4·F_5·p_2)"
        families = ((f"{X}^2*{Y}^2",6),(f"{T}*{X}*{Y}",4),(f"{T}^2",2),(f"{q}*{X}*{Y}",5),(f"{q}^2",4),(f"{T}*{q}",3))
        for numerator, count in families:
            self.assertEqual(check_process(f"({numerator})/{D}^{count}", "3s2h")["polarization_degrees"], {4:2,5:2})
        self.assertEqual(check_process(f"{Y}^2/{D}^4", "4s1h")["mass_dimension"], -2)
        for text in (f"{X}*{Y}/{D}^3",f"{T}^2/{D}",f"{T}^2/{D}^2 + {T}/{D}"):
            with self.assertRaises(ValueError): check_process(text, "3s2h")

    def test_exact_transformations(self):
        a=f"{T}^2"
        for left,right in ((f"({a}+{a})/{D}^2",f"2*{a}/{D}^2"),(f"{a}*(p_1·p_2)/({D}^2*(p_1·p_2))",f"{a}/{D}^2"),(f"{a}*((p_1·p_2)+(p_1·p_3))/({D}^2*(p_1·p_2)*(p_1·p_3))",f"{a}/({D}^2*(p_1·p_2))+{a}/({D}^2*(p_1·p_3))")):
            self.assertTrue(exact_equivalent(left,right))

    def test_numerical_mismatch_and_parse_failure_separate(self):
        validator=PairValidator(seed=12011)
        valid=f"{T}^2/{D}^2"
        self.assertEqual(validator.validate(valid,"junk","3s2h")["reason"],"parse_error")
        self.assertEqual(validator.validate(valid,f"2*({valid})","3s2h")["reason"],"numerical_mismatch")
        self.assertEqual(validator.validate(valid,valid,"4s1h")["reason"],"process_degree")

class TestFamilyGuards(unittest.TestCase):
    def test_skeleton_ignores_independent_coefficients_and_species_labels(self):
        a=f"{X}^2*{Y}^2/{D}^6 + {T}^2/{D}^2"
        b="7*(p_2·F_4·p_1)^2*(p_2·F_5·p_3)^2/(p_4·p_5)^6 - 11*Tr(F_5·F_4)^2/(p_5·p_4)^2"
        self.assertEqual(canonical_family(a,"3s2h"),canonical_family(b,"3s2h"))
        self.assertNotEqual(canonical_family(a,"3s2h",ignore_coefficients=False),canonical_family(b,"3s2h",ignore_coefficients=False))

    def test_scale_expansion_and_relabel_are_reserved(self):
        original=f"{T}^2/(24*{D}^2)"
        guard=FrozenFamilyGuard({"3s2h":[original]},seed=71213)
        self.assertTrue(guard.is_reserved(f"-7*({original})","3s2h"))
        self.assertTrue(guard.is_reserved("((e_4·p_5)*(e_5·p_4)-(p_4·p_5)*(e_4·e_5))^2/(6*(p_4·p_5)^2)","3s2h"))
        self.assertFalse(guard.is_reserved(f"{X}^2*{Y}^2/{D}^6","3s2h"))

    def test_adding_heldout_families_reserves_new_skeletons(self):
        guard = FrozenFamilyGuard({"3s2h": [f"{T}^2/{D}^2"]}, seed=71213)
        heldout = f"{X}^2*{Y}^2/{D}^6"
        self.assertFalse(guard.is_reserved(heldout, "3s2h"))
        guard.add_references({"3s2h": [heldout]})
        self.assertTrue(guard.is_reserved(f"-11*({heldout})/24", "3s2h"))
        expanded = parse_laurent(heldout, expand_f=True)
        self.assertTrue(guard.matches_fingerprint(guard.fingerprint(expanded, "3s2h"), "3s2h"))

    def test_frozen_independent_coefficient_descendants_conservatively_reserved(self):
        original = f"{T}^2/{D}^2 + {X}^2*{Y}^2/{D}^6"
        guard = FrozenFamilyGuard({"3s2h": [original]}, seed=71213)
        self.assertTrue(guard.is_reserved(f"3*{T}^2/{D}^2 - 11*{X}^2*{Y}^2/{D}^6", "3s2h"))

    def test_general_polarization_checks_are_included(self):
        ctx=NumericalContext("3s2h",seed=5167)
        values=ctx.atom(("ee",4,5))
        self.assertTrue(any(abs(values[i])>1e-4 for i,label in enumerate(ctx.labels) if "general" in label))

if __name__ == "__main__": unittest.main()
