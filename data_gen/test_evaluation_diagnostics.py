"""Regression tests for transparent numerical verdicts; no checkpoint required."""

from __future__ import annotations

import json
import math
import unittest
from unittest import mock

from data_testing.evaluation_diagnostics import (
    ExpressionValidationError,
    compare_expressions,
)


class NumericalDiagnosticsTests(unittest.TestCase):
    def compare(self, table, *, required=None, atol=1e-10, rtol=1e-8):
        def evaluate(expression, sample):
            value = table[sample][expression]
            if isinstance(value, Exception):
                raise value
            return value

        result = compare_expressions(
            "source", "candidate", range(len(table)), evaluator=evaluate,
            atol=atol, rtol=rtol, required_valid_samples=required,
        )
        # Strict JSON, including nonfinite values and exceptions in the evidence.
        json.dumps(result, allow_nan=False)
        return result

    def test_copy_is_evaluated_at_every_point(self):
        callback = mock.Mock(side_effect=lambda expression, sample: complex(sample, -sample))
        result = compare_expressions(
            "copy", "copy", [1, 2, 3], evaluator=callback, atol=1e-10, rtol=1e-8,
        )
        self.assertTrue(result["equivalent"])
        self.assertTrue(result["conclusive"])
        self.assertEqual(result["status"], "equivalent")
        self.assertEqual(callback.call_count, 6)
        self.assertEqual(result["counts"]["valid_samples"], 3)
        self.assertIsNone(result["first_failure"])

    def test_mismatch_keeps_complex_values_and_defined_errors(self):
        result = self.compare([
            {"source": 1 + 2j, "candidate": 1 + 2j},
            {"source": 3 + 4j, "candidate": 0},
            {"source": 5, "candidate": 6},
        ], atol=0.1, rtol=0.01)
        self.assertEqual(result["status"], "numerical_mismatch")
        self.assertTrue(result["conclusive"])
        self.assertFalse(result["equivalent"])
        first = result["first_mismatch"]
        self.assertEqual(first["sample_index"], 1)
        self.assertEqual(first["source"], {"real": 3.0, "imag": 4.0})
        self.assertEqual(first["candidate"], {"real": 0.0, "imag": 0.0})
        self.assertEqual(first["absolute_error"], 5)
        self.assertEqual(first["relative_error"], 1)
        self.assertEqual(first["allowed_error"], 0.1)
        self.assertEqual(first["scaled_error"], 50)
        self.assertEqual(result["counts"]["attempted_samples"], 3)
        self.assertEqual(result["counts"]["mismatched_samples"], 2)

    def test_near_zero_never_receives_unit_relative_floor(self):
        result = self.compare([{"source": 0, "candidate": 1e-9}])
        self.assertEqual(result["status"], "numerical_mismatch")
        self.assertEqual(result["first_mismatch"]["relative_error"], 1)
        close = self.compare([{"source": 0, "candidate": 0.5e-10}])
        self.assertTrue(close["equivalent"])

    def test_zero_tolerance_metrics_are_json_serializable(self):
        match = self.compare([{"source": 0, "candidate": 0}], atol=0, rtol=0)
        self.assertEqual(match["checks"][0]["relative_error"], 0)
        self.assertEqual(match["checks"][0]["scaled_error"], 0)
        mismatch = self.compare([{"source": 0, "candidate": 1}], atol=0, rtol=0)
        self.assertEqual(mismatch["checks"][0]["scaled_error"], "inf")

    def test_empty_samples_are_inconclusive_not_vacuously_equivalent(self):
        result = self.compare([])
        self.assertEqual(result["status"], "insufficient_valid_samples")
        self.assertFalse(result["equivalent"])
        self.assertFalse(result["conclusive"])
        self.assertEqual(result["counts"]["requested_valid_samples"], 1)

    def test_nonfinite_source_requires_explicit_bounded_replacements(self):
        table = [
            {"source": math.nan, "candidate": math.nan},
            {"source": 2, "candidate": 2},
            {"source": 3, "candidate": 3},
        ]
        strict = self.compare(table)
        self.assertEqual(strict["status"], "insufficient_valid_samples")
        self.assertEqual(strict["first_failure"]["status"], "nonfinite_value")
        self.assertEqual(strict["first_failure"]["role"], "source")
        self.assertEqual(strict["first_failure"]["value"]["real"], "nan")
        self.assertEqual(strict["counts"]["candidate_error_samples"], 0)
        replacement = self.compare(table, required=2)
        self.assertTrue(replacement["equivalent"])
        self.assertEqual(replacement["counts"]["attempted_samples"], 3)
        self.assertEqual(replacement["counts"]["source_error_samples"], 1)

    def test_singular_source_is_recorded_with_concise_exception(self):
        result = self.compare([
            {"source": ZeroDivisionError("source pole"), "candidate": 0},
            {"source": 2, "candidate": 2},
        ])
        self.assertEqual(result["status"], "insufficient_valid_samples")
        self.assertEqual(result["first_failure"]["status"], "numerical_evaluation_error")
        self.assertEqual(result["first_failure"]["error_type"], "ZeroDivisionError")
        self.assertEqual(result["first_failure"]["reason"], "source pole")

    def test_candidate_only_error_cannot_be_resampled_away(self):
        result = self.compare([
            {"source": 1, "candidate": 1},
            {"source": 2, "candidate": ZeroDivisionError("candidate pole")},
            {"source": 3, "candidate": 3},
        ], required=1)
        self.assertEqual(result["status"], "numerical_evaluation_error")
        self.assertFalse(result["equivalent"])
        self.assertFalse(result["conclusive"])
        self.assertEqual(result["counts"]["candidate_error_samples"], 1)
        self.assertEqual(result["counts"]["equivalent_samples"], 2)
        self.assertEqual(result["first_failure"]["source"]["real"], 2)

    def test_source_syntax_and_unexpected_callback_errors_are_not_replaceable(self):
        for error, status in (
            (KeyError("unknown source momentum"), "unsupported_expression"),
            (SyntaxError("invalid source"), "invalid_syntax"),
            (RuntimeError("broken callback"), "numerical_evaluation_error"),
        ):
            with self.subTest(error=error):
                result = self.compare([
                    {"source": error, "candidate": 0},
                    {"source": 1, "candidate": 1},
                ], required=1)
                self.assertEqual(result["status"], status)
                self.assertFalse(result["equivalent"])
                self.assertFalse(result["first_failure"]["replaceable_source_sample"])

    def test_candidate_nonfinite_is_not_ordinary_inequivalence(self):
        for bad in (math.inf, -math.inf, math.nan, complex(1, math.inf)):
            with self.subTest(value=bad):
                result = self.compare([
                    {"source": 1, "candidate": 1},
                    {"source": 2, "candidate": bad},
                ], required=1)
                self.assertEqual(result["status"], "nonfinite_value")
                self.assertEqual(result["counts"]["mismatched_samples"], 0)
                self.assertFalse(result["equivalent"])

    def test_any_valid_counterexample_outweighs_inconclusive_samples(self):
        result = self.compare([
            {"source": math.nan, "candidate": 0},
            {"source": 2, "candidate": 3},
            {"source": 4, "candidate": ValueError("domain error")},
        ])
        self.assertEqual(result["status"], "numerical_mismatch")
        self.assertTrue(result["conclusive"])
        self.assertEqual(result["first_failure"]["sample_index"], 0)
        self.assertEqual(result["first_mismatch"]["sample_index"], 1)
        self.assertEqual(len(result["failures"]), 3)

    def test_strict_validator_rejects_both_expressions_before_evaluation(self):
        callback = mock.Mock()
        def validate(expression):
            if expression == "source":
                raise SyntaxError("unconsumed suffix")
            raise KeyError("e_1 is not a graviton")

        result = compare_expressions(
            "source", "candidate", [0], evaluator=callback, validator=validate,
            atol=1e-10, rtol=1e-8,
        )
        callback.assert_not_called()
        self.assertEqual(result["status"], "invalid_syntax")
        self.assertEqual(result["failures"][1]["status"], "unsupported_expression")
        self.assertEqual(result["counts"]["attempted_samples"], 0)
        json.dumps(result, allow_nan=False)

    def test_explicit_parser_error_category_is_preserved(self):
        def validate(expression):
            raise ExpressionValidationError("unsupported_expression", "free vector")

        result = compare_expressions(
            "p_1", "p_1", [0], evaluator=lambda *_: 0, validator=validate,
            atol=1e-10, rtol=1e-8,
        )
        self.assertEqual(result["status"], "unsupported_expression")
        self.assertEqual(result["first_failure"]["reason"], "free vector")

    def test_bad_scalar_return_is_an_error(self):
        for value in ([1, 2], "0", None):
            with self.subTest(value=value):
                result = self.compare([{"source": 1, "candidate": value}])
                self.assertEqual(result["status"], "numerical_evaluation_error")
                self.assertEqual(result["first_failure"]["error_type"], "TypeError")

    def test_requested_count_is_not_reduced_to_available_count(self):
        result = self.compare([{"source": 1, "candidate": 1}], required=2)
        self.assertEqual(result["status"], "insufficient_valid_samples")
        self.assertFalse(result["equivalent"])

    def test_invalid_tolerances_and_counts_are_configuration_errors(self):
        for kwargs in ({"atol": -1}, {"rtol": math.inf}, {"atol": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compare([], **kwargs)
        for required in (0, -1, 1.5, True):
            with self.subTest(required=required), self.assertRaises(ValueError):
                self.compare([], required=required)


class StrictGravityDiagnosticsTests(unittest.TestCase):
    """Exercise real contractions and the production strict scalar boundary."""

    @classmethod
    def setUpClass(cls):
        from data_gen.data_gen_gravity.v2_validation import NumericalContext
        cls.points = NumericalContext("3s2h", seed=42071).points[:3]

    def compare(self, source, candidate, *, process="3s2h"):
        from data_gen.data_gen_gravity.core import eval_expression
        from data_testing.evaluate_model import validate_gravity_expression
        return compare_expressions(
            source, candidate, self.points, evaluator=eval_expression,
            validator=lambda expression: validate_gravity_expression(expression, process),
            atol=2e-9, rtol=2e-8,
        )

    def test_rational_trace_repeated_pole_identity(self):
        d, a, b, c = "(p_4·p_5)", "(e_4·p_5)", "(e_5·p_4)", "(e_4·e_5)"
        source = f"{c}*{c}/6-{c}*{a}*{b}/(3*{d})+{a}*{a}*{b}*{b}/(6*{d}*{d})"
        candidate = f"Tr(F_4·F_5)*Tr(F_4·F_5)/(24*{d}*{d})"
        result = self.compare(source, candidate)
        self.assertTrue(result["equivalent"], result)

    def test_mixed_chain_order_remains_significant(self):
        source = "p_1·F_4·F_5·p_2"
        reversed_endpoints = "p_2·F_5·F_4·p_1"
        swapped = "p_1·F_5·F_4·p_2"
        self.assertTrue(self.compare(source, reversed_endpoints)["equivalent"])
        self.assertEqual(self.compare(source, swapped)["status"], "numerical_mismatch")

    def test_unsupported_process_symbols_and_free_vectors_are_rejected(self):
        for candidate in ("p_6·p_1", "e_1·p_2", "F_1", "p_1", "p_1·p_2 @"):
            with self.subTest(candidate=candidate):
                result = self.compare("0", candidate)
                self.assertFalse(result["equivalent"])
                self.assertIn(result["status"], {"invalid_expression", "unsupported_expression", "invalid_syntax"})
                self.assertEqual(result["counts"]["attempted_samples"], 0)
        self.assertEqual(
            self.compare("0", "p_1·F_4·p_2", process="4s1h")["status"],
            "unsupported_expression",
        )

    def test_full_parser_consumption_is_required(self):
        for malformed in ("", "p_1·p_2 p_3·p_4", "p_1·p_2)", "p_1·p_2 +"):
            with self.subTest(malformed=malformed):
                result = self.compare("0", malformed)
                self.assertFalse(result["equivalent"])
                self.assertEqual(result["counts"]["attempted_samples"], 0)


class NumericalComparisonWrapperTests(unittest.TestCase):
    def setUp(self):
        from data_testing import evaluate_model
        self.evaluator = evaluate_model
        self.settings = mock.patch.multiple(
            evaluate_model,
            NUMERIC_BACKEND="sqed",
            N_PARTICLES=4,
            NUMERIC_TOL_ABS=1e-10,
            NUMERIC_TOL_REL=1e-8,
        )
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def test_boolean_wrapper_delegates_without_losing_detailed_api(self):
        with mock.patch.object(self.evaluator, "numerical_comparison", return_value={"equivalent": True}) as check:
            self.assertTrue(self.evaluator.numerically_equivalent_exprs("source", "candidate", []))
        check.assert_called_once_with("source", "candidate", [], gravity_process=None)

    def test_no_backend_accepts_empty_kinematics(self):
        for backend, points, process in (("sqed", [], None), ("ym", [], None), ("gravity", {"3s2h": []}, "3s2h")):
            with self.subTest(backend=backend), mock.patch.object(self.evaluator, "NUMERIC_BACKEND", backend):
                result = self.evaluator.numerical_comparison("0", "0", points, gravity_process=process)
                self.assertEqual(result["status"], "insufficient_valid_samples")
                self.assertFalse(result["equivalent"])

    def test_process_and_cache_configuration_errors_are_explicit(self):
        with mock.patch.object(self.evaluator, "NUMERIC_BACKEND", "gravity"):
            missing_process = self.evaluator.numerical_comparison("0", "0", {})
            self.assertEqual(missing_process["status"], "unsupported_expression")
            self.assertEqual(missing_process["first_failure"]["phase"], "configuration")
            wrong_cache = self.evaluator.numerical_comparison("0", "0", [], gravity_process="3s2h")
            self.assertEqual(wrong_cache["status"], "invalid_configuration")
        wrong_cache = self.evaluator.numerical_comparison("0", "0", {})
        self.assertEqual(wrong_cache["status"], "invalid_configuration")

    def test_sqed_and_ym_reject_invalid_scalars_before_sampling(self):
        for backend in ("sqed", "ym"):
            with self.subTest(backend=backend), mock.patch.object(self.evaluator, "NUMERIC_BACKEND", backend):
                for expression in ("p_1", "p_1·p_2 trailing", "p_9·p_1", "p_1·p_2)"):
                    with self.subTest(expression=expression):
                        result = self.evaluator.numerical_comparison("0", expression, [])
                        self.assertIn(result["status"], {"unsupported_expression", "invalid_expression"})
                        self.assertEqual(result["counts"]["attempted_samples"], 0)

    def test_wrapper_collects_entire_cache_after_a_counterexample(self):
        with mock.patch.object(self.evaluator, "eval_numeric_expr", side_effect=[0, 1, 2, 2, 3, 3]) as evaluate:
            result = self.evaluator.numerical_comparison("0", "1", [(None, None)] * 3)
        self.assertEqual(result["status"], "numerical_mismatch")
        self.assertEqual(result["counts"]["attempted_samples"], 3)
        self.assertEqual(result["counts"]["equivalent_samples"], 2)
        self.assertEqual(evaluate.call_count, 6)

    def test_gravity_report_names_the_actual_oracle_domain(self):
        from data_gen.data_gen_gravity.v2_validation import NumericalContext
        points = NumericalContext("3s2h", seed=42171).points[:2]
        with mock.patch.object(self.evaluator, "NUMERIC_BACKEND", "gravity"):
            result = self.evaluator.numerical_comparison("p_1·p_2", "p_2·p_1", {"3s2h": points}, gravity_process="3s2h")
        self.assertTrue(result["equivalent"], result)
        self.assertEqual(result["counts"]["requested_valid_samples"], 2)
        self.assertEqual(result["helicity_domain"], "same-positive-helicity")
        self.assertEqual(result["process"], "3s2h")
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
