"""Regression checks for strict SQED evaluation and numerical tolerances."""

from __future__ import annotations

import cmath
import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_gen import gen_data as gd
from data_gen.data_gen_gravity import core as gravity_core
from data_gen.data_gen_ym import numerics as ym_numerics
from data_gen.numeric_utils import numeric_values_close
from data_testing import evaluate_model as evaluator


INVALID_SQED_EXPRESSIONS = (
    "p_1 · p_2 + p_5 · p_1",
    "p_1 · p_2 + e_5 · p_1",
    "p_1 · p_2 + p_1",
    "p_1 · p_2 + e_1 · p_2",
    "p_1 · p_2 + F_4 · p_2",
)


def independent_minkowski_dot(vector_a, vector_b) -> float:
    """Compute (+,-,-,-) contraction without production evaluator helpers."""
    if len(vector_a) != 4 or len(vector_b) != 4:
        raise ValueError("Minkowski oracle requires two four-vectors")
    return float(
        vector_a[0] * vector_b[0]
        - sum(
            component_a * component_b
            for component_a, component_b in zip(
                vector_a[1:],
                vector_b[1:],
            )
        )
    )


class NumericValuesCloseTests(unittest.TestCase):
    """Cross-check the shared comparator against Python's independent oracle."""

    ABS_TOL = 1e-10
    REL_TOL = 1e-8

    def test_matches_math_isclose_without_a_unit_floor(self) -> None:
        cases = (
            (0.0, 0.5e-10),
            (1.0, 1.0 + 0.5e-8),
            (0.0, 1e-9),
            (0.0, 1e-8),
            (-3.0, -3.0 - 2e-8),
        )
        for value_a, value_b in cases:
            with self.subTest(value_a=value_a, value_b=value_b):
                expected = math.isclose(
                    value_a,
                    value_b,
                    abs_tol=self.ABS_TOL,
                    rel_tol=self.REL_TOL,
                )
                self.assertEqual(
                    numeric_values_close(
                        value_a,
                        value_b,
                        tol_abs=self.ABS_TOL,
                        tol_rel=self.REL_TOL,
                    ),
                    expected,
                )

        self.assertFalse(
            math.isclose(
                0.0,
                1e-9,
                abs_tol=self.ABS_TOL,
                rel_tol=self.REL_TOL,
            )
        )
        self.assertFalse(
            numeric_values_close(
                0.0,
                1e-9,
                tol_abs=self.ABS_TOL,
                tol_rel=self.REL_TOL,
            )
        )
        self.assertEqual(
            numeric_values_close(
                0.0j,
                complex(1e-9, -1e-9),
                tol_abs=self.ABS_TOL,
                tol_rel=self.REL_TOL,
            ),
            cmath.isclose(
                0.0j,
                complex(1e-9, -1e-9),
                abs_tol=self.ABS_TOL,
                rel_tol=self.REL_TOL,
            ),
        )

    def test_non_finite_values_never_compare_equal(self) -> None:
        for value_a, value_b in (
            (math.inf, math.inf),
            (-math.inf, -math.inf),
            (math.nan, math.nan),
            (0.0, math.inf),
            (complex(math.inf, 0.0), complex(math.inf, 0.0)),
            (complex(math.nan, 1.0), complex(math.nan, 1.0)),
        ):
            with self.subTest(value_a=value_a, value_b=value_b):
                self.assertFalse(
                    numeric_values_close(
                        value_a,
                        value_b,
                        tol_abs=self.ABS_TOL,
                        tol_rel=self.REL_TOL,
                    )
                )

    def test_sqed_and_ym_pair_validators_reject_near_zero_mismatch(
        self,
    ) -> None:
        validators = (gd._validate_pair, ym_numerics._validate_pair)
        for validate_pair in validators:
            with self.subTest(validator=validate_pair.__module__):
                close_result, close_reason = validate_pair(
                    "0",
                    "1/20000000000",
                    4,
                    2.0,
                    n_checks=1,
                    tol_abs=self.ABS_TOL,
                    tol_rel=self.REL_TOL,
                    pol_modes=("coulomb",),
                )
                self.assertTrue(close_result, close_reason)

                mismatch_result, _ = validate_pair(
                    "0",
                    "1/1000000000",
                    4,
                    2.0,
                    n_checks=1,
                    tol_abs=self.ABS_TOL,
                    tol_rel=self.REL_TOL,
                    pol_modes=("coulomb",),
                )
                self.assertFalse(mismatch_result)

    def test_complex_gravity_validator_uses_the_same_tolerance_rule(
        self,
    ) -> None:
        common = {
            "process": "4s1h",
            "seeds": (17,),
            "reference_modes": ("first",),
            "gauge_shift": False,
            "atol": self.ABS_TOL,
            "rtol": self.REL_TOL,
        }
        close_result, _ = gravity_core.numerically_equivalent(
            "0",
            "1/20000000000",
            **common,
        )
        mismatch_result, _ = gravity_core.numerically_equivalent(
            "0",
            "1/1000000000",
            **common,
        )
        self.assertTrue(close_result)
        self.assertFalse(mismatch_result)


class StrictSqedEvaluationTests(unittest.TestCase):
    """Exercise strict parsing and independently check valid SQED algebra."""

    N_PARTICLES = 4

    def setUp(self) -> None:
        self.momenta, self.polarisations = gd.generate_kinematics(
            self.N_PARTICLES,
            M=2.0,
            pol_mode="coulomb",
            seed=123,
        )

    def test_valid_field_strength_chain_matches_direct_minkowski_formula(
        self,
    ) -> None:
        expression = "p_1 · F_2 · p_4"
        actual = gd.eval_infix_numeric(
            expression,
            self.momenta,
            self.polarisations,
            strict=True,
        )

        p_1 = self.momenta[0]
        p_2 = self.momenta[1]
        p_4 = self.momenta[3]
        e_2 = self.polarisations[0]
        expected = (
            independent_minkowski_dot(p_1, p_2)
            * independent_minkowski_dot(e_2, p_4)
            - independent_minkowski_dot(p_1, e_2)
            * independent_minkowski_dot(p_2, p_4)
        )
        self.assertTrue(
            math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12),
            f"strict evaluator returned {actual}, direct contraction returned {expected}",
        )

        trace_actual = gd.eval_infix_numeric(
            "Tr(F_2 · F_3)",
            self.momenta,
            self.polarisations,
            strict=True,
        )
        p_3 = self.momenta[2]
        e_3 = self.polarisations[1]
        trace_expected = 2.0 * (
            independent_minkowski_dot(e_2, p_3)
            * independent_minkowski_dot(e_3, p_2)
            - independent_minkowski_dot(p_2, p_3)
            * independent_minkowski_dot(e_2, e_3)
        )
        self.assertTrue(
            math.isclose(
                trace_actual,
                trace_expected,
                abs_tol=1e-12,
                rel_tol=1e-12,
            ),
            "strict trace expansion disagrees with the direct contraction",
        )

        for valid in (
            "p_1 · p_2",
            "e_2 · p_1",
            "(p_1 · F_2 · p_4)/(p_1 · p_2)",
        ):
            with self.subTest(valid=valid):
                self.assertTrue(
                    math.isfinite(
                        gd.eval_infix_numeric(
                            valid,
                            self.momenta,
                            self.polarisations,
                            strict=True,
                        )
                    )
                )

    def test_invalid_vectors_and_free_vectors_are_rejected(self) -> None:
        invalid = INVALID_SQED_EXPRESSIONS + (
            "p_1 · F_4 · p_2",
            "p_1 · F_1 · p_4",
            "e_2",
            "F_2",
        )
        for expression in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises((KeyError, ValueError)):
                    gd.eval_infix_numeric(
                        expression,
                        self.momenta,
                        self.polarisations,
                        strict=True,
                    )

    def test_strict_parser_rejects_unconsumed_or_malformed_syntax(self) -> None:
        malformed = (
            "",
            "p_1 · p_2 @",
            "p_1 · p_2 p_3 · p_4",
            "(p_1 · p_2",
            "p_1 · p_2)",
            "p_1 · p_2 +",
            "p_1 ·",
            "p_1 · · p_2",
            "unknown_symbol",
            "p_0 · p_1",
            "e_0 · p_1",
            "p_1 · F_0 · p_4",
            "p_1 · F_2",
            "F_2 · p_1",
            "F_2 · F_3",
            "Tr()",
            "Tr(F_2)",
        )
        for expression in malformed:
            with self.subTest(expression=expression):
                with self.assertRaises((KeyError, ValueError)):
                    gd.eval_infix_numeric(
                        expression,
                        self.momenta,
                        self.polarisations,
                        strict=True,
                    )

    def test_legacy_permissive_mode_remains_opt_in_default(self) -> None:
        target = "p_1 · p_2"
        malformed = "p_1 · p_2 + p_5 · p_1"
        self.assertEqual(
            gd.eval_infix_numeric(
                target,
                self.momenta,
                self.polarisations,
            ),
            gd.eval_infix_numeric(
                malformed,
                self.momenta,
                self.polarisations,
            ),
        )


class EvaluatorIntegrationTests(unittest.TestCase):
    """Verify strict SQED and tolerance behavior at the scoring API boundary."""

    SETTINGS = (
        "NUMERIC_BACKEND",
        "N_PARTICLES",
        "NUMERIC_EQUIV_SAMPLES",
        "NUMERIC_EQUIV_SEED",
        "NUMERIC_EQUIV_MASS",
        "NUMERIC_EQUIV_POL_MODES",
        "NUMERIC_TOL_ABS",
        "NUMERIC_TOL_REL",
        "RAW_CSV_PATH",
    )

    def setUp(self) -> None:
        self.original_settings = {
            name: getattr(evaluator, name) for name in self.SETTINGS
        }
        evaluator.NUMERIC_BACKEND = "sqed"
        evaluator.N_PARTICLES = 4
        evaluator.NUMERIC_EQUIV_SAMPLES = 1
        evaluator.NUMERIC_EQUIV_SEED = 123
        evaluator.NUMERIC_EQUIV_MASS = 2.0
        evaluator.NUMERIC_EQUIV_POL_MODES = ("coulomb",)
        evaluator.NUMERIC_TOL_ABS = 1e-10
        evaluator.NUMERIC_TOL_REL = 1e-8
        evaluator.RAW_CSV_PATH = Path("<in-memory-regression-test>")
        self.kinematics = evaluator.precompute_kinematics()

    def tearDown(self) -> None:
        for name, value in self.original_settings.items():
            setattr(evaluator, name, value)

    def test_invalid_predictions_score_false_without_raising(self) -> None:
        target = "p_1 · p_2"
        for prediction in INVALID_SQED_EXPRESSIONS:
            with self.subTest(prediction=prediction):
                self.assertFalse(
                    evaluator.numerically_equivalent_exprs(
                        target,
                        prediction,
                        self.kinematics,
                    )
                )

    def test_sqed_router_explicitly_requests_strict_mode(self) -> None:
        momenta, polarisations = self.kinematics[0]
        with mock.patch.object(
            evaluator.gd,
            "eval_infix_numeric",
            wraps=evaluator.gd.eval_infix_numeric,
        ) as eval_mock:
            evaluator.eval_numeric_expr(
                "p_1 · p_2",
                momenta,
                polarisations,
            )

        eval_mock.assert_called_once()
        self.assertIs(eval_mock.call_args.kwargs.get("strict"), True)

    def test_invalid_references_fail_fast(self) -> None:
        target = "p_1 · p_2"
        for reference in INVALID_SQED_EXPRESSIONS:
            with self.subTest(reference=reference):
                with self.assertRaises(ValueError):
                    evaluator.validate_numeric_reference_rows(
                        [{"simple": target, "scrambled": reference}],
                        self.kinematics,
                    )

    def test_evaluator_rejects_near_zero_relative_floor_false_positives(
        self,
    ) -> None:
        self.assertTrue(
            evaluator.numerically_equivalent_exprs(
                "0",
                "1/20000000000",
                self.kinematics,
            )
        )
        for expression in ("1/1000000000", "1/100000000"):
            with self.subTest(expression=expression):
                self.assertFalse(
                    evaluator.numerically_equivalent_exprs(
                        "0",
                        expression,
                        self.kinematics,
                    )
                )

                value = gd.eval_infix_numeric(
                    expression,
                    *self.kinematics[0],
                    strict=True,
                )
                self.assertGreater(abs(value), evaluator.NUMERIC_TOL_ABS)
                self.assertFalse(
                    math.isclose(
                        0.0,
                        value,
                        abs_tol=evaluator.NUMERIC_TOL_ABS,
                        rel_tol=evaluator.NUMERIC_TOL_REL,
                    )
                )

        self.assertTrue(
            evaluator.numerically_equivalent_exprs(
                "1",
                "1 + 1/200000000",
                self.kinematics,
            )
        )


class GravityEvaluatorIntegrationTests(unittest.TestCase):
    """Exercise the process-aware complex backend at the scoring boundary."""

    SETTINGS = (
        "NUMERIC_BACKEND",
        "N_PARTICLES",
        "DATA_SOURCE",
        "NUMERIC_EQUIV_SAMPLES",
        "NUMERIC_EQUIV_SEED",
        "NUMERIC_EQUIV_POL_MODES",
        "NUMERIC_TOL_ABS",
        "NUMERIC_TOL_REL",
        "GRAVITY_REFERENCE_MODES",
        "GRAVITY_GAUGE_SHIFT",
        "GRAVITY_METADATA_CSV_PATH",
        "GRAVITY_PROCESS",
        "EXISTING_RAW_CSV_PATH",
        "EXISTING_CSV_DEDUPE",
        "EXISTING_CSV_MAX_ROWS",
        "RAW_CSV_PATH",
    )

    def setUp(self) -> None:
        self.original_settings = {
            name: getattr(evaluator, name) for name in self.SETTINGS
        }
        evaluator.NUMERIC_BACKEND = "gravity"
        evaluator.N_PARTICLES = 5
        evaluator.DATA_SOURCE = "csv"
        evaluator.NUMERIC_EQUIV_SAMPLES = 1
        evaluator.NUMERIC_EQUIV_SEED = 17
        evaluator.NUMERIC_EQUIV_POL_MODES = None
        evaluator.NUMERIC_TOL_ABS = None
        evaluator.NUMERIC_TOL_REL = None
        evaluator.GRAVITY_REFERENCE_MODES = ("first",)
        evaluator.GRAVITY_GAUGE_SHIFT = False
        evaluator.GRAVITY_METADATA_CSV_PATH = None
        evaluator.GRAVITY_PROCESS = "3s2h"
        evaluator.RAW_CSV_PATH = Path("<in-memory-gravity-test>")
        evaluator.validate_runtime_config()
        self.kinematics = evaluator.precompute_kinematics()

    def tearDown(self) -> None:
        for name, value in self.original_settings.items():
            setattr(evaluator, name, value)

    def test_compact_and_expanded_benchmarks_match_for_both_processes(
        self,
    ) -> None:
        for process, compact in gravity_core.BENCHMARKS.items():
            with self.subTest(process=process):
                expanded = gravity_core.expand_expression(compact)
                self.assertTrue(
                    evaluator.numerically_equivalent_exprs(
                        compact,
                        expanded,
                        self.kinematics,
                        gravity_process=process,
                    )
                )
                self.assertFalse(
                    evaluator.numerically_equivalent_exprs(
                        compact,
                        "0",
                        self.kinematics,
                        gravity_process=process,
                    )
                )

    def test_strict_gravity_boundary_rejects_malformed_and_free_vectors(
        self,
    ) -> None:
        invalid = (
            "unknown_symbol",
            "M_1",
            "p_1",
            "e_4",
            "F_4",
            "p_6 · p_1",
            "p_1 · p_2 @",
            "(p_1 · p_2",
            "p_1 · p_2 +",
        )
        for expression in invalid:
            with self.subTest(expression=expression):
                self.assertFalse(
                    evaluator.numerically_equivalent_exprs(
                        "0",
                        expression,
                        self.kinematics,
                        gravity_process="3s2h",
                    )
                )

    def test_process_assignment_controls_allowed_graviton_legs(self) -> None:
        expression = "p_1 · F_4 · p_2"
        evaluator.validate_gravity_expression(expression, "3s2h")
        with self.assertRaises(KeyError):
            evaluator.validate_gravity_expression(expression, "4s1h")
        self.assertFalse(
            evaluator.numerically_equivalent_exprs(
                "0",
                expression,
                self.kinematics,
                gravity_process="4s1h",
            )
        )

    def test_reference_validation_uses_the_row_process(self) -> None:
        process = "4s1h"
        compact = gravity_core.BENCHMARKS[process]
        expanded = gravity_core.expand_expression(compact)
        evaluator.validate_numeric_reference_rows(
            [{"simple": compact, "scrambled": expanded}],
            self.kinematics,
            [process],
        )
        with self.assertRaises(ValueError):
            evaluator.validate_numeric_reference_rows(
                [{"simple": compact, "scrambled": "p_1 · F_4 · p_2"}],
                self.kinematics,
                [process],
            )

    def test_metadata_discovery_preserves_dedupe_and_row_cap_alignment(
        self,
    ) -> None:
        pairs = [
            ("p_1 · p_2", "p_1 · p_2"),
            ("p_1 · p_2", "p_1 · p_2"),
            ("p_3 · p_4", "p_3 · p_4"),
        ]
        processes = ["3s2h", "3s2h", "4s1h"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            raw_path = directory / "sample_raw.csv"
            metadata_path = directory / "sample_metadata.csv"
            with raw_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["simple", "scrambled"])
                writer.writerows(pairs)
            with metadata_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["simple", "scrambled", "process"],
                )
                writer.writeheader()
                for pair, process in zip(pairs, processes, strict=True):
                    writer.writerow(
                        {
                            "simple": pair[0],
                            "scrambled": pair[1],
                            "process": process,
                        }
                    )

            evaluator.GRAVITY_PROCESS = None
            evaluator.GRAVITY_METADATA_CSV_PATH = None
            evaluator.EXISTING_RAW_CSV_PATH = raw_path
            evaluator.EXISTING_CSV_DEDUPE = True
            evaluator.EXISTING_CSV_MAX_ROWS = 2
            prepared = [
                {"simple": pairs[0][0], "scrambled": pairs[0][1]},
                {"simple": pairs[2][0], "scrambled": pairs[2][1]},
            ]

            self.assertEqual(
                evaluator.infer_gravity_metadata_path(raw_path),
                metadata_path,
            )
            self.assertEqual(
                evaluator.resolve_gravity_processes(prepared),
                ["3s2h", "4s1h"],
            )

            with metadata_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[-1]["simple"] = "misaligned"
            with metadata_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                evaluator.resolve_gravity_processes(prepared)


if __name__ == "__main__":
    unittest.main()
