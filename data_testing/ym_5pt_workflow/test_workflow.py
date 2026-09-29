"""Physical identities and failure guards for the prepared five-point workflow.

These tests need no checkpoint. The README gives the separate real-model
reproduction command, which verifies each actual decoder prediction.
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path
import tempfile
import unittest

import numpy as np
import sympy as sp

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from . import algebra as au
from .inference import completed_tokens
from .verification import compare, evaluate, make_points
from .workflow import (
    DATA, PREPARED, load_cases, read_expression, reconstruct, require_zero,
    token_roundtrip, verify_sources, weighted,
)


def tokenizer():
    return ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)


class TensorAlgebraTests(unittest.TestCase):
    def test_field_strength_sign_and_antisymmetry(self):
        chain = au.parse("p_1·F_2·e_3")
        expanded = au.parse(
            "(p_1·p_2)*(e_2·e_3) - (p_1·e_2)*(p_2·e_3)"
        )
        self.assertEqual(sp.expand(chain - expanded), 0)
        self.assertEqual(sp.expand(chain + au.parse("e_3·F_2·p_1")), 0)

    def test_two_field_strength_trace(self):
        expected = au.parse(
            "2*((p_1·e_2)*(e_1·p_2) - (p_1·p_2)*(e_1·e_2))"
        )
        self.assertEqual(sp.expand(au.parse("Tr(F_1·F_2)") - expected), 0)

    def test_masslessness_momentum_conservation_and_transversality(self):
        for leg in range(1, 6):
            with self.subTest(leg=leg):
                self.assertEqual(au.on_shell(au.parse(f"p_{leg}·p_{leg}")), 0)
                self.assertEqual(au.on_shell(au.parse(f"e_{leg}·p_{leg}")), 0)
                for tag in ("p", "e"):
                    contraction = au.parse(" + ".join(f"{tag}_{leg}·p_{j}" for j in range(1, 6)))
                    self.assertEqual(au.on_shell(contraction), 0)

    def test_parser_rejects_non_scalar_or_executable_syntax(self):
        for text in (
            "p_1·F_2", "p_1·e_2·p_3", "F_1", "p_6·p_1",
            "1/0", "1.5", "True", "unknown", "[1,2]",
            "__import__('os').getcwd()",
        ):
            with self.subTest(expression=text), self.assertRaises((ValueError, SyntaxError)):
                au.parse(text)


class PreparedAmplitudeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(PREPARED)
        cls.seed_text = read_expression(DATA / "gluon5feyn12345.csv.gz")
        cls.full_text = read_expression(DATA / "gluon5feyn.csv.gz")
        cls.full, cls.sources, cls.completed = verify_sources(
            cls.cases, cls.seed_text, cls.full_text, tokenizer()
        )

    def test_prepared_sources_complete_to_the_full_amplitude(self):
        self.assertEqual(len(self.sources), 16)
        self.assertEqual(au.on_shell(self.completed - self.full), 0)

    def test_raw_seed_is_not_the_gauge_invariant_target(self):
        self.assertNotEqual(au.ward(au.parse(self.seed_text), 1), 0)
        self.assertEqual(au.ward(self.full, 1), 0)

    def test_averaging_the_cyclic_sum_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Exact verification failed"):
            require_zero(self.completed / 5 - self.full, "incorrect cyclic average")

    def test_wrong_component_multiplier_is_rejected(self):
        # Flipping one fixed representative's sign must invalidate the result.
        coefficient = sp.Rational(self.cases[0]["multiplier"])
        error = -2 * coefficient * sum(au.cyclic(self.sources[0], k) for k in range(5))
        with self.assertRaisesRegex(RuntimeError, "Exact verification failed"):
            require_zero(self.completed + error - self.full, "flipped component sign")

    def test_independent_numerics_include_covariant_polarizations(self):
        points, error = make_points(421737, 2)
        self.assertEqual(len(points), 4)
        self.assertLess(error, 1e-10)
        self.assertTrue(np.all(points[0][1][:, 0] == 0))
        self.assertTrue(np.any(np.abs(points[2][1][:, 0]) > 1e-12))
        self.assertTrue(compare(evaluate(self.completed, points), evaluate(self.full, points))["passes"])


class ExportAndInputTests(unittest.TestCase):
    def test_external_halves_survive_adjacent_digit_tokenization(self):
        # A prediction ending in a numeric denominator exposed the historical
        # tokenizer's ambiguity between separate integer leaves and one number.
        prediction = "(p_1·p_2)/(2)"
        for coefficient in ("1/2", "-1/2"):
            with self.subTest(coefficient=coefficient):
                text = weighted(prediction, coefficient)
                token_roundtrip(text, tokenizer(), "coefficient regression")
                self.assertEqual(
                    sp.cancel(au.parse(text) - sp.Rational(coefficient) * au.parse(prediction)), 0
                )

    def test_reconstruction_roundtrip_preserves_scalar_and_tensor_terms(self):
        rows = [
            {"prediction": "(p_1·F_2·e_3)/(2)", "multiplier": "1/2"},
            {"prediction": "Tr(F_3·F_4)", "multiplier": "-1/2"},
        ]
        core, full = reconstruct(rows)
        expected_core = sum(sp.Rational(row["multiplier"]) * au.parse(row["prediction"]) for row in rows)
        self.assertEqual(sp.expand(au.parse(core) - expected_core), 0)
        self.assertEqual(sp.expand(au.parse(full) - sum(au.cyclic(expected_core, k) for k in range(5))), 0)
        token_roundtrip(core, tokenizer(), "core export")
        token_roundtrip(full, tokenizer(), "completed export")

    def test_csv_reader_accepts_one_row_and_rejects_silent_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expression.csv.gz"
            with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([1, "p_1·p_2"])
            self.assertEqual(read_expression(path), "p_1·p_2")
            with gzip.open(path, "at", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([2, "p_2·p_3"])
            with self.assertRaises(ValueError):
                read_expression(path)


class DecoderAndNumericsTests(unittest.TestCase):
    def test_only_complete_prediction_content_is_accepted(self):
        self.assertEqual(completed_tokens([2, 21, 25, 26, 3, 0, 0]), [21, 25, 26])
        for raw in (
            [], [21, 25, 26, 3], [2, 21, 25, 26], [2, 3],
            [2, 1, 3], [2, 0, 3], [2, 2, 3], [2, 21, 25, 26, 3, 12],
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                completed_tokens(raw)

    def test_numerical_comparison_rejects_missing_or_nonfinite_evidence(self):
        for left, right in (([], []), ([1], [1, 2]), ([np.nan], [0]), ([np.inf], [np.inf])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                compare(left, right)
        self.assertFalse(compare([1], [2])["passes"])
        self.assertTrue(compare([1], [1 + 1e-10])["passes"])


if __name__ == "__main__":
    unittest.main()
