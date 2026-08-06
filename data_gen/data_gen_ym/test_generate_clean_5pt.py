"""Focused tests for the corrected five-point Yang--Mills generator."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ..Tokenizer import ScatteringAmplitudeTokenizer
from . import generate_clean_5pt as clean5


TRACE5 = "Tr(F_1 · F_2 · F_3 · F_4 · F_5)"
REVERSED_TRACE5 = "Tr(F_1 · F_5 · F_4 · F_3 · F_2)"
DENOMINATOR5 = "p_1 · p_2*p_2 · p_3*p_3 · p_4"
NONZERO_SIMPLE5 = f"({TRACE5})/({DENOMINATOR5})"


class FivePointCanonicalizationTests(unittest.TestCase):
    def test_scalar_products_are_sorted_without_four_point_complements(self) -> None:
        self.assertEqual(
            clean5.canonicalize_simple_expression("p_5 · p_4").expression,
            "p_4 · p_5",
        )
        self.assertEqual(
            clean5.canonicalize_simple_expression("p_4 · p_5").expression,
            "p_4 · p_5",
        )
        difference = clean5.canonicalize_simple_expression(
            "p_1 · p_2 - p_4 · p_5"
        )
        self.assertEqual(difference.expression, "p_1 · p_2 - p_4 · p_5")

    def test_five_point_labels_are_accepted_and_six_is_rejected(self) -> None:
        self.assertIsNone(
            clean5.canonicalize_simple_expression("p_5 · p_5").expression
        )
        self.assertEqual(
            clean5.canonicalize_simple_expression(
                "p_1 · F_5 · p_2"
            ).expression,
            "p_1 · F_5 · p_2",
        )
        with self.assertRaises(clean5.ExpressionSyntaxError):
            clean5.canonicalize_simple_expression("p_6 · p_6")

    def test_odd_five_trace_reversal_has_a_minus_sign(self) -> None:
        self.assertEqual(
            clean5.canonicalize_simple_expression(REVERSED_TRACE5).expression,
            f"-{TRACE5}",
        )
        self.assertIsNone(
            clean5.canonicalize_simple_expression(
                f"{TRACE5} + {REVERSED_TRACE5}"
            ).expression
        )

    def test_boundary_and_open_chain_reversal_rules_include_leg_five(self) -> None:
        self.assertIsNone(
            clean5.canonicalize_simple_expression(
                "p_5 · F_5 · F_2 · p_1"
            ).expression
        )
        reversed_pair = (
            "p_1 · F_2 · F_3 · F_5 · p_4 "
            "+ p_4 · F_5 · F_3 · F_2 · p_1"
        )
        self.assertIsNone(
            clean5.canonicalize_simple_expression(reversed_pair).expression
        )
        self.assertIsNone(
            clean5.canonicalize_simple_expression(
                "p_5 · F_1 · F_5 · F_3 · p_5"
            ).expression
        )

    def test_rational_factors_cancel_at_five_points(self) -> None:
        expression = (
            f"({TRACE5}*p_4 · p_5)/"
            f"({DENOMINATOR5}*p_4 · p_5)"
        )
        self.assertEqual(
            clean5.canonicalize_simple_expression(expression).expression,
            NONZERO_SIMPLE5,
        )


class FivePointNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.zero_points = clean5.build_kinematic_points(
            base_seed=27101,
            checks_per_mode=2,
            energy_scale=2.0,
        )
        cls.validation_points = clean5.build_kinematic_points(
            base_seed=37109,
            checks_per_mode=2,
            energy_scale=2.0,
        )
        cls.tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8,
            max_sequence_length=None,
        )

    def test_five_point_momentum_conservation_zero_is_rejected(self) -> None:
        conservation_zero = (
            "p_1 · p_2 + p_1 · p_3 + p_1 · p_4 + p_1 · p_5"
        )
        self.assertTrue(
            clean5.numerically_zero(
                conservation_zero,
                self.zero_points,
                tolerance=1e-12,
            )
        )
        self.assertTrue(
            clean5.numerically_zero(
                "p_1 · p_2 - p_3 · p_4 - p_3 · p_5 - p_4 · p_5",
                self.zero_points,
                tolerance=1e-12,
            )
        )
        self.assertFalse(
            clean5.numerically_zero(
                "p_1 · p_2 - p_4 · p_5",
                self.zero_points,
                tolerance=1e-12,
            )
        )

    def test_zero_subset_is_removed_from_a_nonzero_target(self) -> None:
        expression = (
            "p_1 · p_2 + p_1 · p_3 + p_1 · p_4 + p_1 · p_5 "
            "+ 2*p_2 · p_3"
        )
        cleaned, removed = clean5.remove_numerically_zero_subsets(
            expression,
            self.zero_points,
            tolerance=1e-12,
            max_subset_terms=6,
        )
        self.assertEqual(cleaned, "2*p_2 · p_3")
        self.assertEqual(removed, 4)

    def test_valid_pair_with_leg_five_is_tokenized_without_unknowns(self) -> None:
        prepared = clean5.prepare_pair(
            NONZERO_SIMPLE5,
            NONZERO_SIMPLE5,
            tokenizer=self.tokenizer,
            zero_points=self.zero_points,
            validation_points=self.validation_points,
            zero_tolerance=1e-12,
            tol_abs=1e-10,
            tol_rel=1e-8,
            max_subset_terms=6,
            max_tokens=4096,
        )
        self.assertIsNotNone(prepared)
        self.assertNotIn(self.tokenizer.vocab["<UNK>"], prepared.simple_tokens)
        self.assertNotIn(self.tokenizer.vocab["<UNK>"], prepared.scrambled_tokens)


class FivePointOutputTests(unittest.TestCase):
    def _args(self, directory: Path):
        parser = clean5.build_parser()
        return parser.parse_args(
            [
                "--samples",
                "2",
                "--jobs",
                "1",
                "--candidate-batch-size",
                "3",
                "--generator-batch-size",
                "3",
                "--max-candidates-factor",
                "3",
                "--zero-checks",
                "1",
                "--validation-checks",
                "1",
                "--progress-every",
                "0",
                "--raw-out",
                str(directory / "raw.csv.gz"),
                "--tok-out",
                str(directory / "tok.csv.gz"),
                "--report-out",
                str(directory / "report.json"),
            ]
        )

    def test_parser_and_default_paths_are_five_point_specific(self) -> None:
        args = clean5.build_parser().parse_args([])
        self.assertEqual(args.max_tokens, 4096)
        raw, tokenized, report = clean5.default_paths(500000)
        self.assertEqual(raw.name, "ym_5pt_500000_canonical_nonzero.csv.gz")
        self.assertEqual(
            tokenized.name,
            "ym_5pt_500000_canonical_nonzero_tok.csv.gz",
        )
        self.assertEqual(report.name, "ym_5pt_500000_canonical_nonzero.report.json")

    def test_streaming_refill_and_report_use_five_point_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory)
            zero = ("-p_5 · p_5", "-p_5 · p_5")
            valid_1 = (f"2*({TRACE5})/({DENOMINATOR5})",) * 2
            valid_2 = (f"3*({TRACE5})/({DENOMINATOR5})",) * 2

            with mock.patch(
                "data_gen.data_gen_ym.generate_clean_4pt.build_dataset_batched",
                return_value=[zero, valid_1, valid_2],
            ) as build_mock:
                stats, report = clean5.generate_to_files(args)

            self.assertEqual(build_mock.call_args.args[0], 5)
            self.assertEqual(stats.accepted, 2)
            self.assertEqual(stats.exact_zero_targets_rejected, 1)
            self.assertEqual(report["generator"], "clean_5pt_yang_mills")
            self.assertEqual(report["settings"]["particles"], 5)
            self.assertEqual(report["settings"]["max_tokens"], 4096)

            with gzip.open(args.raw_out, "rt", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with gzip.open(args.tok_out, "rt", newline="") as handle:
                token_rows = list(csv.DictReader(handle))
            self.assertEqual(len(raw_rows), 2)
            self.assertEqual(len(token_rows), 2)
            tokenizer = ScatteringAmplitudeTokenizer(max_particles=8)
            for raw_row, token_row in zip(raw_rows, token_rows):
                self.assertEqual(
                    json.loads(token_row["simple"]),
                    tokenizer.encode_infix(raw_row["simple"]),
                )
                self.assertEqual(
                    json.loads(token_row["scrambled"]),
                    tokenizer.encode_infix(raw_row["scrambled"]),
                )


if __name__ == "__main__":
    unittest.main()
