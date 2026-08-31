"""Focused tests for the zero-free four-point Yang--Mills generator."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .generate_clean_4pt import (
    ExpressionSyntaxError,
    TokenizationError,
    _publish,
    _publish_all,
    _validate_args,
    build_kinematic_points,
    build_parser,
    canonicalize_simple_expression,
    generate_to_files,
    numerically_zero,
    parenthesize_for_semantic_tokenization,
    prepare_pair,
    remove_numerically_zero_subsets,
)

try:
    from ..Tokenizer import ScatteringAmplitudeTokenizer
except ImportError:  # pragma: no cover - historical top-level package layout
    from Tokenizer import ScatteringAmplitudeTokenizer


TRACE = "Tr(F_1 · F_2 · F_3 · F_4)"
DENOMINATOR = "p_1 · p_2*p_1 · p_4"
NONZERO_SIMPLE = f"({TRACE})/({DENOMINATOR})"
OBSERVED_ZERO = (
    "-(Tr(F_1 · F_4 · F_3 · F_2))/(p_1 · p_2*p_1 · p_4) "
    "+ (Tr(F_1 · F_2 · F_3 · F_4)*p_3 · p_4)/"
    "(p_1 · p_2*p_1 · p_4*p_3 · p_4)"
)
NUMERICAL_ZERO_CHAIN = (
    "-(p_4 · F_1 · p_3*p_4 · F_2 · F_4 · F_3 · p_4)/"
    "(p_1 · p_4*p_2 · p_4*(p_3 · p_4)^2)"
)


class ExactCanonicalizationTests(unittest.TestCase):
    def assert_canonical(self, expression: str, expected: str | None) -> None:
        result = canonicalize_simple_expression(expression)
        self.assertEqual(result.expression, expected)
        if expected is not None:
            second = canonicalize_simple_expression(expected)
            self.assertEqual(second.expression, expected)

    def test_four_point_complementary_scalar_products_match(self) -> None:
        self.assert_canonical("p_3 · p_4", "p_1 · p_2")
        self.assert_canonical("p_2 · p_4", "p_1 · p_3")
        self.assert_canonical("p_2 · p_3", "p_1 · p_4")

    def test_trace_cyclicity_and_even_reversal(self) -> None:
        self.assert_canonical(
            "Tr(F_2 · F_3 · F_4 · F_1)",
            "Tr(F_1 · F_2 · F_3 · F_4)",
        )
        self.assert_canonical(
            "Tr(F_1 · F_4 · F_3 · F_2)",
            "Tr(F_1 · F_2 · F_3 · F_4)",
        )

    def test_odd_trace_reversal_has_a_minus_sign(self) -> None:
        self.assert_canonical(
            "Tr(F_1 · F_3 · F_2)",
            "-Tr(F_1 · F_2 · F_3)",
        )
        self.assert_canonical(
            "Tr(F_1 · F_2 · F_3) + Tr(F_1 · F_3 · F_2)",
            None,
        )

    def test_open_chain_reversal_signs(self) -> None:
        even = "p_4 · F_3 · F_2 · p_1"
        self.assert_canonical(even, "p_1 · F_2 · F_3 · p_4")
        odd_sum = (
            "p_1 · F_2 · F_3 · F_4 · p_2 "
            "+ p_2 · F_4 · F_3 · F_2 · p_1"
        )
        self.assert_canonical(odd_sum, None)

    def test_rational_factor_cancellation_and_combination(self) -> None:
        expression = (
            f"({TRACE}*p_3 · p_4)/(p_1 · p_2*p_1 · p_4*p_3 · p_4)"
        )
        self.assert_canonical(expression, f"({TRACE})/(p_1 · p_2*p_1 · p_4)")
        self.assert_canonical(
            f"2*({TRACE})/({DENOMINATOR}) "
            f"- 3*({TRACE})/({DENOMINATOR}) "
            f"+ ({TRACE})/({DENOMINATOR})",
            None,
        )

    def test_zero_power_factors_are_removed_from_both_rational_sides(self) -> None:
        self.assert_canonical("(p_1 · p_2)^0", "1")
        self.assert_canonical("1/((p_1 · p_2)^0)", "1")
        with self.assertRaises(ExpressionSyntaxError):
            canonicalize_simple_expression("(p_9 · p_9)^0")

    def test_observed_model_zero_is_an_exact_regression(self) -> None:
        result = canonicalize_simple_expression(OBSERVED_ZERO)
        self.assertIsNone(result.expression)
        self.assertEqual(result.combined_terms_removed, 2)

    def test_manifest_zero_factor_is_removed(self) -> None:
        result = canonicalize_simple_expression(
            f"p_1 · F_1 · p_2 + ({TRACE})/({DENOMINATOR})"
        )
        self.assertEqual(result.expression, NONZERO_SIMPLE)
        self.assertEqual(result.exact_zero_terms_removed, 1)

    def test_unsupported_factor_fails_closed(self) -> None:
        malformed_zeros = (
            "e_1 · e_2",
            "p_9 · p_9",
            "p_9 · F_1 · p_1",
            "Tr(F_5 · F_1 · F_1 · F_1)",
        )
        for expression in malformed_zeros:
            with self.subTest(expression=expression):
                with self.assertRaises(ExpressionSyntaxError):
                    canonicalize_simple_expression(expression)


class NumericalCleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.zero_points = build_kinematic_points(
            base_seed=17011,
            checks_per_mode=2,
            energy_scale=2.0,
        )
        cls.validation_points = build_kinematic_points(
            base_seed=91009,
            checks_per_mode=2,
            energy_scale=2.0,
        )
        cls.tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8,
            max_sequence_length=None,
        )

    def test_residual_chain_zero_is_numerically_rejected(self) -> None:
        self.assertTrue(
            numerically_zero(NUMERICAL_ZERO_CHAIN, self.zero_points, tolerance=1e-12)
        )
        cleaned, removed = remove_numerically_zero_subsets(
            NUMERICAL_ZERO_CHAIN,
            self.zero_points,
            tolerance=1e-12,
            max_subset_terms=6,
        )
        self.assertIsNone(cleaned)
        self.assertEqual(removed, 1)

    def test_scaled_momentum_conservation_zero_uses_relative_residual(self) -> None:
        scaled_zero = (
            "1000000000000*p_1 · p_2 + "
            "1000000000000*p_1 · p_3 + "
            "1000000000000*p_1 · p_4"
        )
        self.assertTrue(
            numerically_zero(
                scaled_zero,
                self.zero_points,
                tolerance=1e-12,
                relative_tolerance=1e-10,
            )
        )

    def test_valid_pair_is_accepted_and_zero_pair_is_rejected(self) -> None:
        prepared = prepare_pair(
            NONZERO_SIMPLE,
            NONZERO_SIMPLE,
            tokenizer=self.tokenizer,
            zero_points=self.zero_points,
            validation_points=self.validation_points,
            zero_tolerance=1e-12,
            tol_abs=1e-10,
            tol_rel=1e-8,
            max_subset_terms=6,
            max_tokens=2048,
        )
        self.assertIsNotNone(prepared)
        self.assertTrue(prepared.simple_tokens)

        rejected = prepare_pair(
            OBSERVED_ZERO,
            OBSERVED_ZERO,
            tokenizer=self.tokenizer,
            zero_points=self.zero_points,
            validation_points=self.validation_points,
            zero_tolerance=1e-12,
            tol_abs=1e-10,
            tol_rel=1e-8,
            max_subset_terms=6,
            max_tokens=2048,
        )
        self.assertIsNone(rejected)


class SemanticTokenizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8,
            max_sequence_length=None,
        )

    def assert_prefix(self, expression: str, expected: str) -> None:
        normalized = parenthesize_for_semantic_tokenization(expression)
        self.assertEqual(
            self.tokenizer.decode_prefix(self.tokenizer.encode_infix(normalized)),
            expected,
        )
        self.assertEqual(
            parenthesize_for_semantic_tokenization(normalized),
            normalized,
        )

    def test_products_of_dot_chains_are_distinct_scalar_factors(self) -> None:
        self.assert_prefix(
            "e_1 · p_2*e_2 · p_1",
            "* · e_1 p_2 · e_2 p_1",
        )

    def test_products_of_open_chains_are_distinct_factors(self) -> None:
        self.assert_prefix(
            "p_1 · F_2 · p_3*p_2 · F_4 · p_1",
            "* · · p_1 F_2 p_3 · · p_2 F_4 p_1",
        )

    def test_grouped_compact_output_remains_canonicalizable(self) -> None:
        compact = (
            f"({TRACE}*(p_1 · p_4)^2)/"
            "(p_1 · p_2*p_1 · p_3*(p_1 · p_4)^2)"
        )
        canonical = canonicalize_simple_expression(compact).expression
        self.assertIsNotNone(canonical)
        normalized = parenthesize_for_semantic_tokenization(canonical)
        self.assertEqual(
            canonicalize_simple_expression(normalized).expression,
            canonical,
        )

    def test_power_applies_to_the_complete_dot_chain(self) -> None:
        self.assert_prefix(
            "p_3 · p_4^2",
            "^ · p_3 p_4 2:",
        )

    def test_unary_minus_wraps_a_power_instead_of_its_base(self) -> None:
        self.assert_prefix(
            "-p_3 · p_4^2",
            "u- ^ · p_3 p_4 2:",
        )


class StreamingGenerationTests(unittest.TestCase):
    def _args(self, directory: Path) -> argparse.Namespace:
        parser = build_parser()
        return parser.parse_args(
            [
                "--samples",
                "3",
                "--jobs",
                "1",
                "--candidate-batch-size",
                "4",
                "--generator-batch-size",
                "4",
                "--max-candidates-factor",
                "4",
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

    def test_filter_refills_to_exact_count_and_keeps_outputs_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory)
            valid_1 = (NONZERO_SIMPLE, NONZERO_SIMPLE)
            valid_2 = (f"2*({TRACE})/({DENOMINATOR})",) * 2
            valid_3 = (f"3*({TRACE})/({DENOMINATOR})",) * 2
            zero = (OBSERVED_ZERO, OBSERVED_ZERO)
            batches = [
                [zero, valid_1, valid_1, valid_2],
                [valid_3, valid_3, valid_3, valid_3],
            ]

            with mock.patch(
                "data_gen.data_gen_ym.generate_clean_4pt.build_dataset_batched",
                side_effect=batches,
            ):
                stats, report = generate_to_files(args)

            self.assertEqual(stats.accepted, 3)
            self.assertEqual(stats.exact_zero_targets_rejected, 1)
            self.assertGreaterEqual(stats.duplicate_rejections, 1)
            self.assertEqual(report["stats"]["accepted"], 3)
            self.assertEqual(report["report_schema_version"], 2)
            self.assertEqual(
                report["settings"]["tokenization_normalization"],
                "fully_parenthesized_numeric_ast_v1",
            )

            with gzip.open(args.raw_out, "rt", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with gzip.open(args.tok_out, "rt", newline="") as handle:
                token_rows = list(csv.DictReader(handle))
            with gzip.open(args.raw_out, "rb") as handle:
                raw_hash = hashlib.sha256(handle.read()).hexdigest()
            with gzip.open(args.tok_out, "rb") as handle:
                token_hash = hashlib.sha256(handle.read()).hexdigest()
            self.assertEqual(
                report["outputs"]["raw"]["sha256_uncompressed"],
                raw_hash,
            )
            self.assertEqual(
                report["outputs"]["tokenized"]["sha256_uncompressed"],
                token_hash,
            )
            self.assertEqual(len(raw_rows), 3)
            self.assertEqual(len(token_rows), 3)
            for raw, tokenized in zip(raw_rows, token_rows):
                simple_tokens = json.loads(tokenized["simple"])
                scrambled_tokens = json.loads(tokenized["scrambled"])
                self.assertEqual(
                    simple_tokens,
                    ScatteringAmplitudeTokenizer(max_particles=8).encode_infix(
                        raw["simple"]
                    ),
                )
                self.assertEqual(
                    scrambled_tokens,
                    ScatteringAmplitudeTokenizer(max_particles=8).encode_infix(
                        raw["scrambled"]
                    ),
                )

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory)
            raw_path = Path(args.raw_out)
            raw_path.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_to_files(args)
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "sentinel")
            self.assertFalse(Path(args.tok_out).exists())
            self.assertFalse(Path(args.report_out).exists())

    def test_overlapping_seed_grids_are_rejected_by_cli_and_direct_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            args.zero_seed = 1
            args.validation_seed = 2
            args.zero_checks = 3
            args.validation_checks = 3

            parser = build_parser()
            with mock.patch.object(parser, "error", side_effect=ValueError) as error:
                with self.assertRaises(ValueError):
                    _validate_args(args, parser)
            self.assertIn("must not overlap", error.call_args.args[0])

            with self.assertRaisesRegex(ValueError, "must not overlap"):
                generate_to_files(args)

    def test_tokenizer_value_error_is_counted_as_token_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory)
            args.samples = 1
            args.candidate_batch_size = 2
            args.max_candidates_factor = 2
            bad = (NONZERO_SIMPLE, NONZERO_SIMPLE)
            good = (f"2*({TRACE})/({DENOMINATOR})",) * 2

            prepare_calls = 0

            def prepare_side_effect(*call_args, **call_kwargs):
                nonlocal prepare_calls
                prepare_calls += 1
                if prepare_calls == 1:
                    raise TokenizationError("unknown token")
                return prepare_pair(*call_args, **call_kwargs)

            with mock.patch(
                "data_gen.data_gen_ym.generate_clean_4pt.build_dataset_batched",
                return_value=[bad, good],
            ), mock.patch(
                "data_gen.data_gen_ym.generate_clean_4pt.prepare_pair",
                side_effect=prepare_side_effect,
            ):
                stats, _report = generate_to_files(args)

            self.assertEqual(stats.accepted, 1)
            self.assertEqual(stats.token_rejections, 1)
            self.assertEqual(stats.numerical_rejections, 0)

    def test_successful_symlink_overwrite_reports_the_published_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory)
            args.samples = 1
            args.overwrite = True
            raw_path = Path(args.raw_out)
            old_target = directory / "old-raw-target"
            old_target.write_text("old raw data\n", encoding="utf-8")
            raw_path.symlink_to(old_target.name)

            with mock.patch(
                "data_gen.data_gen_ym.generate_clean_4pt.build_dataset_batched",
                return_value=[(NONZERO_SIMPLE, NONZERO_SIMPLE)],
            ):
                _stats, report = generate_to_files(args)

            self.assertFalse(raw_path.is_symlink())
            self.assertEqual(old_target.read_text(encoding="utf-8"), "old raw data\n")
            self.assertEqual(
                report["outputs"]["raw"]["path"],
                str(raw_path.absolute()),
            )
            persisted = json.loads(Path(args.report_out).read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["outputs"]["raw"]["path"],
                str(raw_path.absolute()),
            )


class TransactionalPublicationTests(unittest.TestCase):
    def test_overwrite_failure_restores_every_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            temps = [directory / f"temp-{index}" for index in range(3)]
            destinations = [directory / f"out-{index}" for index in range(3)]
            for index, path in enumerate(temps):
                path.write_text(f"new-{index}", encoding="utf-8")
            for index, path in enumerate(destinations):
                path.write_text(f"old-{index}", encoding="utf-8")

            calls = 0

            def fail_second(temp_path, destination, *, overwrite):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publication failure")
                _publish(temp_path, destination, overwrite=overwrite)

            with mock.patch(
                "data_gen.data_gen_ym.generate_clean_4pt._publish",
                side_effect=fail_second,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    _publish_all(temps, destinations, overwrite=True)

            self.assertEqual(
                [path.read_text(encoding="utf-8") for path in destinations],
                ["old-0", "old-1", "old-2"],
            )

    def test_dangling_symlink_counts_as_an_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            temp_path = directory / "temp"
            destination = directory / "dangling"
            temp_path.write_text("new", encoding="utf-8")
            destination.symlink_to(directory / "missing-target")

            with self.assertRaises(FileExistsError):
                _publish_all([temp_path], [destination], overwrite=False)
            self.assertTrue(destination.is_symlink())

    def test_overwrite_failure_restores_a_dangling_symlink_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            temps = [directory / "temp-0", directory / "temp-1"]
            destinations = [directory / "dangling", directory / "out-1"]
            for index, path in enumerate(temps):
                path.write_text(f"new-{index}", encoding="utf-8")
            missing_target = directory / "still-missing"
            destinations[0].symlink_to(missing_target)
            destinations[1].write_text("old-1", encoding="utf-8")

            calls = 0

            def fail_second(temp_path, destination, *, overwrite):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publication failure")
                _publish(temp_path, destination, overwrite=overwrite)

            with mock.patch(
                "data_gen.data_gen_ym.generate_clean_4pt._publish",
                side_effect=fail_second,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    _publish_all(temps, destinations, overwrite=True)

            self.assertTrue(destinations[0].is_symlink())
            self.assertEqual(destinations[0].readlink(), missing_target)
            self.assertEqual(destinations[1].read_text(encoding="utf-8"), "old-1")


if __name__ == "__main__":
    unittest.main()
