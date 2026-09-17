"""Cyclic ordering, numerical cleaning, and exact-count publication regressions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import generate_clean_4pt as core


TRACE = "Tr(F_1 · F_2 · F_3 · F_4)"
FORWARD_CHAIN = "p_4 · F_2 · F_3 · F_4 · p_1"
REVERSE_CHAIN = "p_1 · F_4 · F_3 · F_2 · p_4"


class CyclicCanonicalizationTests(unittest.TestCase):
    def test_open_chain_keeps_forward_order_over_lexicographic_endpoint_order(self) -> None:
        self.assertEqual(
            core.canonicalize_simple_expression(FORWARD_CHAIN).expression,
            "-" + REVERSE_CHAIN,
        )
        for expression in (FORWARD_CHAIN, "-" + REVERSE_CHAIN):
            result = core.canonicalize_simple_expression(expression, cyclic_order=True)
            self.assertEqual(result.expression, FORWARD_CHAIN)
            self.assertEqual(
                core.canonicalize_simple_expression(result.expression, cyclic_order=True),
                result,
            )
        combined = core.canonicalize_simple_expression(
            FORWARD_CHAIN + " + " + REVERSE_CHAIN, cyclic_order=True,
        )
        self.assertIsNone(combined.expression)

    def test_rotations_wraparound_and_odd_trace_reversal_sign(self) -> None:
        for expression in (
            TRACE,
            "Tr(F_3 · F_4 · F_1 · F_2)",
            "Tr(F_2 · F_1 · F_4 · F_3)",
        ):
            self.assertEqual(
                core.canonicalize_simple_expression(expression, cyclic_order=True).expression,
                TRACE,
            )
        self.assertEqual(
            core.canonicalize_simple_expression(
                "Tr(F_1 · F_3 · F_2)", cyclic_order=True,
            ).expression,
            "-Tr(F_1 · F_2 · F_3)",
        )
        self.assertIsNone(core.canonicalize_simple_expression(
            "Tr(F_3 · F_1 · F_2) + Tr(F_1 · F_3 · F_2)",
            cyclic_order=True,
        ).expression)
        wrapped = "p_4 · F_3 · F_4 · F_1 · p_2"
        self.assertEqual(
            core.canonicalize_simple_expression(wrapped, cyclic_order=True).expression,
            wrapped,
        )

    def test_five_point_cycle_is_supported(self) -> None:
        word = "p_5 · F_2 · F_3 · F_4 · p_1"
        self.assertEqual(core.canonicalize_simple_expression(
            word, n_particles=5, cyclic_order=True,
        ).expression, word)
        self.assertEqual(core.canonicalize_simple_expression(
            "Tr(F_1 · F_5 · F_4 · F_3 · F_2)",
            n_particles=5, cyclic_order=True,
        ).expression, "-Tr(F_1 · F_2 · F_3 · F_4 · F_5)")

    def test_noncyclic_words_rejected_and_exact_antisymmetry_zeros_removed(self) -> None:
        with self.assertRaisesRegex(core.ExpressionSyntaxError, "forward cyclic"):
            core.canonicalize_simple_expression(
                "Tr(F_1 · F_3 · F_2 · F_4)", cyclic_order=True,
            )
        self.assertIsNone(core.canonicalize_simple_expression(
            "p_1 · F_2 · F_3 · F_2 · p_1", cyclic_order=True,
        ).expression)
        self.assertIsNone(core.canonicalize_simple_expression(
            "p_1 · F_1 · p_2", cyclic_order=True,
        ).expression)


class CyclicCleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = core.ScatteringAmplitudeTokenizer(
            max_particles=8, max_sequence_length=None,
        )
        cls.zero_points = core.build_kinematic_points(
            base_seed=core.DEFAULT_ZERO_SEED, checks_per_mode=2, energy_scale=2.0,
        )
        cls.validation_points = core.build_kinematic_points(
            base_seed=core.DEFAULT_VALIDATION_SEED, checks_per_mode=2, energy_scale=2.0,
        )

    def prepare(self, simple: str, scrambled: str):
        return core.prepare_pair(
            simple, scrambled, tokenizer=self.tokenizer,
            zero_points=self.zero_points, validation_points=self.validation_points,
            zero_tolerance=core.DEFAULT_ZERO_TOL,
            tol_abs=core.DEFAULT_TOL_ABS, tol_rel=core.DEFAULT_TOL_REL,
            max_subset_terms=6, max_tokens=2048, cyclic_order=True,
        )

    def test_cyclic_cleaning_prunes_zero_subsets_and_keeps_token_semantics(self) -> None:
        zero = "p_1 · p_2 + p_1 · p_3 + p_1 · p_4"
        self.assertIsNone(self.prepare(zero, zero))
        prepared = self.prepare(TRACE + " + " + zero, TRACE)
        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.simple, TRACE)
        self.assertEqual(prepared.numerical_zero_terms_removed, 3)
        chain_pair = self.prepare(FORWARD_CHAIN, "-" + REVERSE_CHAIN)
        self.assertIsNotNone(chain_pair)
        self.assertIn(FORWARD_CHAIN, chain_pair.simple)
        for pair in (prepared, chain_pair):
            self.assertEqual(self.tokenizer.encode_infix(pair.simple), pair.simple_tokens)
            self.assertEqual(self.tokenizer.encode_infix(pair.scrambled), pair.scrambled_tokens)
            self.assertTrue(core.numerically_equivalent(
                pair.simple, pair.scrambled, self.validation_points,
                tol_abs=core.DEFAULT_TOL_ABS, tol_rel=core.DEFAULT_TOL_REL,
            ))

    def test_independent_validation_rejects_unequal_source(self) -> None:
        self.assertFalse({point.name for point in self.zero_points} & {
            point.name for point in self.validation_points
        })
        with self.assertRaisesRegex(ValueError, "not equivalent"):
            self.prepare(TRACE, "2*" + TRACE)


class CyclicExactCountTests(unittest.TestCase):
    def args(self, directory: Path):
        return core.build_parser().parse_args([
            "--samples", "3", "--jobs", "1",
            "--candidate-batch-size", "4", "--generator-batch-size", "4",
            "--max-candidates-factor", "4", "--zero-checks", "1",
            "--validation-checks", "1", "--progress-every", "0",
            "--raw-out", str(directory / "raw.csv"),
            "--tok-out", str(directory / "tok.csv"),
            "--report-out", str(directory / "report.json"),
        ])

    def test_duplicate_and_rejected_batches_refill_to_exact_aligned_count(self) -> None:
        valid = [(f"{coefficient}*{TRACE}",) * 2 for coefficient in (1, 2, 3)]
        zero = ("p_1 · F_1 · p_2",) * 2
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            with mock.patch.object(core, "build_dataset_batched", side_effect=[
                [zero, valid[0], valid[0], valid[1]],
                [valid[0], valid[2]],
            ]) as generate:
                stats, report = core.generate_to_files(args, cyclic_order=True)
            self.assertEqual(stats.accepted, args.samples)
            self.assertEqual(stats.generation_batches, 2)
            self.assertEqual(stats.exact_zero_targets_rejected, 1)
            self.assertEqual(stats.duplicate_rejections, 2)
            for call in generate.call_args_list:
                self.assertIs(call.kwargs["cyclic_order"], True)
                self.assertIs(call.kwargs["validate"], False)
            self.assertNotEqual(generate.call_args_list[0].kwargs["seed"],
                                generate.call_args_list[1].kwargs["seed"])
            self.assertEqual(report["settings"]["particle_order"], [1, 2, 3, 4])
            self.assertEqual(report["settings"]["ordering_reference"],
                             "data/data_ym/gluon4feyn1234.csv.gz")
            self.assertEqual(report["settings"]["canonicalization_policy"],
                             "forward_cyclic_canonical_nonzero_v1")
            with Path(args.raw_out).open(newline="") as raw_handle, \
                 Path(args.tok_out).open(newline="") as token_handle:
                raw_rows = list(csv.DictReader(raw_handle))
                token_rows = list(csv.DictReader(token_handle))
            self.assertEqual(len(raw_rows), args.samples)
            self.assertEqual(len(token_rows), args.samples)
            self.assertEqual(len({tuple(row.values()) for row in raw_rows}), args.samples)
            tokenizer = core.ScatteringAmplitudeTokenizer(max_particles=8)
            for raw, token in zip(raw_rows, token_rows):
                for column in ("simple", "scrambled"):
                    self.assertEqual(tokenizer.encode_infix(raw[column]), json.loads(token[column]))

    def test_exhausted_budget_publishes_no_partial_files_or_overwrites(self) -> None:
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                args = self.args(directory)
                args.max_candidates_factor = 1
                args.overwrite = existing
                destinations = [Path(args.raw_out), Path(args.tok_out), Path(args.report_out)]
                if existing:
                    for path in destinations:
                        path.write_text("original " + path.name)
                with mock.patch.object(core, "build_dataset_batched", return_value=[
                    (TRACE, TRACE), (TRACE, TRACE),
                    ("p_1 · F_1 · p_2",) * 2, ("p_1 · F_1 · p_2",) * 2,
                ]), self.assertRaisesRegex(RuntimeError, "accepted only 1/3"):
                    core.generate_to_files(args, cyclic_order=True)
                if existing:
                    for path in destinations:
                        self.assertEqual(path.read_text(), "original " + path.name)
                else:
                    self.assertEqual(list(directory.iterdir()), [])
                self.assertFalse(list(directory.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
