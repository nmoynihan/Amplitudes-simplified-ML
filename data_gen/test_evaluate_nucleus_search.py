"""Search regressions with mocked generation and mocked/real numeric checks.

Run from Amplitudes-simplified-ML with:
    python -m unittest data_gen.test_evaluate_nucleus_search
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import tempfile
import types
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_testing import evaluate_model as evaluator
from data_testing import evaluate_nucleus_search as search


class NucleusSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8, max_sequence_length=None
        )
        self.model = types.SimpleNamespace(device="cpu")
        self.config = search.RunConfig(max_attempts=3, beam_size=4, max_length=64)
        self.row = search.InputRow(
            row_index=7,
            row_id="original-row-42",
            input_expression="p_1 · p_2",
            # The simple target is deliberately different: only scrambled counts.
            simple_expression="0",
        )
        with search.evaluator_settings(self.config):
            self.cache = evaluator.precompute_kinematics()

    def sequence(self, expression: str) -> list[int]:
        return [2, *self.tokenizer.encode_infix(expression), 3]

    def output(self, top1: str | list[int], *beams: str | list[int]):
        def encode(candidate: str | list[int]) -> list[int]:
            return self.sequence(candidate) if isinstance(candidate, str) else candidate

        return torch.tensor([encode(top1)], dtype=torch.long), [
            [encode(candidate) for candidate in beams]
        ]

    def run_search(self, decoder, *, row=None, config=None, cache=None):
        config = config or self.config
        with search.evaluator_settings(config):
            return search.search_amplitude(
                self.model,
                self.tokenizer,
                row or self.row,
                config,
                self.cache if cache is None else cache,
                decode_fn=decoder,
            )

    def test_first_attempt_success_stops_before_remaining_candidates(self) -> None:
        match = "p_2 · p_1 + 0"
        tokens = self.tokenizer.encode_infix(match)
        expression = self.tokenizer.decode_infix(tokens)
        self.assertGreater(len(tokens), len(self.tokenizer.encode_infix(self.row.input_expression)))
        config = replace(self.config, tol_abs=1e-7, tol_rel=1e-6)
        # The later equivalent candidate is shorter, but the first match wins.
        decoder = mock.Mock(return_value=self.output(match, "p_2 · p_1", "1"))

        def numeric_match(*args, **kwargs):
            self.assertEqual(evaluator.resolve_numeric_tolerances(), (config.tol_abs, config.tol_rel))
            return True

        with mock.patch.object(
            evaluator, "numerically_equivalent_exprs", side_effect=numeric_match,
        ) as equivalent:
            result = self.run_search(decoder, config=config)

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["attempts_used"], 1)
        self.assertEqual(result["first_successful_attempt"], 1)
        self.assertEqual(result["candidates_checked"], 1)
        self.assertEqual(result["candidates_encountered"], 1)
        self.assertEqual(result["cache_hits"], 0)
        self.assertEqual(result["matching_token_ids"], tokens)
        self.assertEqual(result["matching_expression"], expression)
        self.assertEqual(result["input_token_length"], len(self.tokenizer.encode_infix(self.row.input_expression)))
        self.assertEqual(result["matching_token_length"], len(tokens))
        decoder.assert_called_once()
        equivalent.assert_called_once_with(
            expression, self.row.input_expression, self.cache, gravity_process=None,
        )

    def test_later_success_stops_and_always_conditions_on_original_input(self) -> None:
        config = replace(self.config, max_attempts=5)
        outputs = iter([
            self.output("0"), self.output("1"),
            self.output("0", "p_2 · p_1", "2"),
        ])
        sources = []

        def decode(model, source, **kwargs):
            self.assertIs(model, self.model)
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(kwargs["decoding_method"], "nucleus")
            self.assertEqual(kwargs["beam_size"], self.config.beam_size)
            self.assertEqual(kwargs["p_nucleus"], self.config.p_nucleus)
            self.assertEqual(kwargs["temperature_nucleus"], self.config.temperature_nucleus)
            sources.append(source.tolist())
            return next(outputs)

        decoder = mock.Mock(side_effect=decode)
        with mock.patch.object(
            evaluator, "numerically_equivalent_exprs", side_effect=[False, False, True],
        ) as equivalent:
            result = self.run_search(decoder, config=config)

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["attempts_used"], 3)
        self.assertEqual(result["first_successful_attempt"], 3)
        self.assertEqual(result["candidates_checked"], 3)
        self.assertEqual(result["candidates_encountered"], 4)
        self.assertEqual(result["cache_hits"], 1)
        tokens = self.tokenizer.encode_infix("p_2 · p_1")
        expression = self.tokenizer.decode_infix(tokens)
        self.assertEqual(result["matching_token_ids"], tokens)
        self.assertEqual(result["matching_expression"], expression)
        self.assertEqual(equivalent.call_args_list, [
            mock.call(candidate, self.row.input_expression, self.cache, gravity_process=None)
            for candidate in ("0", "1", expression)
        ])
        self.assertEqual(decoder.call_count, 3)
        self.assertEqual(sources, [[self.sequence(self.row.input_expression)]] * 3)

    def test_non_top1_match_uses_returned_beam_order_and_stops(self) -> None:
        decoder = mock.Mock(return_value=self.output("0", "1", "p_2 · p_1", "2"))
        with mock.patch.object(
            evaluator,
            "numerically_equivalent_exprs",
            wraps=evaluator.numerically_equivalent_exprs,
        ) as equivalent:
            result = self.run_search(decoder)

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["first_successful_attempt"], 1)
        self.assertEqual(result["candidates_checked"], 3)
        decoder.assert_called_once()
        visited = []
        for call in equivalent.call_args_list:
            expressions = call.args[:2]
            visited.extend(expr for expr in expressions if expr != self.row.input_expression)
        self.assertEqual(visited, ["0", "1", self.tokenizer.decode_infix(self.tokenizer.encode_infix("p_2 · p_1"))])
        self.assertEqual(result["matching_token_length"], len(self.tokenizer.encode_infix("p_2 · p_1")))

    def test_lengths_count_original_source_and_prediction_content_only(self) -> None:
        # An equivalent supplied token sequence need not have the same length as
        # either the raw reference expression or the paired simple diagnostic.
        input_tokens = self.tokenizer.encode_infix("0 + (p_1 · p_2 + 0)")
        matching_tokens = self.tokenizer.encode_infix("p_2 · p_1 + 0")
        row = replace(self.row, input_tokens=input_tokens)
        self.assertGreater(len(input_tokens), len(self.tokenizer.encode_infix(row.input_expression)))
        for candidate in (
            [2, *matching_tokens, 3, 0, 0],
            [2, *matching_tokens, 0, 0],
            [*matching_tokens, 3, 0],
        ):
            with self.subTest(candidate=candidate):
                decoder = mock.Mock(return_value=self.output(candidate))
                result = self.run_search(decoder, row=row)
                self.assertEqual(result["status"], "matched")
                self.assertEqual(result["input_token_length"], len(input_tokens))
                self.assertEqual(result["matching_token_length"], len(matching_tokens))
                self.assertEqual(result["matching_token_ids"], matching_tokens)
                self.assertEqual(decoder.call_args.args[1].tolist(), [[2, *input_tokens, 3]])

    def test_budget_exhaustion_counts_calls_independently_of_beam_size(self) -> None:
        config = replace(self.config, max_attempts=2, beam_size=7)
        decoder = mock.Mock(return_value=self.output("0", "1", "2"))
        with mock.patch.object(
            evaluator, "numerically_equivalent_exprs", return_value=False,
        ) as equivalent:
            result = self.run_search(decoder, config=config)

        self.assertEqual(result["status"], "exhausted")
        self.assertEqual(result["attempts_used"], 2)
        self.assertIsNone(result["first_successful_attempt"])
        self.assertEqual(decoder.call_count, 2)
        self.assertEqual(result["candidates_checked"], 3)
        self.assertEqual(result["candidates_encountered"], 6)
        self.assertEqual(result["cache_hits"], 3)
        self.assertIsNone(result["matching_expression"])
        self.assertIsNone(result["matching_token_ids"])
        self.assertEqual(result["input_token_length"], len(self.tokenizer.encode_infix(self.row.input_expression)))
        self.assertIsNone(result["matching_token_length"])
        self.assertEqual(equivalent.call_args_list, [
            mock.call(candidate, self.row.input_expression, self.cache, gravity_process=None)
            for candidate in ("0", "1", "2")
        ])
        self.assertTrue(all(call.kwargs["beam_size"] == 7 for call in decoder.call_args_list))

    def test_duplicates_and_malformed_hypotheses_consume_the_budget(self) -> None:
        malformed = [2, self.tokenizer.vocab["*"], self.tokenizer.vocab["p_1"], 3]
        decoder = mock.Mock(return_value=self.output(malformed, "0", malformed, "0"))
        with mock.patch.object(
            evaluator,
            "numerically_equivalent_exprs",
            wraps=evaluator.numerically_equivalent_exprs,
        ) as equivalent:
            result = self.run_search(decoder)

        self.assertEqual(result["status"], "exhausted")
        self.assertEqual(result["attempts_used"], 3)
        self.assertEqual(decoder.call_count, 3)
        self.assertEqual(result["candidates_checked"], 2)
        self.assertEqual(result["candidates_encountered"], 12)
        self.assertEqual(result["cache_hits"], 10)
        self.assertTrue(result["error_diagnostics"])
        # A repeated malformed expression is cached, and only "0" reaches numerics.
        equivalent.assert_called_once()

    def test_cache_is_local_to_an_amplitude(self) -> None:
        decoder = mock.Mock(return_value=self.output("0"))
        with mock.patch.object(
            evaluator,
            "numerically_equivalent_exprs",
            wraps=evaluator.numerically_equivalent_exprs,
        ) as equivalent:
            first = self.run_search(decoder)
            second = self.run_search(decoder)
        self.assertEqual(first["candidates_checked"], 1)
        self.assertEqual(second["candidates_checked"], 1)
        self.assertEqual(equivalent.call_count, 2)
        self.assertEqual(decoder.call_count, 6)

    def test_invalid_references_never_start_decoding(self) -> None:
        for expression in ("p_1", "p_1 · p_5", "e_1 · p_2", "1/0", ""):
            with self.subTest(expression=expression):
                decoder = mock.Mock()
                result = self.run_search(
                    decoder, row=replace(self.row, input_expression=expression)
                )
                self.assertEqual(result["status"], "invalid_input")
                self.assertEqual(result["attempts_used"], 0)
                self.assertEqual(result["candidates_checked"], 0)
                self.assertIsNone(result["input_token_length"])
                self.assertIsNone(result["matching_token_length"])
                self.assertTrue(result["error_diagnostics"])
                decoder.assert_not_called()

    def test_empty_kinematics_cache_is_invalid_input(self) -> None:
        decoder = mock.Mock()
        result = self.run_search(decoder, cache=[])
        self.assertEqual(result["status"], "invalid_input")
        self.assertEqual(result["attempts_used"], 0)
        self.assertTrue(result["error_diagnostics"])
        decoder.assert_not_called()

    def test_source_tokens_must_numerically_represent_original_reference(self) -> None:
        decoder = mock.Mock()
        result = self.run_search(
            decoder,
            row=replace(self.row, input_tokens=self.tokenizer.encode_infix("0")),
        )
        self.assertEqual(result["status"], "invalid_input")
        self.assertEqual(result["attempts_used"], 0)
        decoder.assert_not_called()

    def test_reference_failure_on_later_kinematics_point_is_invalid(self) -> None:
        decoder = mock.Mock()
        real_evaluate = evaluator.eval_numeric_expr

        def fail_late(*args, **kwargs):
            if args[1] is self.cache[1][0]:
                return math.nan
            return real_evaluate(*args, **kwargs)

        with mock.patch.object(evaluator, "eval_numeric_expr", side_effect=fail_late):
            result = self.run_search(decoder)

        self.assertEqual(result["status"], "invalid_input")
        self.assertEqual(result["attempts_used"], 0)
        self.assertIn("point 2", str(result["error_diagnostics"]))
        decoder.assert_not_called()

    def test_exact_tokens_do_not_bypass_numerical_equivalence(self) -> None:
        decoder = mock.Mock(return_value=self.output(self.row.input_expression))
        with mock.patch.object(evaluator, "numerically_equivalent_exprs", return_value=False) as equivalent:
            result = self.run_search(decoder)

        self.assertEqual(result["status"], "exhausted")
        self.assertEqual(result["attempts_used"], 3)
        equivalent.assert_called_once()

    def test_bad_simple_diagnostic_does_not_replace_valid_scrambled_reference(self) -> None:
        decoder = mock.Mock(return_value=self.output("p_2 · p_1"))
        result = self.run_search(
            decoder, row=replace(self.row, simple_expression="p_1 · p_8")
        )
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["attempts_used"], 1)

    def test_strict_sqed_field_strength_identity_can_be_longer_than_input(self) -> None:
        reference = "p_1 · F_2 · p_4"
        expanded = "(p_1 · p_2)*(e_2 · p_4)-(p_1 · e_2)*(p_2 · p_4)"
        self.assertGreater(
            len(self.tokenizer.encode_infix(expanded)),
            len(self.tokenizer.encode_infix(reference)),
        )
        config = replace(self.config, numeric_pol_modes=("coulomb", "covariant"))
        with search.evaluator_settings(config):
            cache = evaluator.precompute_kinematics()
        self.assertEqual(len(cache), 2 * config.numeric_samples)
        # Wrong value, illegal scalar, nonfinite expression, then the true identity.
        pole = "1/(p_1 · p_2-p_1 · p_2)"
        self.tokenizer.decode_infix(self.tokenizer.encode_infix(pole))
        decoder = mock.Mock(return_value=self.output("0", "p_1", pole, expanded))
        result = self.run_search(
            decoder, config=config, cache=cache,
            row=replace(self.row, input_expression=reference),
        )
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["candidates_checked"], 4)
        self.assertEqual(result["matching_token_ids"], self.tokenizer.encode_infix(expanded))

    def test_sampling_seed_reproduces_run_without_reseeding_attempts(self) -> None:
        def sampled_run():
            samples = []

            def decode(*args, **kwargs):
                samples.append(float(torch.rand(())))
                return self.output("0")

            search.seed_sampling(918)
            self.run_search(decode)
            return samples

        first = sampled_run()
        second = sampled_run()
        self.assertEqual(len(first), self.config.max_attempts)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), self.config.max_attempts)

    def test_numerical_match_requires_every_configured_sample_and_mode(self) -> None:
        config = replace(self.config, numeric_pol_modes=("coulomb", "covariant"))
        with search.evaluator_settings(config):
            cache = evaluator.precompute_kinematics()
        candidate = self.tokenizer.decode_infix(self.tokenizer.encode_infix("p_2 · p_1"))
        real_evaluate = evaluator.eval_numeric_expr
        candidate_checks = 0

        def differ_only_at_last_point(expression, *args, **kwargs):
            nonlocal candidate_checks
            value = real_evaluate(expression, *args, **kwargs)
            if expression == candidate:
                candidate_checks += 1
                if candidate_checks == len(cache):
                    return value + 1.0
            return value

        decoder = mock.Mock(return_value=self.output("p_2 · p_1"))
        with mock.patch.object(evaluator, "eval_numeric_expr", side_effect=differ_only_at_last_point):
            result = self.run_search(decoder, config=config, cache=cache)
        self.assertEqual(candidate_checks, len(cache))
        self.assertEqual(result["status"], "exhausted")

    def test_backend_default_tolerances_are_preserved_and_globals_restored(self) -> None:
        original = (
            evaluator.NUMERIC_BACKEND, evaluator.NUMERIC_TOL_ABS,
            evaluator.NUMERIC_TOL_REL, evaluator.NUMERIC_EQUIV_SEED,
        )
        for backend, expected in (
            ("sqed", (1e-12, 1e-10)),
            ("ym", (1e-10, 1e-8)),
            ("gravity", (2e-9, 2e-8)),
        ):
            with self.subTest(backend=backend):
                with search.evaluator_settings(replace(self.config, numeric_backend=backend)):
                    self.assertEqual(evaluator.resolve_numeric_tolerances(), expected)
        self.assertEqual(
            (
                evaluator.NUMERIC_BACKEND, evaluator.NUMERIC_TOL_ABS,
                evaluator.NUMERIC_TOL_REL, evaluator.NUMERIC_EQUIV_SEED,
            ),
            original,
        )

    def test_yang_mills_uses_all_gluon_polarizations(self) -> None:
        config = replace(self.config, numeric_backend="ym", numeric_samples=2)
        with search.evaluator_settings(config):
            cache = evaluator.precompute_kinematics()
        self.assertEqual(len(cache), 4)  # Both default modes for both samples.
        decoder = mock.Mock(return_value=self.output("p_2 · e_1"))
        result = self.run_search(
            decoder, config=config, cache=cache,
            row=replace(self.row, input_expression="e_1 · p_2"),
        )
        self.assertEqual(result["status"], "matched")

    def test_gravity_process_metadata_and_reference_modes(self) -> None:
        config = replace(
            self.config, numeric_backend="gravity", n_particles=5,
            numeric_samples=1,
        )
        with search.evaluator_settings(config):
            cache = evaluator.precompute_kinematics()
        self.assertEqual(len(cache["4s1h"]), 4)  # Three references plus gauge shift.
        decoder = mock.Mock(return_value=self.output("p_2 · p_1"))
        result = self.run_search(
            decoder, config=config, cache=cache,
            row=replace(self.row, process="4s1h"),
        )
        self.assertEqual(result["status"], "matched")

        missing_decoder = mock.Mock()
        missing = self.run_search(missing_decoder, config=config, cache=cache)
        self.assertEqual(missing["status"], "invalid_input")
        self.assertTrue(missing["error_diagnostics"])
        missing_decoder.assert_not_called()


class InputAndConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8, max_sequence_length=None
        )

    def write_csv(self, name, header, rows):
        path = self.directory / name
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if header is not None:
                writer.writerow(header)
            writer.writerows(rows)
        return path

    def load(self, path, **overrides):
        config = search.RunConfig(test_data=path, **overrides)
        with search.evaluator_settings(config):
            return search.load_inputs(config, self.tokenizer)

    def assert_saved_summary_matches_console(self, run_dir, console):
        report = (run_dir / "summary.txt").read_text(encoding="utf-8")
        # The startup dataset line precedes progress; the last one begins the
        # complete final report, which must be saved exactly as it was printed.
        final_start = console.rindex("Dataset:")
        self.assertEqual(report, console[final_start:])
        self.assertTrue(report.endswith("\n"))
        for name in ("results.csv", "config.json", "summary.json", "summary.txt"):
            self.assertTrue((run_dir / name).is_file())
            self.assertIn(str(run_dir / name), report)
        for label in ("Matched ", "Success rate ", "Attempts to success:", "Total runtime:"):
            self.assertIn(label, report)
        return report

    def test_raw_pairs_preserve_original_ids_order_and_duplicate_records(self) -> None:
        path = self.write_csv("pairs.csv", ["id", "simple", "scrambled"], [
            ["z", "0", "p_1 · p_2"],
            ["a", "0", "p_1 · p_2"],
            ["z", "1", "p_1 · p_3"],
        ])
        rows, total, detected = self.load(path, input_format="raw")
        self.assertEqual(total, 3)
        self.assertEqual(detected, "raw")
        self.assertEqual([row.row_id for row in rows], ["z", "a", "z"])
        self.assertEqual([row.row_index for row in rows], [1, 2, 3])
        self.assertEqual([row.source_line for row in rows], [2, 3, 4])
        self.assertEqual([row.input_expression for row in rows], ["p_1 · p_2", "p_1 · p_2", "p_1 · p_3"])

    def test_row_limit_keeps_total_count_and_does_not_refill_invalid_rows(self) -> None:
        path = self.write_csv("limited.csv", ["id", "amplitude"], [
            ["first", ""], ["second", "p_1 · p_2"], ["third", "p_1 · p_3"],
        ])
        rows, total, _ = self.load(path, max_rows=2)
        self.assertEqual(total, 3)
        self.assertEqual([row.row_id for row in rows], ["first", "second"])
        self.assertTrue(rows[0].error)
        self.assertIsNone(rows[1].error)

    def test_gzip_token_pair_uses_scrambled_and_keeps_malformed_rows(self) -> None:
        tokens = self.tokenizer.encode_infix("p_1 · p_2")
        simple_tokens = self.tokenizer.encode_infix("0")
        path = self.write_csv("pairs.csv.gz", ["id", "simple", "scrambled"], [
            ["9", json.dumps(simple_tokens), json.dumps(tokens)],
            ["2", json.dumps(simple_tokens), json.dumps(tokens)],
            ["bad", json.dumps(simple_tokens), "[true]"],
            ["last", json.dumps(simple_tokens), "[25,"],
        ])
        for input_format in ("auto", "token-pair"):
            with self.subTest(input_format=input_format):
                rows, total, _ = self.load(path, input_format=input_format)
                self.assertEqual(total, 4)
                self.assertEqual([row.row_id for row in rows], ["9", "2", "bad", "last"])
                self.assertEqual(rows[0].input_tokens, tokens)
                self.assertEqual(rows[0].input_expression, self.tokenizer.decode_infix(tokens))
                self.assertEqual(rows[0].simple_expression, "0")
                self.assertIsNone(rows[1].error)
                self.assertTrue(rows[2].error)
                self.assertTrue(rows[3].error)

    def test_custom_token_and_identity_columns_are_supported(self) -> None:
        tokens = self.tokenizer.encode_infix("p_1 · p_2")
        path = self.write_csv("tokens.csv", ["key", "prefix"], [
            ["original", json.dumps(tokens)],
        ])
        rows, total, detected = self.load(
            path, input_format="tokens", tokens_column="prefix", id_column="key"
        )
        self.assertEqual(total, 1)
        self.assertEqual(detected, "tokens")
        self.assertEqual(rows[0].row_id, "original")
        self.assertEqual(rows[0].input_tokens, tokens)

    def test_headerless_records_keep_original_id_and_physical_source_line(self) -> None:
        path = self.write_csv("feyn.csv", None, [
            [], ["amplitude-b", "p_1 · p_2"], [], ["amplitude-a", "p_1 · p_3"],
        ])
        rows, total, detected = self.load(path)
        self.assertEqual(total, 2)
        self.assertEqual(detected, "feyn")
        self.assertEqual([row.row_id for row in rows], ["amplitude-b", "amplitude-a"])
        self.assertEqual([row.source_line for row in rows], [2, 4])

    def test_gravity_metadata_aligns_by_record_id_without_deduplication(self) -> None:
        path = self.write_csv("gravity.csv", ["id", "simple", "scrambled"], [
            ["b", "0", "p_1 · p_2"], ["a", "0", "p_1 · p_2"],
        ])
        metadata = self.write_csv("metadata.csv", ["id", "scrambled", "process"], [
            ["b", "p_1 · p_2", "4s1h"], ["a", "p_1 · p_2", "3s2h"],
        ])
        rows, total, _ = self.load(
            path, numeric_backend="gravity", gravity_metadata_csv=metadata
        )
        self.assertEqual(total, 2)
        self.assertEqual([row.process for row in rows], ["4s1h", "3s2h"])
        self.assertTrue(all(row.error is None for row in rows))

        metadata = self.write_csv("misaligned.csv", ["id", "process"], [
            ["a", "4s1h"], ["b", "3s2h"],
        ])
        rows, _, _ = self.load(
            path, numeric_backend="gravity", gravity_metadata_csv=metadata
        )
        self.assertTrue(all(row.error and "mismatch" in row.error for row in rows))

    def test_gravity_token_rows_align_to_expression_metadata(self) -> None:
        expression = "p_1 · p_2"
        tokens = self.tokenizer.encode_infix(expression)
        path = self.write_csv("gravity_tok.csv", ["id", "simple", "scrambled"], [
            ["original-5", json.dumps(tokens), json.dumps(tokens)],
        ])
        metadata = self.write_csv("metadata.csv", ["id", "simple", "scrambled", "process"], [
            ["original-5", expression, expression, "4s1h"],
        ])
        rows, _, _ = self.load(
            path, input_format="token-pair", numeric_backend="gravity",
            gravity_metadata_csv=metadata,
        )
        self.assertIsNone(rows[0].error)
        self.assertEqual(rows[0].process, "4s1h")

    def test_explicit_gravity_process_conflict_is_reported_per_row(self) -> None:
        path = self.write_csv("gravity.csv", ["id", "amplitude", "process"], [
            ["a", "p_1 · p_2", "4s1h"], ["b", "p_1 · p_2", "3s2h"],
        ])
        rows, _, _ = self.load(
            path, numeric_backend="gravity", gravity_process="4s1h"
        )
        self.assertIsNone(rows[0].error)
        self.assertTrue(rows[1].error)
        self.assertIn("conflict", rows[1].error.lower())

    def test_cli_overrides_sampling_and_numeric_seeds_independently(self) -> None:
        config = search.parse_args([
            "--checkpoint", "weights.pt", "--test-data", "test.csv.gz",
            "--max-attempts", "9", "--beam-size", "2", "--sampling-seed", "4",
            "--numeric-seed", "77", "--numeric-samples", "2", "--max-length", "32",
            "--numeric-backend", "ym", "--numeric-pol-modes", "covariant",
            "--no-gravity-gauge-shift", "--tol-abs", "0.000001",
        ])
        search.validate_config(config)
        self.assertEqual(config.checkpoint, Path("weights.pt"))
        self.assertEqual(config.test_data, Path("test.csv.gz"))
        self.assertEqual(config.max_attempts, 9)
        self.assertEqual(config.beam_size, 2)
        self.assertEqual(config.sampling_seed, 4)
        self.assertEqual(config.numeric_seed, 77)
        self.assertEqual(config.numeric_pol_modes, ("covariant",))
        self.assertFalse(config.gravity_gauge_shift)
        self.assertEqual(config.tol_abs, 1e-6)

    def test_invalid_configuration_is_rejected_before_inference(self) -> None:
        invalid_options = (
            {"max_attempts": 0}, {"beam_size": 0}, {"max_rows": 0},
            {"max_length": 1}, {"p_nucleus": 0}, {"p_nucleus": 1.1},
            {"p_nucleus": math.nan}, {"temperature_nucleus": 0},
            {"temperature_nucleus": math.inf}, {"sampling_seed": -1},
            {"numeric_seed": 2**32}, {"numeric_samples": 0},
            {"n_particles": 3}, {"n_particles": 9}, {"mass": -1.0},
            {"numeric_backend": "ym", "energy_scale": 0},
            {"numeric_backend": "gravity", "n_particles": 4},
            {"numeric_pol_modes": ()}, {"tol_abs": -1e-6},
            {"tol_abs": 0, "tol_rel": 0},
        )
        for overrides in invalid_options:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    search.validate_config(search.RunConfig(**overrides))

    def test_runner_saves_success_and_continues_with_subsequent_rows(self) -> None:
        path = self.write_csv("run.csv", ["id", "simple", "scrambled"], [
            ["first", "0", "p_1 · p_2"],
            ["second", "0", "p_1 · p_3 + 0"],
            ["invalid", "0", "p_1 · p_8"],
        ])
        checkpoint = self.directory / "model.pt"
        checkpoint.touch()
        config = search.RunConfig(
            checkpoint=checkpoint, test_data=path, output_dir=self.directory / "results",
            device="cpu", max_length=32, max_attempts=3,
        )
        model = types.SimpleNamespace(
            device="cpu", max_seq_len=64, vocab_size=self.tokenizer.vocab_size
        )
        decode_calls = 0
        sources = []
        completed_before_second_row = []
        matching_tokens = self.tokenizer.encode_infix("0 + (p_2 · p_1 + 0)")
        first_input_length = len(self.tokenizer.encode_infix("p_1 · p_2"))
        second_input_length = len(self.tokenizer.encode_infix("p_1 · p_3 + 0"))
        prediction_length = len(matching_tokens)
        self.assertLess(first_input_length, second_input_length)
        self.assertGreater(prediction_length, first_input_length)
        console = io.StringIO()

        def decode(model, source, **kwargs):
            nonlocal decode_calls
            decode_calls += 1
            sources.append(source.tolist())
            if decode_calls == 2:
                result_files = list(config.output_dir.glob("*/results.csv"))
                self.assertEqual(len(result_files), 1)
                with result_files[0].open(newline="", encoding="utf-8") as handle:
                    completed_before_second_row.extend(csv.DictReader(handle))
            tokens = matching_tokens if decode_calls == 1 else self.tokenizer.encode_infix("0")
            # First-row candidates after the match must never reach numerics.
            beams = [[2, *self.tokenizer.encode_infix("1"), 3]] if decode_calls == 1 else []
            return torch.tensor([[2, *tokens, 3]]), [beams]

        with (
            mock.patch.object(evaluator, "load_model", return_value=model) as load_model,
            mock.patch.object(evaluator, "decode_with_model", side_effect=decode) as decoder,
            mock.patch.object(
                evaluator, "numerically_equivalent_exprs", side_effect=[True, False],
            ) as equivalent,
            mock.patch.object(search, "format_summary", wraps=search.format_summary) as format_summary,
            redirect_stdout(console),
        ):
            run_dir = search.run(config)

        load_model.assert_called_once()
        self.assertEqual(decoder.call_count, 1 + config.max_attempts)
        self.assertEqual(sources, [
            [[2, *self.tokenizer.encode_infix(expression), 3]]
            for expression in ["p_1 · p_2", *(["p_1 · p_3 + 0"] * config.max_attempts)]
        ])
        self.assertEqual(
            [call.args[:2] for call in equivalent.call_args_list],
            [(self.tokenizer.decode_infix(matching_tokens), "p_1 · p_2"), ("0", "p_1 · p_3 + 0")],
        )
        with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["row_id"] for row in rows], ["first", "second", "invalid"])
        self.assertEqual([row["status"] for row in rows], ["matched", "exhausted", "invalid_input"])
        self.assertEqual(completed_before_second_row, rows[:1])
        self.assertEqual([row["attempts_used"] for row in rows], ["1", "3", "0"])
        self.assertEqual([row["first_successful_attempt"] for row in rows], ["1", "", ""])
        self.assertEqual([row["candidates_checked"] for row in rows], ["1", "1", "0"])
        self.assertEqual([row["candidates_encountered"] for row in rows], ["1", "3", "0"])
        self.assertEqual([row["cache_hits"] for row in rows], ["0", "2", "0"])
        self.assertEqual(json.loads(rows[0]["matching_token_ids"]), matching_tokens)
        self.assertEqual(rows[0]["matching_expression"], self.tokenizer.decode_infix(matching_tokens))
        self.assertEqual([row["input_token_length"] for row in rows], [
            str(first_input_length), str(second_input_length), "",
        ])
        self.assertEqual([row["matching_token_length"] for row in rows], [str(prediction_length), "", ""])
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["success_rate_denominator"], 2)
        self.assertEqual(summary["invalid_input"], 1)
        self.assertEqual(summary["attempts_used_total"], 4)
        valid_mean = (first_input_length + second_input_length) / 2
        self.assertEqual(summary["token_lengths"], {
            "valid_searched_inputs": {"count": 2, "mean": valid_mean},
            "matched_inputs": {"count": 1, "mean": first_input_length},
            "matched_predictions": {"count": 1, "mean": prediction_length},
        })
        format_summary.assert_called_once_with(summary, run_dir)
        report = self.assert_saved_summary_matches_console(run_dir, console.getvalue())
        self.assertIn(
            f"Mean original input length (valid searched rows: matched + exhausted, n=2): {valid_mean:.2f} content tokens",
            report,
        )
        self.assertIn(
            f"Mean original input length (matched rows only, n=1): {first_input_length:.2f} content tokens",
            report,
        )
        self.assertIn(
            f"Mean nucleus prediction length (same matched rows, first numerical match, n=1): {prediction_length:.2f} content tokens",
            report,
        )
        resolved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(resolved["sampling_seed"], config.sampling_seed)
        self.assertEqual(resolved["numeric_seed"], config.numeric_seed)
        self.assertEqual(resolved["tol_abs"], 1e-12)

    def test_runner_reports_empty_matching_and_valid_populations(self) -> None:
        checkpoint = self.directory / "model.pt"
        checkpoint.touch()
        for population, expressions, valid_count in (
            ("exhausted", ["p_1 · p_2", "p_1 · p_3 + 0"], 2),
            ("invalid", ["p_1 · p_8", ""], 0),
        ):
            with self.subTest(population=population):
                path = self.write_csv(f"{population}.csv", ["amplitude"], [[expr] for expr in expressions])
                config = search.RunConfig(
                    checkpoint=checkpoint, test_data=path, output_dir=self.directory / "results",
                    device="cpu", max_length=32, max_attempts=1,
                )
                model = types.SimpleNamespace(
                    device="cpu", max_seq_len=64, vocab_size=self.tokenizer.vocab_size,
                )
                console = io.StringIO()
                with (
                    mock.patch.object(evaluator, "load_model", return_value=model),
                    mock.patch.object(
                        evaluator, "decode_with_model",
                        return_value=(torch.tensor([[2, *self.tokenizer.encode_infix("0"), 3]]), [[]]),
                    ) as decoder,
                    mock.patch.object(evaluator, "numerically_equivalent_exprs", return_value=False),
                    redirect_stdout(console),
                ):
                    run_dir = search.run(config)
                self.assertEqual(decoder.call_count, valid_count)
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                valid_mean = (
                    sum(len(self.tokenizer.encode_infix(expr)) for expr in expressions) / valid_count
                    if valid_count else None
                )
                self.assertEqual(summary["token_lengths"], {
                    "valid_searched_inputs": {"count": valid_count, "mean": valid_mean},
                    "matched_inputs": {"count": 0, "mean": None},
                    "matched_predictions": {"count": 0, "mean": None},
                })
                with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row["matching_token_length"] for row in rows], ["", ""])
                self.assertEqual([row["matching_token_ids"] for row in rows], ["", ""])
                if not valid_count:
                    self.assertEqual([row["input_token_length"] for row in rows], ["", ""])
                report = self.assert_saved_summary_matches_console(run_dir, console.getvalue())
                for label in (
                    "Mean original input length (matched rows only, n=0)",
                    "Mean nucleus prediction length (same matched rows, first numerical match, n=0)",
                ):
                    self.assertIn(f"{label}: N/A content tokens", report)
                if not valid_count:
                    self.assertIn(
                        "Mean original input length (valid searched rows: matched + exhausted, n=0): N/A content tokens",
                        report,
                    )
                    self.assertIsNone(summary["success_rate"])
                    self.assertIn("Success rate N/A (no valid inputs)", report)

    def test_empty_summary_has_null_token_means_and_zero_counts(self) -> None:
        summary = search.summarize([], dataset_rows=0, runtime=0.0)
        self.assertEqual(summary["token_lengths"], {
            "valid_searched_inputs": {"count": 0, "mean": None},
            "matched_inputs": {"count": 0, "mean": None},
            "matched_predictions": {"count": 0, "mean": None},
        })
        report = search.format_summary(summary, self.directory)
        self.assertEqual(report.count("n=0): N/A content tokens"), 3)

    def test_checkpoint_target_capacity_is_checked_before_decoding(self) -> None:
        path = self.write_csv("run.csv", ["amplitude"], [["p_1 · p_2"]])
        checkpoint = self.directory / "model.pt"
        checkpoint.touch()
        config = search.RunConfig(
            checkpoint=checkpoint, test_data=path, output_dir=self.directory / "results",
            device="cpu", max_length=65,
        )
        model = types.SimpleNamespace(
            device="cpu", max_seq_len=64, vocab_size=self.tokenizer.vocab_size
        )
        with mock.patch.object(evaluator, "load_model", return_value=model):
            with mock.patch.object(evaluator, "decode_with_model") as decoder:
                with self.assertRaisesRegex(ValueError, "target capacity"):
                    search.run(config)
        decoder.assert_not_called()

    def test_source_beyond_checkpoint_capacity_is_reported_without_decoding(self) -> None:
        path = self.write_csv("run.csv", ["id", "amplitude"], [["too-long", "p_1 · p_2"]])
        checkpoint = self.directory / "model.pt"
        checkpoint.touch()
        config = search.RunConfig(
            checkpoint=checkpoint, test_data=path, output_dir=self.directory / "results",
            device="cpu", max_length=4,
        )
        model = types.SimpleNamespace(
            device="cpu", max_seq_len=4, vocab_size=self.tokenizer.vocab_size
        )
        with mock.patch.object(evaluator, "load_model", return_value=model):
            with mock.patch.object(evaluator, "decode_with_model") as decoder:
                with redirect_stdout(io.StringIO()):
                    run_dir = search.run(config)
        decoder.assert_not_called()
        with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
            result = next(csv.DictReader(handle))
        self.assertEqual(result["row_id"], "too-long")
        self.assertEqual(result["status"], "invalid_input")
        self.assertEqual(result["attempts_used"], "0")
        self.assertEqual(result["input_token_length"], "")
        self.assertEqual(result["matching_token_length"], "")
        self.assertIn("source capacity", result["error_diagnostics"])


if __name__ == "__main__":
    unittest.main()
