"""Strict candidate records, with no checkpoint or dataset dependencies."""

import unittest
from unittest import mock

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_testing.evaluation_records import check_candidate, inspect_sequence


class EvaluationRecordTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
        self.source = "(p_1 · p_2) + 0"
        self.source_tokens = self.tokenizer.encode_infix(self.source)
        self.compare = mock.Mock(return_value={"status": "equivalent", "equivalent": True})

    def raw(self, expression):
        return [2, *self.tokenizer.encode_infix(expression), 3]

    def check(self, sequence, **kwargs):
        return check_candidate(
            sequence, self.tokenizer, source_expression=self.source,
            source_tokens=self.source_tokens, compare=self.compare, **kwargs,
        )

    def test_copy_is_equivalent_but_not_a_simplification(self):
        record = self.check(self.raw(self.source))
        self.assertTrue(record["num_eq_scrambled"])
        self.assertTrue(record["copies_input"])
        self.assertEqual(record["token_reduction"], 0)
        self.assertFalse(record["shorter_equivalent"])

    def test_missing_independent_reference_produces_no_target_metrics(self):
        record = self.check(self.raw(self.source))
        for key in ("num_eq_simple", "exact_token", "exact_string", "reference_check"):
            self.assertIsNone(record[key], key)
        self.assertFalse(record["reference_provided"])
        self.compare.assert_called_once()

    def test_completed_shorter_equivalent_expression_succeeds(self):
        record = self.check(self.raw("p_1 · p_2"))
        self.assertTrue(record["shorter_equivalent"])
        self.assertFalse(record["copies_input"])
        self.assertEqual(record["token_reduction"], 2)
        self.assertEqual(record["source_token_count"], len(self.source_tokens))

    def test_shorter_numerical_mismatch_is_not_success(self):
        self.compare.return_value = {
            "status": "mismatch", "equivalent": False, "first_failure": {"absolute_error": 3.0},
        }
        record = self.check(self.raw("0"))
        self.assertTrue(record["syntax_valid"])
        self.assertGreater(record["token_reduction"], 0)
        self.assertFalse(record["shorter_equivalent"])
        self.assertEqual(record["numerical_check"]["first_failure"]["absolute_error"], 3.0)

    def test_missing_eos_is_explicit_even_with_complete_parse(self):
        sequence = self.raw("p_1 · p_2")[:-1]
        record = self.check(sequence)
        self.assertEqual(record["raw_token_ids"], sequence)
        self.assertTrue(record["decode_ok"])
        self.assertEqual(record["completion_status"], "missing_eos")
        self.assertFalse(record["completed"])
        self.assertFalse(record["shorter_equivalent"])

    def test_bos_eos_and_trailing_padding_remain_visible(self):
        sequence = self.raw("p_1 · p_2") + [0, 0]
        record = self.check(sequence)
        self.assertTrue(record["token_valid"])
        self.assertTrue(record["completed"])
        self.assertEqual(record["raw_token_ids"], sequence)
        self.assertEqual(record["tokens"], self.tokenizer.encode_infix("p_1 · p_2"))
        self.assertEqual(record["content_token_count"], 3)

    def test_each_internal_special_token_is_rejected_without_repair(self):
        for token in (0, 1, 2):
            with self.subTest(token=token):
                sequence = [2, token, *self.source_tokens, 3]
                record = self.check(sequence)
                self.assertFalse(record["decode_ok"])
                self.assertFalse(record["syntax_valid"])
                self.assertIn("illegal_internal_special_token", record["sequence_errors"])
                self.assertEqual(record["raw_token_ids"], sequence)
                self.assertEqual(record["numerical_check"]["status"], "token_decode_error")
        self.compare.assert_not_called()

    def test_unmapped_tokens_including_checkpoint_extra_id_are_rejected(self):
        for token in (-1, 57, 999):
            with self.subTest(token=token):
                record = self.check([2, token, 3])
                self.assertFalse(record["token_valid"])
                self.assertIn("unmapped_token", record["sequence_errors"])
                self.assertEqual(record["unmapped_positions"], [1])
        self.compare.assert_not_called()

    def test_unexpected_post_eos_content_is_not_discarded(self):
        for suffix in ([4], [3], [0, 12], [57]):
            with self.subTest(suffix=suffix):
                sequence = self.raw("p_1 · p_2") + suffix
                record = self.check(sequence)
                self.assertFalse(record["decode_ok"])
                self.assertEqual(record["raw_token_ids"], sequence)
                self.assertIn("unexpected_content_after_eos", record["sequence_errors"])
        self.compare.assert_not_called()

    def test_missing_bos_empty_and_noninteger_sequences_are_rejected(self):
        for sequence, reason in (([], "missing_bos"), ([2, 3], "empty_prediction"),
                                 ([12, 3], "missing_bos"), ([2, "12", 3], "non_integer_token"),
                                 ([2, True, 3], "non_integer_token")):
            with self.subTest(sequence=sequence):
                record = self.check(sequence)
                self.assertFalse(record["decode_ok"])
                self.assertIn(reason, record["sequence_errors"])
        self.compare.assert_not_called()

    def test_prefix_must_be_complete_and_fully_consumed(self):
        # Binary + is missing one argument; two vector leaves leave a suffix.
        for sequence in ([2, 4, 25, 3], [2, 25, 26, 3]):
            with self.subTest(sequence=sequence):
                record = self.check(sequence)
                self.assertIn("prefix_decode_error", record["sequence_errors"])
                self.assertFalse(record["decode_ok"])
                self.assertIn("ValueError", record["decode_error"])
        self.compare.assert_not_called()

    def test_explicit_reference_is_checked_separately_from_source(self):
        reference = "p_1 · p_2"
        reference_tokens = self.tokenizer.encode_infix(reference)
        record = self.check(self.raw(reference), reference_expression=reference,
                            reference_tokens=reference_tokens)
        self.assertTrue(record["reference_provided"])
        self.assertTrue(record["exact_token"])
        self.assertTrue(record["exact_string"])
        self.assertTrue(record["num_eq_simple"])
        self.assertEqual(self.compare.call_count, 2)
        self.assertEqual(self.compare.call_args_list[0].args[0], self.source)
        self.assertEqual(self.compare.call_args_list[1].args[0], reference)

    def test_unsupported_scalar_or_process_symbol_cannot_be_syntax_valid(self):
        self.compare.return_value = {"status": "unsupported_expression", "equivalent": False}
        record = self.check(self.raw("p_1"))
        self.assertTrue(record["decode_ok"])
        self.assertFalse(record["syntax_valid"])
        self.assertFalse(record["shorter_equivalent"])

    def test_inconclusive_numeric_result_stays_distinct_from_mismatch(self):
        self.compare.return_value = {"status": "insufficient_valid_samples", "equivalent": False}
        record = self.check(self.raw("0"))
        self.assertTrue(record["syntax_valid"])
        self.assertEqual(record["numerical_check"]["status"], "insufficient_valid_samples")
        self.assertFalse(record["shorter_equivalent"])

    def test_trace_mixed_chain_rational_and_repeated_factors_decode(self):
        for expression in (
            "(2*(p_1 · p_2))/(3*(p_4 · p_5))",
            "Tr(F_4 · F_5)",
            "p_1 · F_4 · F_5 · p_2",
            "(p_1 · F_4 · p_2)/((p_4 · p_5)*(p_4 · p_5))",
        ):
            with self.subTest(expression=expression):
                record = inspect_sequence(self.raw(expression), self.tokenizer)
                self.assertTrue(record["decode_ok"], record["decode_error"])
                self.assertEqual(record["tokens"], self.tokenizer.encode_infix(expression))
                self.assertTrue(record["expr"])


if __name__ == "__main__":
    unittest.main()
