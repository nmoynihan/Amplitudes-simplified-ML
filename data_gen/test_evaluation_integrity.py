"""Preflight tests use actual checkpoint-compatible prefix tokenization."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_testing.evaluation_integrity import InputIntegrityError, validate_input_integrity


class InputIntegrityTests(unittest.TestCase):
    def setUp(self):
        from data_gen.data_gen_gravity.v2_validation import NumericalContext
        from data_testing import evaluate_model
        self.tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
        self.evaluator = evaluate_model
        self.points = {"3s2h": NumericalContext("3s2h", seed=43171).points[:2]}
        patch = mock.patch.multiple(
            evaluate_model, NUMERIC_BACKEND="gravity", NUMERIC_TOL_ABS=2e-9, NUMERIC_TOL_REL=2e-8,
        )
        patch.start()
        self.addCleanup(patch.stop)

    def compare(self, original, decoded, process):
        return self.evaluator.numerical_comparison(original, decoded, self.points, gravity_process=process)

    def validate(self, source, *, target=None, tokens=None, compare=None, reference_provided=False):
        raw = {"scrambled": source, "simple": target or source}
        token_row = {
            "scrambled": self.tokenizer.encode_infix(source) if tokens is None else tokens,
            "simple": self.tokenizer.encode_infix(target or source),
        }
        return validate_input_integrity(
            [raw], [token_row], self.tokenizer, ["3s2h"], compare or self.compare,
            reference_provided=reference_provided,
        )[0]

    def test_real_rationals_traces_chains_and_repeats_preserve_meaning(self):
        examples = (
            "(2*(p_1·F_4·p_2))/(3*(p_4·p_5))",
            "Tr(F_4·F_5)*Tr(F_4·F_5)/(24*(p_4·p_5)*(p_4·p_5))",
            "(p_1·F_4·F_5·p_2)/(p_4·p_5)",
            "(p_1·p_2)/((p_1·p_3)+(p_1·p_4))",
        )
        for source in examples:
            with self.subTest(source=source):
                result = self.validate(source)
                evidence = result["source"]
                self.assertEqual(evidence["status"], "verified")
                self.assertTrue(evidence["exact_check"]["ok"])
                self.assertTrue(evidence["numerical_check"]["equivalent"])
                self.assertEqual(evidence["original_expression"], source)
                self.assertEqual(evidence["original_tokens"], self.tokenizer.encode_infix(source))
                self.assertEqual(evidence["bos_eos_token_count"], evidence["content_token_count"] + 2)
                self.assertIsNone(result["reference"])
                json.dumps(result, allow_nan=False)

    def test_independent_target_is_checked_separately(self):
        result = self.validate("p_1·p_2", target="p_2·p_1", reference_provided=True)
        self.assertTrue(result["reference_provided"])
        self.assertEqual(result["reference"]["status"], "verified")
        self.assertEqual(result["reference"]["original_expression"], "p_2·p_1")

    def test_absent_independent_target_skips_internal_placeholder(self):
        result = validate_input_integrity(
            [{"scrambled": "p_1·p_2", "simple": "bad placeholder"}],
            [{"scrambled": self.tokenizer.encode_infix("p_1·p_2"), "simple": []}],
            self.tokenizer, ["3s2h"], self.compare, reference_provided=False,
        )[0]
        self.assertIsNone(result["reference"])
        self.assertFalse(result["reference_provided"])

    def test_ambiguous_adjacent_numeric_leaves_fail_explicitly(self):
        for original in ("2/3", "(p_1·p_2)^2/2"):
            with self.subTest(original=original), self.assertRaisesRegex(InputIntegrityError, "representation_error"):
                self.validate(original)

    def test_exact_mismatch_cannot_pass_even_if_positive_helicity_samples_agree(self):
        falsely_accept = mock.Mock(return_value={"equivalent": True, "status": "equivalent"})
        with self.assertRaisesRegex(InputIntegrityError, "exact expression semantics") as context:
            self.validate("2*3", tokens=self.tokenizer.encode_infix("23"), compare=falsely_accept)
        falsely_accept.assert_not_called()
        self.assertEqual(context.exception.record["source"]["exact_check"]["reason"], "token_semantics_mismatch")

    def test_decoder_does_not_silently_strip_special_or_unknown_ids(self):
        good = self.tokenizer.encode_infix("p_1·p_2")
        for bad in (0, 1, 2, 3, 999, -1, True, "25"):
            tokens = good[:1] + [bad] + good[1:]
            with self.subTest(token=bad), self.assertRaisesRegex(InputIntegrityError, "representation_error"):
                self.validate("p_1·p_2", tokens=tokens)

    def test_strict_numerics_reject_source_suffix_ignored_by_tokenizer(self):
        with self.assertRaisesRegex(InputIntegrityError, "input_integrity_error") as context:
            self.validate("p_1·p_2 @")
        self.assertEqual(context.exception.record["source"]["numerical_check"]["status"], "invalid_expression")

    def test_wrong_process_symbol_is_not_validated_by_formatting_agreement(self):
        with self.assertRaisesRegex(InputIntegrityError, "unsupported_expression"):
            self.validate("p_1·F_1·p_2")

    def test_source_singular_at_all_samples_retains_evaluation_error(self):
        # The gravity backend raises ValueError("Division by zero"), whereas
        # arithmetic exceptions/nonfinite outputs have replacement semantics.
        with self.assertRaisesRegex(InputIntegrityError, "numerical_evaluation_error"):
            self.validate("(p_1·p_2)/0")

    def test_numerical_check_must_pass_even_after_exact_agreement(self):
        compare = mock.Mock(return_value={"equivalent": False, "status": "numerical_mismatch"})
        with self.assertRaisesRegex(InputIntegrityError, "numerical_mismatch"):
            self.validate("p_1·p_2", compare=compare)
        compare.assert_called_once()

    def test_missing_exact_backend_can_use_strict_numeric_validation(self):
        with mock.patch.object(self.evaluator, "NUMERIC_BACKEND", "sqed"):
            from data_gen import gen_data
            points = [gen_data.generate_kinematics(4, M=2, pol_mode="coulomb", seed=43271)]
            result = validate_input_integrity(
                [{"scrambled": "p_1·p_2"}],
                [{"scrambled": self.tokenizer.encode_infix("p_1·p_2")}],
                self.tokenizer, [None],
                lambda original, decoded, process: self.evaluator.numerical_comparison(original, decoded, points),
                reference_provided=False,
            )[0]
        self.assertFalse(result["source"]["exact_check"]["available"])
        self.assertTrue(result["source"]["numerical_check"]["equivalent"])

    def test_misaligned_rows_metadata_and_provenance_fail_before_inference(self):
        raw = [{"scrambled": "0", "simple": "0"}]
        tokens = [{"scrambled": self.tokenizer.encode_infix("0"), "simple": self.tokenizer.encode_infix("0")}]
        for rows, token_rows, processes, references in (
            (raw, [], ["3s2h"], True),
            (raw, tokens, [], True),
            (raw, tokens, ["3s2h"], []),
            (raw, tokens, ["3s2h"], [1]),
            ([], [], [], False),
        ):
            with self.subTest(rows=rows, tokens=token_rows, processes=processes, references=references), self.assertRaises(ValueError):
                validate_input_integrity(rows, token_rows, self.tokenizer, processes, self.compare, reference_provided=references)


if __name__ == "__main__":
    unittest.main()
