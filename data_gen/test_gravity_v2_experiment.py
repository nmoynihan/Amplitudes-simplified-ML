"""Leakage and output-boundary regressions for the v2 experiment interface."""
import unittest

from data_gen.gravity_v2_experiment import (
    capacity_report, make_partition, oracle, score_tokens, tokenizer,
)


class ExperimentTests(unittest.TestCase):
    def test_partition_groups_descendants_even_across_categories(self):
        rows = [dict(family_id=f"family-{i}", process="3s2h",
                     assigned_category=category)
                for i in range(20) for category in ("rational", "trace")]
        split = make_partition(rows, seed=17, validation_fraction=0.2)
        fitting = {rows[i]["family_id"] for i in split["fitting_indices"]}
        validation = {rows[i]["family_id"] for i in split["validation_indices"]}
        self.assertFalse(fitting & validation)
        self.assertEqual(split["validation_rows"], 8)
        self.assertEqual(split, make_partition(rows, seed=17, validation_fraction=0.2))
        reverse = make_partition(rows[::-1], seed=17, validation_fraction=0.2)
        self.assertEqual(split["validation_family_ids"], reverse["validation_family_ids"])

    def test_reject_missing_family_and_single_family(self):
        with self.assertRaises(ValueError):
            make_partition([dict(process="3s2h")])
        with self.assertRaises(ValueError):
            make_partition([dict(process="3s2h", family_id="only")])

    def test_capacity_includes_bos_eos_and_decoder_target(self):
        architecture = dict(vocab_size=58, max_seq_len=5000)
        self.assertTrue(capacity_report(architecture, 4096, 4096, 4098)["passes"])
        with self.assertRaisesRegex(ValueError, "Decoder ceiling"):
            capacity_report(architecture, 4096, 400, 401)
        with self.assertRaisesRegex(ValueError, "4096"):
            capacity_report(architecture, 4097, 100, 4098)
        with self.assertRaisesRegex(ValueError, "positional"):
            capacity_report(dict(vocab_size=58, max_seq_len=4097), 4096, 100, 4097)

    def test_malformed_or_truncated_predictions_are_not_successes(self):
        ev, points = oracle(seed=970151, samples=1)
        tok = tokenizer()
        source = "(p_1·p_2)"
        invalid = score_tokens([2, 57, 3], source, "3s2h", tok, ev, points)
        self.assertEqual(invalid["status"], "parse_error")
        invalid = score_tokens([2, 0, 25, 3], source, "3s2h", tok, ev, points)
        self.assertEqual(invalid["status"], "parse_error")
        unfinished = score_tokens([2] + tok.encode_infix(source), source, "3s2h", tok, ev, points)
        self.assertFalse(unfinished["equivalent"])
        self.assertEqual(unfinished["status"], "truncated_output")
        valid = score_tokens([2] + tok.encode_infix(source) + [3], source, "3s2h", tok, ev, points)
        self.assertTrue(valid["equivalent"])


if __name__ == "__main__":
    unittest.main()
