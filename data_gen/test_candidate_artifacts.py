"""Tiny mocked evaluate_mode runs exercise complete retained-candidate evidence."""

import csv
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_testing import evaluate_model as evaluator


class CandidateArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
        self.source = "(p_1 · p_2) + 0"
        self.target = "p_1 · p_2"
        self.token_rows = [{name: self.tokenizer.encode_infix(expression) for name, expression in
                            (("simple", self.target), ("scrambled", self.source))}]
        self.raw_rows = [{"simple": self.target, "scrambled": self.source}]
        source, target = self.raw(self.source), self.raw(self.target)
        self.dataset = [{"input": torch.tensor(source), "target": torch.tensor(target),
                         "input_length": len(source), "target_length": len(target)}]
        self.model = types.SimpleNamespace(device="cpu", vocab_size=58)

    def raw(self, expression):
        return [2, *self.tokenizer.encode_infix(expression), 3]

    def hypotheses(self, sequences):
        return [{
            "raw_token_ids": list(ids), "rank": i + 1, "returned_index": i,
            "selected_top1": i == 0, "score": -float(i + 1),
            "sum_logprobs": -float(i + 1) * len(ids),
            "completed": 3 in ids, "eos_emitted": 3 in ids,
            "stop_reason": "eos" if 3 in ids else "max_length",
            "generated_length": len(ids) - 1,
        } for i, ids in enumerate(sequences)]

    def run_evaluation(self, sequences, *, max_checked=None, rerank=True, all_hypotheses=True,
                       has_reference=False, masked=False, reorder=False, legacy_top1=None):
        hypotheses = self.hypotheses(sequences)
        if reorder:
            hypotheses = hypotheses[::-1]
        evidence = [{"candidates": hypotheses, "max_length": 16,
                     "stop_reason": "max_length", "mask_enabled": masked,
                     "forbidden_token_ids": [0, 1, 2, 57] if masked else []}]
        # The legacy output may contain a fabricated EOS. evaluate_mode must
        # only use raw evidence and preserve the actual incomplete sequence.
        legacy = legacy_top1 if legacy_top1 is not None else (sequences[0] if sequences else [2, 3])
        decoded = (torch.tensor([legacy]), [[list(seq) for seq in sequences]], evidence)
        cfg = evaluator.DecodeConfig(
            name="beam", enabled=True, decoding_method="beam", beam_size=max(1, len(sequences)),
            max_length=16, evaluate_beam_hypotheses=all_hypotheses,
            max_beams_to_check=max_checked, rerank_numerical_equiv=rerank,
        )
        good = {self.tokenizer.decode_infix(self.tokenizer.encode_infix(expression))
                for expression in (self.source, self.target)}

        def compare(source, candidate, cache, *, gravity_process):
            self.assertEqual(gravity_process, "3s2h")
            equivalent = candidate in good
            return {"status": "equivalent" if equivalent else "numerical_mismatch",
                    "equivalent": equivalent, "conclusive": True}

        with mock.patch.multiple(
            evaluator, BATCH_SIZE=1, REFERENCE_PROVIDED_OVERRIDE=has_reference,
            MASK_INVALID_TOKENS=masked, SAMPLING_SEED=73, NUMERIC_BACKEND="gravity",
            NUMERIC_EQUIV_POL_MODES=None, GRAVITY_REFERENCE_MODES=("first",),
            GRAVITY_GAUGE_SHIFT=False, NUMERIC_TOL_ABS=None, NUMERIC_TOL_REL=None,
        ), mock.patch.object(evaluator, "decode_with_model", return_value=decoded) as decoder, \
                mock.patch.object(evaluator, "numerical_comparison", side_effect=compare) as numeric, \
                redirect_stdout(io.StringIO()):
            rows, summary = evaluator.evaluate_mode(
                self.model, self.tokenizer, self.dataset, self.raw_rows, self.token_rows,
                {"3s2h": []}, ["3s2h"], cfg,
            )
        return rows[0], summary, decoder, numeric

    def test_all_retained_candidates_and_duplicates_survive_check_limit(self):
        good = self.raw(self.target)
        sequences = [self.raw("0"), good, good, [2, 4, 25, 3], good[:-1]]
        row, summary, decoder, numeric = self.run_evaluation(sequences, max_checked=3)
        evidence = row["_candidate_evidence"]
        records = evidence["candidates"]
        self.assertEqual([record["raw_token_ids"] for record in records], sequences)
        self.assertEqual([record["checked"] for record in records], [True, True, True, False, False])
        self.assertEqual([record["duplicate_of"] for record in records], [None, None, 1, None, None])
        self.assertEqual(evidence["counts"], {
            "generated": 5, "unique": 4, "checked": 3, "checked_unique": 2,
            "valid": 3, "complete": 3, "equivalent": 2, "shorter_equivalent": 2,
        })
        self.assertEqual(numeric.call_count, 3)
        self.assertIsNone(records[3]["numerical_check"])
        self.assertFalse(records[3]["decode_ok"])
        self.assertFalse(records[4]["completed"])
        self.assertEqual(row["candidate_sequences_generated"], 5)
        self.assertEqual(row["candidate_sequences_unique"], 4)
        self.assertEqual(row["candidate_sequences_checked"], 3)
        self.assertEqual(summary["avg_candidate_sequences_checked"], 3)
        self.assertTrue(decoder.call_args.kwargs["return_diagnostics"])

    def test_no_reference_metrics_are_null_in_candidates_rows_and_summary(self):
        row, summary, _, _ = self.run_evaluation([self.raw(self.source)])
        record = row["_candidate_evidence"]["candidates"][0]
        self.assertFalse(record["reference_provided"])
        for key in ("reference_check", "exact_token", "exact_string", "num_eq_simple"):
            self.assertIsNone(record[key], key)
        for key in ("top1_exact_token_match", "top1_exact_string_match", "top1_num_eq_simple",
                    "original_top1_num_eq_simple", "any_beam_num_eq_simple", "target_simple_token_count"):
            self.assertIsNone(row[key], key)
        for key in ("top1_exact_token_matches", "top1_exact_string_matches", "top1_num_eq_simple",
                    "original_top1_num_eq_simple", "any_beam_num_eq_simple"):
            self.assertIsNone(summary[key], key)
        self.assertEqual(row["target_simple"], "")
        self.assertFalse(summary["target_metrics_available"])
        self.assertEqual(summary["top1_num_eq_scrambled"], 1)
        self.assertEqual(summary["top1_copies_input"], 1)
        self.assertEqual(summary["top1_shorter_equivalent"], 0)

    def test_reranking_preserves_original_top1_and_uses_deterministic_duplicate_tie(self):
        good = self.raw(self.target)
        row, summary, _, _ = self.run_evaluation([self.raw("0"), good, good], reorder=True)
        self.assertEqual(row["original_top1_prediction_expr"], "0")
        self.assertEqual(row["original_top1_num_eq_scrambled"], 0)
        self.assertEqual(row["original_top1_shorter_equivalent"], 0)
        self.assertEqual(row["rerank_selected_candidate_index"], 1)
        self.assertEqual(row["selection_reason"], "numerically_equivalent_rerank")
        self.assertEqual(summary["original_top1_num_eq_scrambled"], 0)
        self.assertEqual(summary["top1_num_eq_scrambled"], 1)
        self.assertEqual(summary["reranked_top1_replacements"], 1)
        self.assertTrue(row["_candidate_evidence"]["any_successful_simplification"])

    def test_reranking_disabled_keeps_wrong_original_top1(self):
        row, summary, _, _ = self.run_evaluation([self.raw("0"), self.raw(self.target)], rerank=False)
        self.assertEqual(row["top1_prediction_expr"], "0")
        self.assertEqual(row["selection_reason"], "model_top1")
        self.assertEqual(summary["top1_num_eq_scrambled"], 0)
        self.assertEqual(summary["any_beam_num_eq_scrambled"], 1)

    def test_valid_display_fallback_is_still_numerically_wrong(self):
        row, summary, _, numeric = self.run_evaluation([[2, 4, 25, 3], self.raw("0")])
        self.assertEqual(row["selection_reason"], "valid_decode_fallback")
        self.assertEqual(row["original_top1_decode_ok"], 0)
        self.assertEqual(row["top1_prediction_expr"], "0")
        self.assertEqual(row["top1_num_eq_scrambled"], 0)
        self.assertEqual(row["top1_shorter_equivalent"], 0)
        self.assertEqual(summary["valid_fallback_replacements"], 1)
        self.assertEqual(numeric.call_count, 1)

    def test_legacy_fabricated_eos_cannot_complete_raw_prediction(self):
        incomplete = self.raw(self.target)[:-1]
        row, summary, _, _ = self.run_evaluation([incomplete], legacy_top1=self.raw(self.target))
        record = row["_candidate_evidence"]["candidates"][0]
        self.assertEqual(record["raw_token_ids"], incomplete)
        self.assertEqual(record["decoder"]["stop_reason"], "max_length")
        self.assertEqual(record["completion_status"], "missing_eos")
        self.assertEqual(row["top1_completed"], 0)
        self.assertEqual(row["top1_num_eq_scrambled"], 1)
        self.assertEqual(row["top1_shorter_equivalent"], 0)
        self.assertEqual(summary["top1_shorter_equivalent"], 0)

    def test_single_candidate_check_still_preserves_unchecked_beams(self):
        row, _, _, numeric = self.run_evaluation(
            [self.raw("0"), self.raw(self.target)], all_hypotheses=False,
        )
        self.assertEqual(row["candidate_sequences_generated"], 2)
        self.assertEqual(row["candidate_sequences_checked"], 1)
        self.assertEqual(row["top1_num_eq_scrambled"], 0)
        self.assertEqual(len(row["_candidate_evidence"]["candidates"]), 2)
        self.assertEqual(numeric.call_count, 1)

    def test_mask_setting_is_forwarded_with_unmapped_checkpoint_id(self):
        row, _, decoder, _ = self.run_evaluation([self.raw(self.target)], masked=True)
        self.assertEqual(decoder.call_args.kwargs["forbidden_token_ids"], [0, 1, 2, 57])
        self.assertTrue(row["_candidate_evidence"]["decoding"]["mask_enabled"])

    def test_independent_reference_is_evaluated_and_summary_metrics_available(self):
        row, summary, _, numeric = self.run_evaluation([self.raw(self.target)], has_reference=True)
        self.assertTrue(row["top1_exact_token_match"])
        self.assertTrue(row["top1_num_eq_simple"])
        self.assertTrue(summary["target_metrics_available"])
        self.assertEqual(summary["top1_exact_token_matches"], 1)
        self.assertEqual(numeric.call_count, 2)

    def test_detail_csv_omits_private_evidence_without_losing_public_fields(self):
        row, _, _, _ = self.run_evaluation([self.raw("0"), self.raw(self.target)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.csv"
            evaluator.write_detail_csv(path, [row])
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                restored = next(reader)
                self.assertNotIn("_candidate_evidence", reader.fieldnames)
            self.assertEqual(restored["candidate_sequences_generated"], "2")
            self.assertEqual(restored["original_top1_prediction_expr"], "0")
            self.assertEqual(restored["top1_exact_token_match"], "")
        # Evidence can be persisted as a strict JSON artifact independently.
        encoded = json.dumps(row["_candidate_evidence"], allow_nan=False)
        self.assertEqual(len(json.loads(encoded)["candidates"]), 2)

    def test_no_raw_hypotheses_is_an_explicit_decoder_error(self):
        with self.assertRaisesRegex(RuntimeError, "no raw candidate evidence"):
            self.run_evaluation([])


if __name__ == "__main__":
    unittest.main()
