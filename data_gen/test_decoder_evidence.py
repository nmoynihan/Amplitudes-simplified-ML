"""Bounded decoder evidence tests; no checkpoint, datasets or training required."""

import math
import types
import unittest

import torch

from transformer.transformer_functions import (
    BeamHypotheses,
    TransformerRegressor,
    decode_with_model,
)


class ScriptedModel:
    """Exercise the real search loops with tiny deterministic decoder logits."""

    generate = TransformerRegressor.generate
    generate_beam = TransformerRegressor.generate_beam
    create_padding_mask = TransformerRegressor.create_padding_mask
    create_causal_mask = TransformerRegressor.create_causal_mask

    def __init__(self, policy, vocab_size=8):
        self.device = "cpu"
        self.vocab_size = vocab_size
        self.embedding_dim = 1
        self.pad_token_id = 0
        self.policy = policy
        self.src_embedding = self.tgt_embedding = lambda ids: ids.float().unsqueeze(-1)
        self.src_pos_encoding = self.tgt_pos_encoding = lambda x: x
        self.dropout = self.output_projection = lambda x: x
        self.transformer = types.SimpleNamespace(
            encoder=lambda x, **kwargs: x, decoder=self.decode,
        )

    def eval(self):
        return self

    def decode(self, target, memory, **kwargs):
        result = torch.full((target.size(0), target.size(1), self.vocab_size), -50.0)
        for row in range(target.size(0)):
            logits = self.policy(target[row, :, 0].long().tolist(), int(memory[row, 0, 0]))
            for token, score in logits.items():
                result[row, -1, token] = score
        return result


class DecoderEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.tensor([[2, 4, 3]])

    def test_greedy_evidence_does_not_change_default_outputs(self):
        model = ScriptedModel(lambda prefix, _: {4 if len(prefix) == 1 else 3: 10})
        default = decode_with_model(model, self.source, 8)
        self.assertEqual(len(default), 2)
        output, beams, evidence = decode_with_model(model, self.source, 8, return_diagnostics=True)
        self.assertTrue(torch.equal(default[0], output))
        self.assertIsNone(beams)
        candidate = evidence[0]["candidates"][0]
        self.assertEqual(candidate["raw_token_ids"], [2, 4, 3])
        self.assertTrue(candidate["completed"])
        self.assertEqual(candidate["generated_length"], 2)
        self.assertEqual(candidate["stop_reason"], "eos")
        self.assertTrue(math.isfinite(candidate["score"]))

    def test_greedy_preserves_trailing_padding_after_real_eos(self):
        def policy(prefix, source_id):
            return {3 if len(prefix) >= (1 if source_id == 4 else 3) else 5: 10}
        model = ScriptedModel(policy)
        source = torch.tensor([[4, 3], [5, 3]])
        _, _, evidence = decode_with_model(model, source, 8, return_diagnostics=True)
        self.assertEqual(evidence[0]["candidates"][0]["raw_token_ids"], [2, 3, 0, 0])
        self.assertEqual(evidence[0]["candidates"][0]["generated_length"], 1)
        self.assertEqual(evidence[1]["candidates"][0]["raw_token_ids"], [2, 5, 5, 3])

    def test_beam_limit_preserves_actual_last_token_without_fabricating_eos(self):
        model = ScriptedModel(lambda prefix, _: {4: 10, 5: 9, 3: -100})
        default = decode_with_model(model, self.source, 4, decoding_method="beam", beam_size=2)
        output, beams, evidence = decode_with_model(
            model, self.source, 4, decoding_method="beam", beam_size=2, return_diagnostics=True,
        )
        self.assertTrue(torch.equal(default[0], output))
        self.assertEqual(default[1], beams)
        candidates = evidence[0]["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual([c["rank"] for c in candidates], [1, 2])
        self.assertTrue(candidates[0]["selected_top1"])
        for candidate in candidates:
            self.assertEqual(len(candidate["raw_token_ids"]), 4)
            self.assertEqual(candidate["generated_length"], 3)
            self.assertNotIn(3, candidate["raw_token_ids"])
            self.assertFalse(candidate["eos_emitted"])
            self.assertEqual(candidate["stop_reason"], "max_length")
            # Legacy compatibility output is explicitly NOT the raw evidence.
            self.assertEqual(beams[0][candidate["returned_index"]][-1], 3)
            self.assertAlmostEqual(candidate["score"], candidate["sum_logprobs"] / 3)

    def test_real_eos_and_duplicate_nucleus_hypotheses_are_preserved(self):
        model = ScriptedModel(lambda prefix, _: {4 if len(prefix) == 1 else 3: 20})
        _, _, evidence = decode_with_model(
            model, self.source, 8, decoding_method="nucleus", beam_size=3,
            p_nucleus=0.01, return_diagnostics=True,
        )
        candidates = evidence[0]["candidates"]
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            self.assertEqual(candidate["raw_token_ids"], [2, 4, 3])
            self.assertTrue(candidate["eos_emitted"])
            self.assertEqual(candidate["stop_reason"], "eos")
        self.assertTrue(evidence[0]["nucleus_sampling_renormalized"])
        self.assertTrue(evidence[0]["nucleus_score_before_top_p_renormalization"])

    def test_mask_is_optional_and_consistent_in_all_modes(self):
        model = ScriptedModel(lambda prefix, _: {0: 40, 1: 30, 2: 20, 7: 15, 4: 10, 3: 5})
        for mode in ("greedy", "beam", "nucleus"):
            with self.subTest(mode=mode):
                kwargs = dict(decoding_method=mode, beam_size=2, p_nucleus=0.01, return_diagnostics=True)
                _, _, baseline = decode_with_model(model, self.source, 4, **kwargs)
                self.assertFalse(baseline[0]["mask_enabled"])
                self.assertIn(0, baseline[0]["candidates"][0]["raw_token_ids"][1:])
                _, _, masked = decode_with_model(
                    model, self.source, 4, forbidden_token_ids=[0, 1, 2, 7], **kwargs,
                )
                self.assertTrue(masked[0]["mask_enabled"])
                self.assertEqual(masked[0]["forbidden_token_ids"], [0, 1, 2, 7])
                for candidate in masked[0]["candidates"]:
                    self.assertFalse(set(candidate["raw_token_ids"][1:]) & {0, 1, 2, 7})

    def test_eos_on_last_allowed_step_is_completed(self):
        model = ScriptedModel(lambda prefix, _: {4 if len(prefix) < 3 else 3: 10})
        for mode in ("greedy", "beam", "nucleus"):
            with self.subTest(mode=mode):
                _, _, evidence = decode_with_model(
                    model, self.source, 4, decoding_method=mode, beam_size=1,
                    p_nucleus=0.01, return_diagnostics=True,
                )
                candidate = evidence[0]["candidates"][0]
                self.assertEqual(candidate["raw_token_ids"], [2, 4, 4, 3])
                self.assertTrue(candidate["completed"])

    def test_invalid_masks_and_limits_raise_explicitly(self):
        model = ScriptedModel(lambda prefix, _: {3: 10})
        for mode in ("greedy", "beam", "nucleus"):
            for bad_mask in ([3], [8], [-1]):
                with self.subTest(mode=mode, mask=bad_mask), self.assertRaises(ValueError):
                    decode_with_model(model, self.source, 4, decoding_method=mode, forbidden_token_ids=bad_mask)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                decode_with_model(model, self.source, 1, decoding_method=mode)

    def test_mask_can_leave_only_eos_without_creating_illegal_fillers(self):
        model = ScriptedModel(lambda prefix, _: {0: 20, 3: 10})
        for mode in ("beam", "nucleus"):
            _, _, evidence = decode_with_model(
                model, self.source, 5, decoding_method=mode, beam_size=3,
                forbidden_token_ids=[0, 1, 2, 4, 5, 6, 7], return_diagnostics=True,
            )
            for candidate in evidence[0]["candidates"]:
                self.assertEqual(candidate["raw_token_ids"], [2, 3])
                self.assertTrue(candidate["completed"])

    def test_beam_metadata_remains_aligned_when_lower_scores_are_pruned(self):
        hypotheses = BeamHypotheses(2, 8, 1.0, True)
        for token, score in ((4, -4), (5, -2), (6, -1)):
            hypotheses.add(torch.tensor([2, token]), score,
                           raw_token_ids=[2, token, 3], eos_emitted=True, stop_reason="eos")
        self.assertEqual(hypotheses.terminal_proposals, 3)
        self.assertEqual(len(hypotheses.hyp), 2)
        for (_, hypothesis), metadata in zip(hypotheses.hyp, hypotheses.hyp_metadata):
            self.assertEqual(hypothesis.tolist() + [3], metadata["raw_token_ids"])


if __name__ == "__main__":
    unittest.main()
