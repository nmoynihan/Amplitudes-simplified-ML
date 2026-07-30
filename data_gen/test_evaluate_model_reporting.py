"""Regression tests for non-empty model-prediction reporting."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from data_testing import evaluate_model as evaluator
from data_gen.Tokenizer import ScatteringAmplitudeTokenizer


def candidate(
    *,
    index: int,
    tokens: list[int],
    decode_ok: bool,
    expr: str = "",
    num_eq_scrambled: bool = False,
) -> dict[str, object]:
    return {
        "index": index,
        "tokens": tokens,
        "decode_ok": decode_ok,
        "expr": expr,
        "decode_error": "" if decode_ok else "ValueError: malformed",
        "exact_token": False,
        "exact_string": False,
        "num_eq_simple": num_eq_scrambled,
        "num_eq_scrambled": num_eq_scrambled,
    }


class PredictionReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8,
            max_sequence_length=None,
        )

    def test_invalid_top1_falls_back_to_a_valid_generated_candidate(self) -> None:
        invalid_tokens = [self.tokenizer.vocab["*"], self.tokenizer.vocab["p_1"]]
        valid_tokens = self.tokenizer.encode_infix("p_1 · p_2")
        candidates = [
            candidate(index=0, tokens=invalid_tokens, decode_ok=False),
            candidate(
                index=1,
                tokens=valid_tokens,
                decode_ok=True,
                expr="p_1·p_2",
            ),
        ]

        selected, reason = evaluator.select_candidate_for_reporting(
            candidates,
            rerank_numerical_equiv=True,
        )

        self.assertEqual(selected["index"], 1)
        self.assertEqual(reason, "valid_decode_fallback")
        self.assertEqual(
            evaluator.prediction_text_for_display(self.tokenizer, selected),
            "p_1·p_2",
        )

    def test_numerically_equivalent_candidate_takes_priority(self) -> None:
        candidates = [
            candidate(
                index=0,
                tokens=self.tokenizer.encode_infix("p_1 · p_2"),
                decode_ok=True,
                expr="p_1·p_2",
            ),
            candidate(
                index=1,
                tokens=self.tokenizer.encode_infix("p_1 · p_3"),
                decode_ok=True,
                expr="p_1·p_3",
                num_eq_scrambled=True,
            ),
        ]

        selected, reason = evaluator.select_candidate_for_reporting(
            candidates,
            rerank_numerical_equiv=True,
        )

        self.assertEqual(selected["index"], 1)
        self.assertEqual(reason, "numerically_equivalent_rerank")

    def test_malformed_tokens_are_displayed_when_every_candidate_is_invalid(self) -> None:
        invalid_tokens = [self.tokenizer.vocab["*"], self.tokenizer.vocab["p_1"]]
        candidates = [
            candidate(index=0, tokens=invalid_tokens, decode_ok=False),
            candidate(index=1, tokens=[self.tokenizer.vocab["/"]], decode_ok=False),
        ]

        selected, reason = evaluator.select_candidate_for_reporting(
            candidates,
            rerank_numerical_equiv=True,
        )
        display = evaluator.prediction_text_for_display(
            self.tokenizer,
            selected,
        )

        self.assertEqual(selected["index"], 0)
        self.assertEqual(reason, "model_top1")
        self.assertEqual(display, "[malformed prefix tokens] * p_1")

    def test_empty_prediction_has_an_explicit_display_value(self) -> None:
        display = evaluator.prediction_text_for_display(
            self.tokenizer,
            candidate(index=0, tokens=[], decode_ok=False),
        )

        self.assertEqual(display, "[empty token prediction]")

    def test_terminal_and_human_csv_use_the_nonempty_display_prediction(self) -> None:
        display = "[malformed prefix tokens] * p_1"
        row = {
            "row_id": 0,
            "mode": "nucleus",
            "target_simple": "p_1·p_2",
            "input_scrambled": "p_1·p_2",
            "top1_prediction_expr": "",
            "top1_prediction_display": display,
            "top1_prediction_token_count": 2,
            "top1_decode_ok": 0,
            "top1_decode_error": "ValueError: Malformed prefix: ran out of tokens.",
            "top1_exact_token_match": 0,
            "top1_exact_string_match": 0,
            "top1_num_eq_simple": 0,
            "top1_num_eq_scrambled": 0,
            "selection_reason": "model_top1",
            "selection_replaced_top1": 0,
            "rerank_replaced_top1": 0,
            "original_top1_prediction_expr": "",
            "original_top1_prediction_display": display,
            "any_beam_exact_token_match": 0,
            "any_beam_exact_string_match": 0,
            "any_beam_num_eq_simple": 0,
            "any_beam_num_eq_scrambled": 0,
            "candidate_sequences_checked": 2,
            "candidate_valid_decode_count": 0,
            "target_scrambled_token_count": 3,
        }

        terminal = io.StringIO()
        with redirect_stdout(terminal):
            evaluator.print_examples([row], 1)
        self.assertIn(f"top1 pred     : {display}", terminal.getvalue())
        self.assertIn("decode error  : ValueError:", terminal.getvalue())

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "human.csv"
            evaluator.write_human_csv(output_path, [row])
            with output_path.open(newline="", encoding="utf-8") as handle:
                written = next(csv.DictReader(handle))

        self.assertEqual(written["top_pred"], display)
        self.assertEqual(written["decode_ok"], "no")


if __name__ == "__main__":
    unittest.main()
