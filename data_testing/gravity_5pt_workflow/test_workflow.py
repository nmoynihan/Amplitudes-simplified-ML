"""Contract and failure-path checks; no checkpoint or experiment data required."""
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import csv
import gzip
import io
import json
import tempfile
import unittest
from unittest import mock

import sympy as sp
import torch

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_testing import evaluate_model as ev
from . import __main__ as workflow
from . import derive_generic_groups, grouped_model, serialize_groups, verify_final
from .common import read_source
from .symbolic_utils import symbolic


class SourceContractTests(unittest.TestCase):
    def test_single_row_source_and_gzip(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, opener in ((".csv", open), (".csv.gz", gzip.open)):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / ("source" + suffix)
                    with opener(path, "wt", encoding="utf-8", newline="") as handle:
                        csv.writer(handle).writerow(["amplitude", " (p_1 · p_2) "])
                    self.assertEqual(read_source(path), "(p_1 · p_2)")

    def test_multirow_and_header_sources_are_rejected(self):
        invalid_rows = (
            [["first", "(p_1 · p_2)"], ["second", "(p_1 · p_3)"]],
            [["id", "expression"]],
            [["id", "expression"], ["amplitude", "(p_1 · p_2)"]],
            [["id", "scrambled"]],
            [["id", "simple"]],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            for rows in invalid_rows:
                with self.subTest(rows=rows):
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        csv.writer(handle).writerows(rows)
                    with self.assertRaisesRegex(ValueError, "headerless"):
                        read_source(path)

    def test_nonempty_output_directory_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("amplitude,(p_1 · p_2)\n", encoding="utf-8")
            output = root / "existing"
            output.mkdir()
            sentinel = output / "do-not-overwrite.txt"
            sentinel.write_bytes(b"original evidence\x00\xff")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                workflow.run(source, output, prepare_only=True)
            self.assertEqual(sentinel.read_bytes(), b"original evidence\x00\xff")
            self.assertEqual(list(output.iterdir()), [sentinel])


class SymbolicContractTests(unittest.TestCase):
    def test_field_strength_and_trace_conventions(self):
        identities = (
            ("(p_1 · F_4 · p_5)",
             "(p_1 · p_4)*(e_4 · p_5) - (p_1 · e_4)*(p_4 · p_5)"),
            ("Tr(F_4 · F_5)",
             "2*((e_4 · p_5)*(e_5 · p_4) - (p_4 · p_5)*(e_4 · e_5))"),
        )
        for compact, scalar in identities:
            with self.subTest(compact=compact):
                self.assertEqual(sp.expand(symbolic(compact) - symbolic(scalar)), 0)

    def test_malformed_expressions_are_rejected(self):
        for expression in ("(p_1 · p_2) +", "(p_1 · p_2) trailing", "Tr(F_4 · )"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    symbolic(expression)


class InferenceFailureTests(unittest.TestCase):
    def test_bad_or_unterminated_prediction_never_reaches_reconstruction(self):
        """A valid first core must not hide failure of the second core."""
        tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
        expression = "(p_1 · p_2)"
        good_ids = tokenizer.encode_infix(expression)
        wrong_ids = tokenizer.encode_infix("(p_1 · p_3)")
        cases = (
            ("inequivalent", [2, *wrong_ids, 3], False, True),
            ("no_eos", [2, *good_ids], True, False),
            ("empty", [2, 3], False, True),
        )
        for name, bad_sequence, exact, eos in cases:
            with self.subTest(prediction=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, checkpoint, output = root / "source.csv", root / "fake.pt", root / "run"
                source.write_text("amplitude,(p_1 · p_2)\n", encoding="utf-8")
                checkpoint.write_bytes(b"placeholder; torch.load is mocked")
                choices = [
                    {"name": f"group_{index}", "group": index, "input": expression,
                     "input_tokens": len(good_ids), "ids": good_ids, "weight": "1"}
                    for index in (1, 2)
                ]
                model = mock.MagicMock()
                model.vocab_size = 10000
                predictions = [
                    ([torch.tensor([2, *good_ids, 3])], None),
                    ([torch.tensor(bad_sequence)], None),
                ]
                with ExitStack() as stack:
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    stack.enter_context(mock.patch.object(derive_generic_groups, "main"))
                    stack.enter_context(mock.patch.object(grouped_model, "main"))
                    stack.enter_context(mock.patch.object(serialize_groups, "prepare_inputs", return_value=choices))
                    stack.enter_context(mock.patch.object(torch, "set_num_threads"))
                    loader = stack.enter_context(mock.patch.object(
                        torch, "load", return_value={"model_args": {}, "model_state_dict": {}}))
                    stack.enter_context(mock.patch.object(ev, "TransformerRegressor", return_value=model))
                    stack.enter_context(mock.patch.object(ev, "positional_encoding_capacity", return_value=5000))
                    decoder = stack.enter_context(mock.patch.object(ev, "decode_with_model", side_effect=predictions))
                    numeric = stack.enter_context(mock.patch.object(serialize_groups, "numeric_check"))
                    final = stack.enter_context(mock.patch.object(verify_final, "main"))
                    with self.assertRaisesRegex(ValueError, "exact EOS-terminated"):
                        workflow.run(source, output, checkpoint=checkpoint)
                    loader.assert_called_once_with(checkpoint.resolve(), map_location="cpu", weights_only=True)
                    self.assertEqual(decoder.call_count, 2)
                    numeric.assert_not_called()
                    final.assert_not_called()

                records = json.loads((output / "serialization_results.json").read_text())
                self.assertEqual(len(records), 2)
                self.assertTrue(records[0]["exact_equal"])
                self.assertEqual(records[1]["exact_equal"], exact)
                self.assertEqual(records[1]["eos"], eos)
                manifest = json.loads((output / "run_manifest.json").read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["neural_calls"], 2)
                for filename in (
                    "selected_groups.json", "successful_model_inputs.csv", "two_call_summary.json",
                    "two_call_reconstruction.txt", "final_verification.json",
                    "model_plus_cleanup_67.txt", "model_plus_helicity_cleanup_53.txt",
                    "model_plus_cleanup_general.txt",
                ):
                    self.assertFalse((output / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
