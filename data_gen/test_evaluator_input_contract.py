"""Small pre-inference boundary and configuration regression tests."""
import csv
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from data_testing import evaluate_model as ev


class EvaluatorInputContractTests(unittest.TestCase):
    def test_smaller_checkpoint_vocabulary_rejects_valid_tokenizer_ids(self):
        model = types.SimpleNamespace(max_seq_len=100, vocab_size=20)
        dataset = types.SimpleNamespace(scrambled_sequences=[[2, 21, 3]], simple_sequences=[[2, 21, 3]])
        with self.assertRaisesRegex(ValueError, "outside the checkpoint vocabulary"):
            ev.validate_model_sequence_capacity(model, dataset)
        self.assertEqual(dataset.scrambled_sequences, [[2, 21, 3]])

    def test_explicit_process_conflicts_with_embedded_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["simple", "scrambled", "process"])
                writer.writeheader()
                writer.writerow(dict(simple="0", scrambled="0", process="4s1h"))
            with mock.patch.multiple(ev, NUMERIC_BACKEND="gravity", DATA_SOURCE="csv",
                                     GRAVITY_PROCESS="3s2h", GRAVITY_METADATA_CSV_PATH=None,
                                     EXISTING_RAW_CSV_PATH=path):
                with self.assertRaisesRegex(ValueError, "process conflict"):
                    ev.resolve_gravity_processes([dict(simple="0", scrambled="0")])

    def test_sampling_and_numeric_seeds_do_not_request_generation(self):
        args = ev.parse_args(["--sampling-seed", "4", "--numeric-seed", "5", "--numeric-samples", "2"])
        with mock.patch.multiple(ev, DATA_SOURCE="csv", SAMPLING_SEED=0,
                                 NUMERIC_EQUIV_SEED=151, NUMERIC_EQUIV_SAMPLES=3,
                                 REFERENCE_PROVIDED_OVERRIDE=None, INPUT_PROVENANCE_PATH=None):
            ev.apply_cli_config(args)
            self.assertEqual(ev.DATA_SOURCE, "csv")
            self.assertEqual((ev.SAMPLING_SEED, ev.NUMERIC_EQUIV_SEED, ev.NUMERIC_EQUIV_SAMPLES), (4, 5, 2))

    def test_rerank_flag_preserves_existing_beam_size(self):
        cfg = ev.DecodeConfig(name="beam", enabled=True, decoding_method="beam", beam_size=20)
        with mock.patch.multiple(ev, DECODE_RUNS=[cfg], REFERENCE_PROVIDED_OVERRIDE=None,
                                 INPUT_PROVENANCE_PATH=None):
            ev.apply_cli_config(ev.parse_args(["--no-rerank-numerical"]))
            self.assertEqual(ev.DECODE_RUNS[0].beam_size, 20)
            self.assertFalse(ev.DECODE_RUNS[0].rerank_numerical_equiv)


if __name__ == "__main__":
    unittest.main()
