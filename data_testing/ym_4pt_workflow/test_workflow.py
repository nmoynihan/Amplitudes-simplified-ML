"""Scientific regressions and failed-decode provenance, without a checkpoint."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import sympy as sp

from .algebra import (DEFAULT_REFERENCE, DEFAULT_SEED, REPO_ROOT, parse,
                      read_expression, rotate, symbolic, ward)
from .inference import completed_tokens
from .prepare import compact_targets, prepare
from .verification import verify_prediction
from .workflow import execute, output_directory


PACKAGE = Path(__file__).resolve().parent
FIXTURE = PACKAGE / "inputs/gluon4feyn1234_model_ready.csv"


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.manifest = prepare(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_recreates_exact_audited_bytes(self):
        self.assertEqual((self.directory / FIXTURE.name).read_bytes(), FIXTURE.read_bytes())
        self.assertEqual(self.manifest["input_tokens"], 1247)
        self.assertFalse(self.manifest["model_called"])
        self.assertEqual([v["tokens"] for v in self.manifest["completion"]["variants"]],
                         [593, 519, 424])

    def test_seed_is_not_the_completed_amplitude(self):
        seed = read_expression(DEFAULT_SEED)
        original = parse(seed)
        self.assertTrue(any(sp.cancel(ward(original, leg)) != 0 for leg in range(1, 5)))
        completed = symbolic(f"(({seed})+({rotate(seed)}))/2")
        self.assertEqual(sp.cancel(completed - symbolic(read_expression(DEFAULT_REFERENCE))), 0)
        self.assertNotEqual(sp.cancel(symbolic(seed) - completed), 0)

    def test_wrong_basis_coefficient_is_rejected(self):
        _, solution = compact_targets()
        altered = dict(solution, coefficients=["0", *solution["coefficients"][1:]])
        bad = self.directory / "bad_basis.json"
        bad.write_text(json.dumps([altered]))
        with self.assertRaisesRegex(ValueError, "Pole representative"):
            compact_targets(bad)

    def test_wrong_normalization_prediction_is_rejected(self):
        historical = json.loads((PACKAGE / "reference/one_shot_results.json").read_text())
        with self.assertRaisesRegex(ValueError, "Model prediction"):
            verify_prediction(historical["source"], f"2*({historical['prediction']})",
                              read_expression(DEFAULT_REFERENCE))


class AlgebraTests(unittest.TestCase):
    def test_independent_field_strength_expansion(self):
        self.assertEqual(sp.expand(parse("p_2·F_1·p_3") - parse(
            "(p_2·p_1)*(e_1·p_3)-(p_2·e_1)*(p_1·p_3)")), 0)
        self.assertEqual(sp.cancel(ward(parse("p_2·F_1·F_4·p_3"), 1)), 0)

    def test_rejects_unknown_names_and_placeholder_collisions(self):
        for text in ("BLOCK0 + p_1·p_2", "__import__('os')", "p_5·p_1",
                     "F_1·p_2", "p_1·p_2·p_3", "1/0", "p1p2 + p_1·p_2"):
            with self.subTest(expression=text), self.assertRaises((ValueError, SyntaxError)):
                parse(text)


class RuntimeTests(unittest.TestCase):
    def test_truncated_or_invalid_decoder_output_is_rejected(self):
        for raw in ([], [2, 4], [2, 3], [2, 1, 3], [4, 3], [2, 4, 3, 7]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                completed_tokens(raw)
        self.assertEqual(completed_tokens([2, 4, 5, 3, 0]), [4, 5])

    def test_nonempty_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sentinel = directory / "existing.txt"
            sentinel.write_text("keep this")
            with self.assertRaises(ValueError):
                output_directory(directory)
            self.assertEqual(sentinel.read_text(), "keep this")

    def test_failed_decode_records_call_and_emitted_tokens(self):
        def fake_prepare(directory, **kwargs):
            shutil.copyfile(FIXTURE, directory / FIXTURE.name)
            return {}

        def failed_prediction(source, checkpoint, *, record_attempt, **kwargs):
            record_attempt({"model_calls": 1, "raw_token_ids": [2, 4], "status": "decoded"})
            raise ValueError("Greedy prediction is missing BOS or emitted EOS")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.touch()
            output = root / "result"
            with patch("data_testing.ym_4pt_workflow.prepare.prepare", side_effect=fake_prepare), \
                 patch("data_testing.ym_4pt_workflow.audit.run_all", return_value={}), \
                 patch("data_testing.ym_4pt_workflow.inference.predict", side_effect=failed_prediction):
                with self.assertRaisesRegex(ValueError, "emitted EOS"):
                    execute("run", output, checkpoint=checkpoint)
            status = json.loads((output / "workflow.json").read_text())
            self.assertEqual((status["status"], status["model_calls"]), ("failed", 1))
            self.assertEqual(json.loads((output / "inference_attempt.json").read_text())["raw_token_ids"], [2, 4])
            self.assertFalse((output / "results.json").exists())
            self.assertFalse((output / "gluon4feyn1234_completed_simplified.csv").exists())

    def test_preparation_works_outside_repo_without_torch(self):
        script = """
import importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1])
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('Preparation must not import torch')
sys.meta_path.insert(0, NoTorch())
from data_testing.ym_4pt_workflow.prepare import prepare
result = prepare(pathlib.Path('prepared'))
if result['input_tokens'] != 1247:
    raise RuntimeError('Incorrect source token count')
"""
        with tempfile.TemporaryDirectory() as temporary:
            process = subprocess.run([sys.executable, "-c", script, str(REPO_ROOT)],
                                     cwd=temporary, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
