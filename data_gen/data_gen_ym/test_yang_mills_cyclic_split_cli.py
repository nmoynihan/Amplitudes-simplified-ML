"""Held-out target splitting through the canonical/nonzero cyclic CLI."""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import generate
from . import generate_clean_4pt as clean
from . import split_cyclic_train_test as splitter
from . import yang_mills_cyclic_generation as cyclic


def _pool(n_particles: int) -> list[tuple[str, str]]:
    trace = "Tr(" + " · ".join(f"F_{i}" for i in range(1, n_particles + 1)) + ")"
    return [
        (f"{coefficient}*{trace}", f"{coefficient}*{trace}{suffix}")
        for coefficient in (1, 2, 3, 4)
        for suffix in ((" + 0", " * 1") if coefficient < 3 else (" + 0",))
    ]


def _args(directory: Path, n_particles: int = 4) -> list[str]:
    return [
        str(n_particles), "--samples", "6", "--jobs", "1", "--no-progress",
        "--zero-checks", "1", "--validation-checks", "1",
        "--raw-out", str(directory / "pool.csv"),
        "--tok-out", str(directory / "pool_tok.csv"),
        "--log-out", str(directory / "pool.log"),
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CyclicSplitCliTests(unittest.TestCase):
    def test_both_modes_split_clean_pool_and_keep_targets_and_inputs_disjoint(self) -> None:
        for n_particles, mode in ((4, "oneshot"), (5, "step")):
            with self.subTest(n_particles=n_particles, mode=mode), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), mock.patch.object(
                    clean, "build_dataset_batched", return_value=_pool(n_particles)
                ) as builder:
                    result = cyclic.main(_args(directory, n_particles) + [
                        "--test-size", "1", "--dataset-kind", mode,
                    ])
                self.assertEqual(result, 0)
                self.assertTrue(builder.call_args.kwargs["cyclic_order"])
                self.assertEqual(len(_read_csv(directory / "pool.csv")), 6)
                self.assertEqual(len(_read_csv(directory / "pool_tok.csv")), 6)
                generation = json.loads((directory / "pool.log").read_text(encoding="utf-8"))
                self.assertEqual(generation["stats"]["accepted"], 6)
                manifest_path = directory / "cyclic_train_test" / (
                    f"ym_cyclic_{n_particles}pt_split_seed20260401.json"
                )
                report = json.loads(manifest_path.read_text(encoding="utf-8"))
                train = _read_csv(Path(report["outputs"]["train_tokenized"]["path"]))
                test = _read_csv(Path(report["outputs"]["test_tokenized"]["path"]))
                self.assertEqual(len(train), 5)
                self.assertEqual(len(test), 1)
                for column in ("simple", "scrambled"):
                    train_values = {tuple(json.loads(row[column])) for row in train}
                    test_values = {tuple(json.loads(row[column])) for row in test}
                    self.assertFalse(train_values & test_values)
                tokenizer = clean.ScatteringAmplitudeTokenizer(max_particles=8)
                for pair_index in (0, 2):
                    compact = clean.canonicalize_simple_expression(
                        _pool(n_particles)[pair_index][0], n_particles=n_particles,
                    ).expression
                    normalized = clean.parenthesize_for_semantic_tokenization(compact)
                    expected = tokenizer.encode_infix(normalized)
                    self.assertEqual(sum(json.loads(row["simple"]) == expected for row in train), 2)
                for raw_kind, token_kind in (("train_raw", "train_tokenized"), ("test_raw", "test_tokenized")):
                    raw_rows = _read_csv(Path(report["outputs"][raw_kind]["path"]))
                    token_rows = _read_csv(Path(report["outputs"][token_kind]["path"]))
                    for raw, tokenized in zip(raw_rows, token_rows):
                        for column in ("simple", "scrambled"):
                            self.assertEqual(json.loads(tokenized[column]), tokenizer.encode_infix(raw[column]))
                self.assertEqual(report["verification"]["train_test_target_overlap"], 0)
                self.assertEqual(report["verification"]["train_test_input_overlap"], 0)
                self.assertIn("train/test target overlap: 0", stdout.getvalue())
                self.assertIn("train/test input overlap: 0", stdout.getvalue())
                self.assertIn(str(manifest_path), stdout.getvalue())

    def test_split_options_are_forwarded_to_the_splitter(self) -> None:
        report = {
            "outputs": {
                "train_raw": {"rows": 5, "path": "train.csv"},
                "test_raw": {"rows": 1, "path": "test.csv"},
            },
            "verification": {"train_test_target_overlap": 0, "train_test_input_overlap": 0},
            "manifest_path": "split.json",
        }
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output_dir = directory / "heldout"
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                clean, "build_dataset_batched", return_value=_pool(4)
            ), mock.patch.object(splitter, "split_cyclic_dataset", return_value=report) as split:
                result = cyclic.main(_args(directory) + [
                    "--test-size", "1", "--split-seed", "19", "--split-output-dir",
                    str(output_dir), "--split-overwrite", "--tokenizer-max-particles", "5",
                ])
            self.assertEqual(result, 0)
            split.assert_called_once_with(
                directory / "pool.csv", directory / "pool_tok.csv",
                n_particles=4, test_size=1, seed=19, output_dir=output_dir.resolve(),
                tokenizer_max_particles=5, overwrite=True,
            )

    def test_no_split_still_generates_clean_exact_sized_raw_output(self) -> None:
        for extra in ([], ["--test-size", "0"]):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    clean, "build_dataset_batched", return_value=_pool(4)
                ), mock.patch.object(splitter, "split_cyclic_dataset") as split:
                    result = cyclic.main(_args(directory) + ["--no-tokenise"] + extra)
                self.assertEqual(result, 0)
                split.assert_not_called()
                self.assertEqual(len(_read_csv(directory / "pool.csv")), 6)
                self.assertFalse((directory / "pool_tok.csv").exists())

    def test_invalid_split_settings_fail_before_generation(self) -> None:
        cases = (
            (["--test-size", "-1"], "--test-size must be non-negative"),
            (["--test-size", "6"], "--test-size must be smaller than --samples"),
            (["--test-size", "7"], "--test-size must be smaller than --samples"),
            (["--test-size", "1", "--no-tokenise"], "requires tokenisation"),
            (["--test-size", "1", "--tokenizer-max-particles", "3"], "must be at least"),
        )
        for extra, error in cases:
            stderr = io.StringIO()
            with self.subTest(extra=extra), contextlib.redirect_stderr(stderr), mock.patch.object(
                clean, "generate_to_files"
            ) as writer:
                with self.assertRaises(SystemExit) as raised:
                    cyclic.main(["4", "--samples", "6"] + extra)
                self.assertEqual(raised.exception.code, 2)
                writer.assert_not_called()
                self.assertIn(error, stderr.getvalue())

    def test_existing_split_output_fails_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output_dir = directory / "heldout"
            output_dir.mkdir()
            paths = splitter.output_paths(output_dir, n_particles=4, source_rows=6, test_size=1, seed=19)
            paths.train_raw.write_text("existing data\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(
                clean, "generate_to_files"
            ) as writer:
                with self.assertRaises(SystemExit) as raised:
                    cyclic.main(_args(directory) + [
                        "--test-size", "1", "--split-seed", "19", "--split-output-dir", str(output_dir),
                    ])
            self.assertEqual(raised.exception.code, 2)
            writer.assert_not_called()
            self.assertEqual(paths.train_raw.read_text(encoding="utf-8"), "existing data\n")

    def test_unsupported_particle_count_fails_before_generation(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(
            clean, "generate_to_files"
        ) as writer:
            with self.assertRaises(SystemExit) as raised:
                cyclic.main(["3", "--samples", "6", "--test-size", "1"])
            self.assertEqual(raised.exception.code, 2)
            writer.assert_not_called()

    def test_existing_generator_does_not_accept_cyclic_split_options(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(
            generate, "build_dataset_batched"
        ) as builder:
            with self.assertRaises(SystemExit) as raised:
                generate.main(["4", "--samples", "6", "--test-size", "1"])
            self.assertEqual(raised.exception.code, 2)
            builder.assert_not_called()

    def test_split_failure_is_reported_as_a_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr), mock.patch.object(
                clean, "build_dataset_batched", return_value=_pool(4)
            ), mock.patch.object(
                splitter, "split_cyclic_dataset", side_effect=ValueError("not enough distinct targets")
            ):
                result = cyclic.main(_args(directory) + ["--test-size", "1"])
            self.assertEqual(result, 1)
            self.assertIn("not enough distinct targets", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
