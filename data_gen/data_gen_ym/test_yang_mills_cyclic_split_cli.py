"""Held-out target splitting through the cyclic generator CLI."""

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
        "--raw-out", str(directory / "pool.csv"),
        "--tok-out", str(directory / "pool_tok.csv"),
        "--log-out", str(directory / "pool.log"),
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CyclicSplitCliTests(unittest.TestCase):
    def test_both_modes_split_generated_pool_and_keep_targets_disjoint(self) -> None:
        for n_particles, mode in ((4, "oneshot"), (5, "step")):
            with self.subTest(n_particles=n_particles, mode=mode), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), mock.patch.object(
                    generate, "build_dataset_batched", return_value=_pool(n_particles)
                ) as builder:
                    cyclic.main(_args(directory, n_particles) + [
                        "--test-size", "1", "--dataset-kind", mode,
                    ])
                self.assertEqual(builder.call_args.kwargs["dataset_kind"], mode)
                self.assertTrue(builder.call_args.kwargs["cyclic_order"])
                self.assertEqual(len(_read_csv(directory / "pool.csv")), 6)
                self.assertEqual(len(_read_csv(directory / "pool_tok.csv")), 6)
                manifest_path = directory / "cyclic_train_test" / (
                    f"ym_cyclic_{n_particles}pt_split_seed20260401.json"
                )
                report = json.loads(manifest_path.read_text(encoding="utf-8"))
                train = _read_csv(Path(report["outputs"]["train_tokenized"]["path"]))
                test = _read_csv(Path(report["outputs"]["test_tokenized"]["path"]))
                train_targets = {tuple(json.loads(row["simple"])) for row in train}
                test_targets = {tuple(json.loads(row["simple"])) for row in test}
                self.assertEqual(len(train), 5)
                self.assertEqual(len(test), 1)
                self.assertFalse(train_targets & test_targets)
                train_raw = _read_csv(Path(report["outputs"]["train_raw"]["path"]))
                for coefficient in (1, 2):
                    self.assertEqual(
                        sum(row["simple"].startswith(f"{coefficient}*") for row in train_raw), 2,
                    )
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
                generate, "build_dataset_batched", return_value=_pool(4)
            ), mock.patch.object(splitter, "split_cyclic_dataset", return_value=report) as split:
                cyclic.main(_args(directory) + [
                    "--test-size", "1", "--split-seed", "19", "--split-output-dir",
                    str(output_dir), "--split-overwrite", "--tokenizer-max-particles", "5",
                ])
            split.assert_called_once_with(
                directory / "pool.csv", directory / "pool_tok.csv",
                n_particles=4, test_size=1, seed=19, output_dir=output_dir,
                tokenizer_max_particles=5, overwrite=True,
            )

    def test_no_split_preserves_existing_generation_behavior(self) -> None:
        for extra in ([], ["--test-size", "0"]):
            with self.subTest(extra=extra), contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                generate, "build_dataset_batched", return_value=[]
            ), mock.patch.object(generate, "write_csv"), mock.patch.object(
                generate, "tokenise_csv"
            ), mock.patch.object(splitter, "split_cyclic_dataset") as split:
                cyclic.main(["4", "--samples", "6", "--no-tokenise", "--no-progress"] + extra)
                split.assert_not_called()

    def test_invalid_split_settings_fail_before_generation(self) -> None:
        cases = (
            (["--test-size", "-1"], "--test-size must be non-negative"),
            (["--test-size", "6"], "--test-size must be smaller than --samples"),
            (["--test-size", "7"], "--test-size must be smaller than --samples"),
            (["--test-size", "1", "--no-tokenise"], "requires tokenisation"),
            (["--test-size", "1", "--tokenizer-max-particles", "3"], "must be at least N"),
        )
        for extra, error in cases:
            stderr = io.StringIO()
            with self.subTest(extra=extra), contextlib.redirect_stderr(stderr), mock.patch.object(
                generate, "build_dataset_batched"
            ) as builder, mock.patch.object(generate, "write_csv") as write:
                with self.assertRaises(SystemExit) as raised:
                    cyclic.main(["4", "--samples", "6"] + extra)
                self.assertEqual(raised.exception.code, 2)
                builder.assert_not_called()
                write.assert_not_called()
                self.assertIn(error, stderr.getvalue())

    def test_unsupported_particle_count_fails_before_generation(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(
            generate, "build_dataset_batched"
        ) as builder:
            with self.assertRaises(SystemExit) as raised:
                cyclic.main(["3", "--samples", "6", "--test-size", "1"])
            self.assertEqual(raised.exception.code, 2)
            builder.assert_not_called()

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
                generate, "build_dataset_batched", return_value=_pool(4)
            ), mock.patch.object(
                splitter, "split_cyclic_dataset", side_effect=ValueError("not enough distinct targets")
            ):
                with self.assertRaises(SystemExit) as raised:
                    cyclic.main(_args(directory) + ["--test-size", "1"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("not enough distinct targets", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
