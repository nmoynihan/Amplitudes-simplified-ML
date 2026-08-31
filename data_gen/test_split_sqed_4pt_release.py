"""Tests for the verified 4PT SQED train/test release builder."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import gen_data
from . import split_sqed_4pt_release as release


def _pair(index: int) -> tuple[str, str]:
    simple = f"{index}*(p_1 · F_2 · p_3)"
    scrambled = (
        f"{index}*(p_1 · p_2*e_2 · p_3 - e_2 · p_1*p_2 · p_3)"
    )
    return simple, scrambled


def _open_csv(path: Path, mode: str):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _write_fixture(
    raw_path: Path,
    token_path: Path,
    rows: list[tuple[str, str]],
) -> None:
    tokenizer = release.ScatteringAmplitudeTokenizer(
        max_particles=8,
        max_sequence_length=None,
    )
    with _open_csv(raw_path, "w") as raw_handle, _open_csv(
        token_path, "w"
    ) as token_handle:
        raw_writer = csv.writer(raw_handle)
        token_writer = csv.writer(token_handle)
        raw_writer.writerow(release.CSV_HEADER)
        token_writer.writerow(release.CSV_HEADER)
        for simple, scrambled in rows:
            raw_writer.writerow((simple, scrambled))
            token_writer.writerow(
                (
                    json.dumps(tokenizer.encode_infix(simple)),
                    json.dumps(tokenizer.encode_infix(scrambled)),
                )
            )


def _read_rows(path: Path) -> list[tuple[str, str]]:
    with _open_csv(path, "r") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader)) != release.CSV_HEADER:
            raise AssertionError("bad output header")
        return [(row[0], row[1]) for row in reader]


class SplitSQED4PTReleaseTests(unittest.TestCase):
    def _fixture_args(self, directory: Path):
        raw_path = directory / "candidate.csv"
        token_path = directory / "candidate_tok.csv"
        # broad: six candidates -> five unique release rows
        # cover: four candidates -> one cross-stratum duplicate + three rows
        # hard: three candidates -> first two release rows + one spare row
        rows = [
            _pair(1),
            _pair(1),
            _pair(2),
            _pair(3),
            _pair(4),
            _pair(5),
            _pair(2),
            _pair(6),
            _pair(7),
            _pair(8),
            _pair(9),
            _pair(10),
            _pair(11),
        ]
        _write_fixture(raw_path, token_path, rows)
        args = release.build_parser().parse_args(
            [
                "--raw",
                str(raw_path),
                "--tokenised",
                str(token_path),
                "--generation-log",
                str(directory / "missing.log"),
                "--output-dir",
                str(directory / "release"),
                "--broad-candidate-rows",
                "6",
                "--sqed-cover-candidate-rows",
                "4",
                "--hard-candidate-rows",
                "3",
                "--broad-release-rows",
                "5",
                "--sqed-cover-release-rows",
                "3",
                "--hard-release-rows",
                "2",
                "--broad-test-rows",
                "2",
                "--sqed-cover-test-rows",
                "1",
                "--hard-test-rows",
                "1",
                "--progress-every",
                "0",
            ]
        )
        return args, rows

    def test_writes_exact_target_disjoint_gzip_release_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args, _rows = self._fixture_args(directory)
            manifest = release.build_release(args)
            paths = release.output_paths(
                args.output_dir,
                release_rows=10,
                test_rows=4,
            )

            train_rows = _read_rows(paths.train_raw)
            test_rows = _read_rows(paths.test_raw)
            train_tokens = _read_rows(paths.train_tokenised)
            test_tokens = _read_rows(paths.test_tokenised)
            self.assertEqual(len(train_rows), 6)
            self.assertEqual(len(test_rows), 4)
            self.assertEqual(len(train_rows), len(train_tokens))
            self.assertEqual(len(test_rows), len(test_tokens))
            self.assertEqual(len(set(train_rows + test_rows)), 10)
            self.assertTrue(
                {simple for simple, _ in train_rows}.isdisjoint(
                    {simple for simple, _ in test_rows}
                )
            )
            for path in paths.csv_paths():
                with path.open("rb") as handle:
                    self.assertEqual(handle.read(2), b"\x1f\x8b")

            self.assertEqual(manifest["dataset"]["train_rows"], 6)
            self.assertEqual(manifest["dataset"]["test_rows"], 4)
            self.assertEqual(
                manifest["verification"][
                    "global_pair_duplicates_removed_by_stratum"
                ],
                {"broad": 1, "sqed_cover": 1, "hard": 0},
            )
            self.assertEqual(
                manifest["verification"]["train_test_target_overlap"],
                0,
            )
            on_disk = json.loads(paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["outputs"], manifest["outputs"])

    def test_release_csvs_are_byte_deterministic(self) -> None:
        hashes: list[list[str]] = []
        for run in range(2):
            with tempfile.TemporaryDirectory() as temp_dir:
                directory = Path(temp_dir)
                args, _rows = self._fixture_args(directory)
                args.split_seed = 99
                release.build_release(args)
                paths = release.output_paths(
                    args.output_dir,
                    release_rows=10,
                    test_rows=4,
                )
                hashes.append(
                    [release._sha256_file(path) for path in paths.csv_paths()]
                )
        self.assertEqual(hashes[0], hashes[1])

    def test_repeated_and_ambiguous_inputs_are_removed_and_refilled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "candidate.csv"
            token_path = directory / "candidate_tok.csv"
            simple, scrambled = _pair(1)
            rows = [
                (simple, scrambled),
                (simple, scrambled.replace(" · ", "·")),
                (_pair(99)[0], scrambled),
                _pair(2),
                _pair(3),
                _pair(4),
                _pair(5),
                _pair(6),
            ]
            _write_fixture(raw_path, token_path, rows)
            args = release.build_parser().parse_args(
                [
                    "--raw",
                    str(raw_path),
                    "--tokenised",
                    str(token_path),
                    "--output-dir",
                    str(directory / "release"),
                    "--broad-candidate-rows",
                    "4",
                    "--sqed-cover-candidate-rows",
                    "2",
                    "--hard-candidate-rows",
                    "2",
                    "--broad-release-rows",
                    "2",
                    "--sqed-cover-release-rows",
                    "2",
                    "--hard-release-rows",
                    "2",
                    "--broad-test-rows",
                    "1",
                    "--sqed-cover-test-rows",
                    "1",
                    "--hard-test-rows",
                    "1",
                    "--progress-every",
                    "0",
                ]
            )

            manifest = release.build_release(args)
            self.assertEqual(
                manifest["verification"][
                    "repeated_scrambled_inputs_removed_by_stratum"
                ],
                {"broad": 1, "sqed_cover": 0, "hard": 0},
            )
            self.assertEqual(
                manifest["verification"][
                    "ambiguous_scrambled_inputs_removed_by_stratum"
                ],
                {"broad": 1, "sqed_cover": 0, "hard": 0},
            )
            paths = release.output_paths(
                args.output_dir,
                release_rows=6,
                test_rows=3,
            )
            output_rows = _read_rows(paths.train_raw) + _read_rows(paths.test_raw)
            tokenised_inputs = [
                tuple(
                    release.ScatteringAmplitudeTokenizer(
                        max_particles=8,
                        max_sequence_length=None,
                    ).encode_infix(row[1])
                )
                for row in output_rows
            ]
            self.assertEqual(len(tokenised_inputs), len(set(tokenised_inputs)))

    def test_zero_summand_targets_are_removed_and_refilled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "candidate.csv"
            token_path = directory / "candidate_tok.csv"
            rows = [
                ("p_1 · F_2 · p_1", _pair(101)[1]),
                (
                    "p_1 · F_2 · p_1 + p_1 · F_2 · p_3",
                    _pair(102)[1],
                ),
                _pair(1),
                _pair(2),
                _pair(3),
                _pair(4),
                _pair(5),
                _pair(6),
            ]
            _write_fixture(raw_path, token_path, rows)
            args = release.build_parser().parse_args(
                [
                    "--raw",
                    str(raw_path),
                    "--tokenised",
                    str(token_path),
                    "--output-dir",
                    str(directory / "release"),
                    "--broad-candidate-rows",
                    "4",
                    "--sqed-cover-candidate-rows",
                    "2",
                    "--hard-candidate-rows",
                    "2",
                    "--broad-release-rows",
                    "2",
                    "--sqed-cover-release-rows",
                    "2",
                    "--hard-release-rows",
                    "2",
                    "--broad-test-rows",
                    "1",
                    "--sqed-cover-test-rows",
                    "1",
                    "--hard-test-rows",
                    "1",
                    "--progress-every",
                    "0",
                ]
            )

            manifest = release.build_release(args)
            verification = manifest["verification"]
            self.assertEqual(
                verification["all_zero_targets_removed_by_stratum"],
                {"broad": 1, "sqed_cover": 0, "hard": 0},
            )
            self.assertEqual(
                verification["mixed_zero_summand_targets_removed_by_stratum"],
                {"broad": 1, "sqed_cover": 0, "hard": 0},
            )
            paths = release.output_paths(
                args.output_dir,
                release_rows=6,
                test_rows=3,
            )
            for simple, _scrambled in _read_rows(paths.train_raw) + _read_rows(
                paths.test_raw
            ):
                analysis = release.analyze_simple_expression(
                    simple,
                    assumptions=release.SQED_4PT_ASSUMPTIONS,
                )
                self.assertEqual(analysis.classification, "clean")

    def test_corrupt_token_row_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args, _rows = self._fixture_args(directory)
            with args.tokenised.open(newline="", encoding="utf-8") as handle:
                token_rows = list(csv.reader(handle))
            token_rows[3][0] = json.dumps([999])
            with args.tokenised.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(token_rows)

            with self.assertRaisesRegex(ValueError, "raw/token mismatch"):
                release.build_release(args)
            self.assertFalse(args.output_dir.exists())

    def test_insufficient_singleton_targets_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "candidate.csv"
            token_path = directory / "candidate_tok.csv"
            simple = "p_1 · F_2 · p_3"
            rows = [
                (simple, _pair(index)[1])
                for index in range(1, 7)
            ]
            _write_fixture(raw_path, token_path, rows)
            args = release.build_parser().parse_args(
                [
                    "--raw",
                    str(raw_path),
                    "--tokenised",
                    str(token_path),
                    "--output-dir",
                    str(directory / "release"),
                    "--broad-candidate-rows",
                    "2",
                    "--sqed-cover-candidate-rows",
                    "2",
                    "--hard-candidate-rows",
                    "2",
                    "--broad-release-rows",
                    "2",
                    "--sqed-cover-release-rows",
                    "2",
                    "--hard-release-rows",
                    "2",
                    "--broad-test-rows",
                    "1",
                    "--sqed-cover-test-rows",
                    "1",
                    "--hard-test-rows",
                    "1",
                    "--progress-every",
                    "0",
                ]
            )

            with self.assertRaisesRegex(ValueError, "singleton targets"):
                release.build_release(args)
            self.assertFalse(args.output_dir.exists())

    def test_existing_destination_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args, _rows = self._fixture_args(directory)
            path = release.output_paths(
                args.output_dir,
                release_rows=10,
                test_rows=4,
            ).train_raw
            path.parent.mkdir(parents=True)
            path.write_text("sentinel", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                release.build_release(args)
            self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")


class DeterministicValidationTests(unittest.TestCase):
    def test_single_field_chain_never_uses_equal_endpoints(self) -> None:
        gen_data.random.seed(1234)
        endpoints = [gen_data._chain_endpoints((2,), 4) for _ in range(500)]
        self.assertTrue(all(left != right for left, right in endpoints))

    def test_even_field_chain_can_still_use_equal_endpoints(self) -> None:
        gen_data.random.seed(5678)
        endpoints = [gen_data._chain_endpoints((2, 3), 4) for _ in range(500)]
        self.assertTrue(any(left == right for left, right in endpoints))

    def test_three_field_middle_label_never_matches_equal_endpoints(self) -> None:
        gen_data.random.seed(6789)
        endpoints = [
            gen_data._chain_endpoints((2, 3, 4), 5)
            for _ in range(1000)
        ]
        self.assertTrue(
            all(not (left == right == 3) for left, right in endpoints)
        )

    def test_generated_4pt_targets_have_no_manifest_zero_summands(self) -> None:
        gen_data.random.seed(9012)
        generated = []
        for _ in range(100):
            built = gen_data._build_base_expression(
                4,
                unit_probability=0.9,
                old_style_probability=0.4,
                denom_repeat_probability=0.35,
                scalar_power_probability=0.15,
                use_denominators=True,
                min_terms=1,
                max_terms=3,
            )
            if built is not None:
                generated.append(built[0])
        self.assertTrue(generated)
        self.assertTrue(
            all(
                gen_data._is_manifestly_clean_sqed_target(expression, 4)
                for expression in generated
            )
        )

    def test_numerically_zero_pair_is_rejected_when_required(self) -> None:
        with mock.patch.object(
            gen_data,
            "generate_kinematics",
            return_value=(object(), object()),
        ), mock.patch.object(gen_data, "eval_infix_numeric", return_value=0.0):
            valid, reason = gen_data._validate_pair(
                "p_1 · p_2",
                "p_2 · p_1",
                4,
                2.0,
                n_checks=1,
                pol_modes=("coulomb",),
                require_nonzero=True,
            )
        self.assertFalse(valid)
        self.assertEqual(reason, "all-checks:numerically-zero")

    def test_validation_kinematics_seeds_are_stable_and_non_null(self) -> None:
        calls: list[int] = []

        def fake_kinematics(_n, *, M, pol_mode, seed):
            self.assertEqual(M, 2.0)
            self.assertEqual(pol_mode, "coulomb")
            self.assertIsInstance(seed, int)
            calls.append(seed)
            return object(), object()

        with mock.patch.object(
            gen_data,
            "generate_kinematics",
            side_effect=fake_kinematics,
        ), mock.patch.object(gen_data, "eval_infix_numeric", return_value=1.0):
            self.assertTrue(
                gen_data._validate_pair(
                    "p_1 · p_2",
                    "p_2 · p_1",
                    4,
                    2.0,
                    n_checks=3,
                    pol_modes=("coulomb",),
                )[0]
            )
            first = list(calls)
            calls.clear()
            self.assertTrue(
                gen_data._validate_pair(
                    "p_1 · p_2",
                    "p_2 · p_1",
                    4,
                    2.0,
                    n_checks=3,
                    pol_modes=("coulomb",),
                )[0]
            )

        self.assertEqual(first, calls)
        self.assertEqual(len(set(first)), 3)


if __name__ == "__main__":
    unittest.main()
