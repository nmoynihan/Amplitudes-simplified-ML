"""Tests for aligned extraction of corrected Yang--Mills test rows."""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, TextIO
from unittest import mock

from . import split_clean_train_test as splitter


def _open_fixture(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _write_rows(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    with _open_fixture(path, "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(splitter.CSV_HEADER)
        writer.writerows(rows)


def _read_rows(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader)) != splitter.CSV_HEADER:
            raise AssertionError("unexpected test output header")
        return [(row[0], row[1]) for row in reader]


class SplitCleanTrainTestTests(unittest.TestCase):
    def _make_pair(
        self,
        directory: Path,
        *,
        n_particles: int,
        rows: int,
        compressed: bool,
    ) -> tuple[Path, Path, list[tuple[str, str]], list[tuple[str, str]]]:
        suffix = ".csv.gz" if compressed else ".csv"
        raw_path = directory / f"source_{n_particles}pt{suffix}"
        token_path = directory / f"source_{n_particles}pt_tok{suffix}"
        tokenizer = splitter._core.ScatteringAmplitudeTokenizer(
            max_particles=n_particles,
            max_sequence_length=None,
        )
        raw_rows = [
            (
                f"{index + 1}*(p1·F2·p3)",
                f"(p1·F2·p3)*{index + 1}",
            )
            for index in range(rows)
        ]
        token_rows = [
            (
                json.dumps(tokenizer.encode_infix(simple)),
                json.dumps(tokenizer.encode_infix(scrambled)),
            )
            for simple, scrambled in raw_rows
        ]
        _write_rows(raw_path, raw_rows)
        _write_rows(token_path, token_rows)
        return raw_path, token_path, raw_rows, token_rows

    def _args(
        self,
        directory: Path,
        *,
        rows: int = 10,
        test_size: int = 2,
    ):
        raw4, token4, _raw_rows4, _token_rows4 = self._make_pair(
            directory,
            n_particles=4,
            rows=rows,
            compressed=True,
        )
        raw5, token5, _raw_rows5, _token_rows5 = self._make_pair(
            directory,
            n_particles=5,
            rows=rows,
            compressed=False,
        )
        return splitter.build_parser().parse_args(
            [
                "--raw-4pt",
                str(raw4),
                "--tokenised-4pt",
                str(token4),
                "--raw-5pt",
                str(raw5),
                "--tokenised-5pt",
                str(token5),
                "--output-dir",
                str(directory / "split"),
                "--test-size",
                str(test_size),
                "--expected-rows-4pt",
                str(rows),
                "--expected-rows-5pt",
                str(rows),
                "--seed-4pt",
                "401",
                "--seed-5pt",
                "501",
                "--tokenizer-max-particles-4pt",
                "4",
                "--tokenizer-max-particles-5pt",
                "5",
                "--progress-every",
                "0",
            ]
        )

    def test_writes_eight_plain_csvs_with_matching_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory)
            summaries = splitter.split_datasets(args)

            self.assertEqual(len(summaries), 2)
            self.assertEqual([summary.source_rows for summary in summaries], [10, 10])
            self.assertEqual([summary.train_rows for summary in summaries], [8, 8])
            self.assertEqual([summary.test_rows for summary in summaries], [2, 2])

            output_files = sorted((directory / "split").glob("*.csv"))
            self.assertEqual(len(output_files), 8)
            self.assertTrue(all(path.suffix == ".csv" for path in output_files))

            for summary in summaries:
                train_raw = _read_rows(summary.paths.train_raw)
                train_token = _read_rows(summary.paths.train_tokenised)
                test_raw = _read_rows(summary.paths.test_raw)
                test_token = _read_rows(summary.paths.test_tokenised)
                self.assertEqual(len(train_raw), len(train_token))
                self.assertEqual(len(test_raw), len(test_token))
                self.assertEqual(
                    set(train_raw).union(test_raw),
                    {
                        (
                            f"{index + 1}*(p1·F2·p3)",
                            f"(p1·F2·p3)*{index + 1}",
                        )
                        for index in range(10)
                    },
                )
                self.assertTrue(set(train_raw).isdisjoint(test_raw))

                tokenizer = splitter._core.ScatteringAmplitudeTokenizer(
                    max_particles=summary.n_particles,
                    max_sequence_length=None,
                )
                for raw_row, token_row in zip(
                    train_raw + test_raw, train_token + test_token
                ):
                    for raw_expression, encoded_text in zip(raw_row, token_row):
                        self.assertEqual(
                            json.loads(encoded_text),
                            tokenizer.encode_infix(raw_expression),
                        )

                expected_indices = sorted(
                    random.Random(summary.seed).sample(range(10), 2)
                )
                self.assertEqual(
                    test_raw,
                    [
                        (
                            f"{index + 1}*(p1·F2·p3)",
                            f"(p1·F2·p3)*{index + 1}",
                        )
                        for index in expected_indices
                    ],
                )

    def test_bad_token_alignment_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=6, test_size=2)
            token_path = args.tokenised_5pt
            with token_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            rows[3][0] = json.dumps([999])
            _write_rows(token_path, [(row[0], row[1]) for row in rows[1:]])

            with self.assertRaisesRegex(ValueError, "raw/token mismatch"):
                splitter.split_datasets(args)

            self.assertEqual(list((directory / "split").glob("*.csv")), [])
            self.assertEqual(list((directory / "split").glob(".*.tmp")), [])

    def test_unequal_source_lengths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=6, test_size=2)
            token_path = args.tokenised_4pt
            with gzip.open(
                token_path, "rt", newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.reader(handle))
            with gzip.open(
                token_path, "wt", newline="", encoding="utf-8"
            ) as handle:
                csv.writer(handle).writerows(rows[:-1])

            with self.assertRaisesRegex(ValueError, "more rows"):
                splitter.split_datasets(args)

            self.assertEqual(list((directory / "split").glob("*.csv")), [])

    def test_existing_output_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=6, test_size=2)
            path = splitter.output_paths(
                args.output_dir,
                n_particles=4,
                source_rows=6,
                test_size=2,
            ).train_raw
            path.parent.mkdir(parents=True)
            path.write_text("sentinel", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                splitter.split_datasets(args)

            self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")
            self.assertEqual(len(list(path.parent.glob("*.csv"))), 1)

    def test_publish_failure_restores_all_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=6, test_size=2)
            args.overwrite = True
            destinations = [
                path
                for n_particles in (4, 5)
                for path in splitter.output_paths(
                    args.output_dir,
                    n_particles=n_particles,
                    source_rows=6,
                    test_size=2,
                ).all()
            ]
            expected: dict[Path, str] = {}
            for index, path in enumerate(destinations):
                path.parent.mkdir(parents=True, exist_ok=True)
                content = f"old output {index}\n"
                path.write_text(content, encoding="utf-8")
                expected[path] = content

            real_publish = splitter._publish
            calls = 0

            def fail_during_publish(temp_path, destination, *, overwrite):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected publication failure")
                return real_publish(
                    temp_path,
                    destination,
                    overwrite=overwrite,
                )

            with mock.patch.object(
                splitter,
                "_publish",
                side_effect=fail_during_publish,
            ):
                with self.assertRaisesRegex(
                    OSError, "injected publication failure"
                ):
                    splitter.split_datasets(args)

            for path, content in expected.items():
                self.assertEqual(path.read_text(encoding="utf-8"), content)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_publish_failure_restores_symlink_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=6, test_size=2)
            args.overwrite = True
            destinations = [
                path
                for n_particles in (4, 5)
                for path in splitter.output_paths(
                    args.output_dir,
                    n_particles=n_particles,
                    source_rows=6,
                    test_size=2,
                ).all()
            ]
            for index, path in enumerate(destinations):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"old output {index}\n", encoding="utf-8")

            symlink_path = destinations[0]
            symlink_path.unlink()
            symlink_target = args.output_dir / "original-output.csv"
            symlink_target.write_text("original target\n", encoding="utf-8")
            symlink_path.symlink_to(symlink_target.name)

            real_publish = splitter._publish
            calls = 0

            def fail_during_publish(temp_path, destination, *, overwrite):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected publication failure")
                return real_publish(
                    temp_path,
                    destination,
                    overwrite=overwrite,
                )

            with mock.patch.object(
                splitter,
                "_publish",
                side_effect=fail_during_publish,
            ):
                with self.assertRaisesRegex(
                    OSError, "injected publication failure"
                ):
                    splitter.split_datasets(args)

            self.assertTrue(symlink_path.is_symlink())
            self.assertEqual(os.readlink(symlink_path), symlink_target.name)
            self.assertEqual(
                symlink_path.read_text(encoding="utf-8"),
                "original target\n",
            )
            self.assertEqual(list(args.output_dir.glob(".*.tmp")), [])

    def test_mid_group_temporary_failure_leaks_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=6, test_size=2)
            real_temporary_sibling = splitter._temporary_sibling
            calls = 0

            def fail_during_allocation(destination):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected temporary allocation failure")
                return real_temporary_sibling(destination)

            with mock.patch.object(
                splitter,
                "_temporary_sibling",
                side_effect=fail_during_allocation,
            ):
                with self.assertRaisesRegex(
                    OSError, "injected temporary allocation failure"
                ):
                    splitter.split_datasets(args)

            self.assertEqual(list(args.output_dir.glob(".*.tmp")), [])

    def test_no_report_falls_back_to_counting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=7, test_size=2)
            args.expected_rows_4pt = None
            args.expected_rows_5pt = None
            args.verify_alignment = "test"
            summaries = splitter.split_datasets(args)
            self.assertEqual([summary.source_rows for summary in summaries], [7, 7])
            self.assertEqual([summary.train_rows for summary in summaries], [5, 5])

    def test_matching_report_without_row_counts_falls_back_to_counting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=7, test_size=2)
            args.expected_rows_4pt = None
            report_path = splitter._report_path_for(args.raw_4pt)
            assert report_path is not None
            report_path.write_text(
                json.dumps(
                    {
                        "outputs": {
                            "raw": {"path": str(args.raw_4pt.resolve())},
                            "tokenized": {
                                "path": str(args.tokenised_4pt.resolve())
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            summaries = splitter.split_datasets(args)

            self.assertEqual(summaries[0].source_rows, 7)
            self.assertEqual(summaries[0].train_rows, 5)

    def test_matching_legacy_report_without_tokenizer_size_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "legacy.csv"
            token_path = directory / "legacy_tok.csv"
            raw_path.touch()
            token_path.touch()
            report_path = splitter._report_path_for(raw_path)
            assert report_path is not None
            report_path.write_text(
                json.dumps(
                    {
                        "outputs": {
                            "raw": {"path": str(raw_path.resolve())},
                            "tokenized": {"path": str(token_path.resolve())},
                        }
                    }
                ),
                encoding="utf-8",
            )
            pair = splitter.InputPair(
                n_particles=4,
                raw=raw_path,
                tokenised=token_path,
                seed=401,
            )

            self.assertEqual(splitter._tokenizer_size(pair), 8)

    def test_reports_infer_independent_tokenizer_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = self._args(directory, rows=7, test_size=2)
            args.expected_rows_4pt = None
            args.expected_rows_5pt = None
            args.tokenizer_max_particles_4pt = None
            args.tokenizer_max_particles_5pt = None
            for n_particles, raw_path, token_path in (
                (4, args.raw_4pt, args.tokenised_4pt),
                (5, args.raw_5pt, args.tokenised_5pt),
            ):
                report_path = splitter._report_path_for(raw_path)
                assert report_path is not None
                report_path.write_text(
                    json.dumps(
                        {
                            "outputs": {
                                "raw": {
                                    "path": str(raw_path.resolve()),
                                    "rows": 7,
                                },
                                "tokenized": {
                                    "path": str(token_path.resolve()),
                                    "rows": 7,
                                },
                            },
                            "settings": {
                                "tokenizer_max_particles": n_particles,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            summaries = splitter.split_datasets(args)
            self.assertEqual([summary.source_rows for summary in summaries], [7, 7])


if __name__ == "__main__":
    unittest.main()
