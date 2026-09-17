"""Leakage and publication checks for the fixed-order Yang--Mills split."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Iterable
from unittest import mock

from . import split_clean_train_test as legacy_splitter
from . import split_cyclic_train_test as splitter


def _write_rows(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "wt" if path.suffix == ".gz" else "w"
    with opener(path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("simple", "scrambled"))
        writer.writerows(rows)


def _read_rows(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["simple", "scrambled"]:
            raise AssertionError("unexpected output header")
        return [(row[0], row[1]) for row in reader]


def _block(n_particles: int = 4) -> str:
    trace = "·".join(f"F_{index}" for index in range(2, n_particles))
    return f"(p_1·F_{n_particles}·F_1·p_2)*Tr({trace})"


def _unique_rows(count: int, n_particles: int = 4) -> list[tuple[str, str]]:
    block = _block(n_particles)
    return [(f"{index + 2}*{block}", f"({block})*{index + 2}") for index in range(count)]


class SplitCyclicTrainTestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = legacy_splitter._core.ScatteringAmplitudeTokenizer(
            max_particles=8, max_sequence_length=None
        )

    def _sources(
        self,
        directory: Path,
        rows: list[tuple[str, str]],
        *,
        compressed: bool = False,
    ) -> tuple[Path, Path, list[tuple[str, str]]]:
        suffix = ".csv.gz" if compressed else ".csv"
        raw = directory / f"source{suffix}"
        tokenised = directory / f"source_tok{suffix}"
        token_rows = [
            tuple(json.dumps(self.tokenizer.encode_infix(expr)) for expr in row)
            for row in rows
        ]
        _write_rows(raw, rows)
        _write_rows(tokenised, token_rows)
        return raw, tokenised, token_rows

    def _split(
        self,
        raw: Path,
        tokenised: Path,
        output_dir: Path,
        *,
        test_size: int = 2,
        n_particles: int = 4,
        seed: int = 401,
        overwrite: bool = False,
    ) -> dict:
        return splitter.split_cyclic_dataset(
            raw,
            tokenised,
            n_particles=n_particles,
            test_size=test_size,
            seed=seed,
            output_dir=output_dir,
            overwrite=overwrite,
        )

    def _output_rows(self, report: dict, key: str) -> list[tuple[str, str]]:
        rows = _read_rows(Path(report["outputs"][key]["path"]))
        self.assertEqual(len(rows), report["outputs"][key]["rows"])
        return rows

    def _assert_no_outputs(self, output_dir: Path) -> None:
        self.assertEqual(list(output_dir.glob("*")), [])

    def _assert_valid_partition(self, report: dict, source_rows: list[tuple[str, str]]) -> None:
        train = self._output_rows(report, "train_raw")
        test = self._output_rows(report, "test_raw")
        self.assertEqual(Counter(train + test), Counter(source_rows))
        for partition, raw_rows in (("train", train), ("test", test)):
            token_rows = self._output_rows(report, f"{partition}_tokenized")
            self.assertEqual(len(raw_rows), len(token_rows))
            for raw_row, token_row in zip(raw_rows, token_rows):
                for raw_expression, encoded in zip(raw_row, token_row):
                    self.assertEqual(
                        json.loads(encoded), self.tokenizer.encode_infix(raw_expression)
                    )
        for column in (0, 1):
            train_keys = {tuple(self.tokenizer.encode_infix(row[column])) for row in train}
            test_keys = {tuple(self.tokenizer.encode_infix(row[column])) for row in test}
            self.assertTrue(train_keys.isdisjoint(test_keys))
        self.assertEqual(report["verification"]["train_test_target_overlap"], 0)
        self.assertEqual(report["verification"]["train_test_input_overlap"], 0)
        self.assertIs(report["verification"]["all_raw_token_rows_retokenized"], True)

    def test_duplicate_targets_and_all_scrambled_variants_stay_in_training(self) -> None:
        block = _block()
        duplicate = (f"2*{block}", f"({block})*2")
        alternative = (duplicate[0], f"2*({block})")
        rows = [duplicate, duplicate, alternative] + _unique_rows(4)[1:]
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw, tokenised, _ = self._sources(directory, rows)
            report = self._split(raw, tokenised, directory / "split", test_size=3)
            self._assert_valid_partition(report, rows)
            self.assertEqual(Counter(self._output_rows(report, "train_raw")), Counter(rows[:3]))
            self.assertEqual(len(self._output_rows(report, "test_raw")), 3)
            self.assertEqual(report["split"]["eligible_singleton_targets"], 3)

    def test_whitespace_and_token_json_formatting_do_not_hide_duplicate_targets(self) -> None:
        plain, scrambled = _unique_rows(1)[0]
        spaced = plain.replace("*", " * ").replace("·", " · ")
        rows = [(plain, scrambled), (spaced, f"2*({_block()})")] + _unique_rows(4)[1:]
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw, tokenised, token_rows = self._sources(directory, rows)
            token_rows[1] = tuple(
                json.dumps(json.loads(encoded), separators=(",", ":"))
                for encoded in token_rows[1]
            )
            _write_rows(tokenised, token_rows)
            report = self._split(raw, tokenised, directory / "split", test_size=3)
            self._assert_valid_partition(report, rows)
            self.assertEqual(self._output_rows(report, "train_raw"), rows[:2])

    def test_repeated_scrambled_inputs_are_excluded_even_for_distinct_targets(self) -> None:
        block = _block()
        shared_input = f"({block})*2"
        rows = [(f"2*{block}", shared_input), (shared_input, shared_input)]
        rows += _unique_rows(4)[1:]
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw, tokenised, _ = self._sources(directory, rows)
            report = self._split(raw, tokenised, directory / "split", test_size=3)
            self._assert_valid_partition(report, rows)
            self.assertEqual(self._output_rows(report, "train_raw"), rows[:2])
            self.assertEqual(report["split"]["eligible_singleton_targets"], 3)

    def test_seed_is_reproducible_and_raw_cyclic_block_text_is_preserved(self) -> None:
        rows = _unique_rows(12)
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw, tokenised, _ = self._sources(directory, rows)
            before = (raw.read_bytes(), tokenised.read_bytes())
            first = self._split(raw, tokenised, directory / "first", test_size=4)
            second = self._split(raw, tokenised, directory / "second", test_size=4)
            self._assert_valid_partition(first, rows)
            self._assert_valid_partition(second, rows)
            for key in ("train_raw", "train_tokenized", "test_raw", "test_tokenized"):
                self.assertEqual(
                    Path(first["outputs"][key]["path"]).read_bytes(),
                    Path(second["outputs"][key]["path"]).read_bytes(),
                )
            for expression, _scrambled in self._output_rows(first, "train_raw") + self._output_rows(first, "test_raw"):
                self.assertIn("F_4·F_1", expression)
                self.assertIn("Tr(F_2·F_3)", expression)
            self.assertEqual((raw.read_bytes(), tokenised.read_bytes()), before)

    def test_gzip_sources_and_manifest_for_four_and_five_points(self) -> None:
        for n_particles in (4, 5):
            with self.subTest(n_particles=n_particles), tempfile.TemporaryDirectory() as temp_dir:
                directory = Path(temp_dir)
                rows = _unique_rows(6, n_particles)
                raw, tokenised, _ = self._sources(directory, rows, compressed=True)
                report = self._split(raw, tokenised, directory / "split", n_particles=n_particles)
                self._assert_valid_partition(report, rows)
                manifest = directory / "split" / f"ym_cyclic_{n_particles}pt_split_seed401.json"
                self.assertEqual(json.loads(manifest.read_text(encoding="utf-8")), report)
                self.assertEqual(report["split"]["seed"], 401)
                self.assertEqual(report["split"]["method"], "singleton_target_holdout")
                self.assertEqual(report["split"]["leakage_key"], "sha256(tokenized_simple)")
                self.assertEqual(report["split"]["eligible_singleton_targets"], 6)
                for partition, count in (("train", 4), ("test", 2)):
                    self.assertEqual(
                        Path(report["outputs"][f"{partition}_raw"]["path"]).name,
                        f"ym_cyclic_{n_particles}pt_{partition}_{count}.csv",
                    )
                    self.assertEqual(
                        Path(report["outputs"][f"{partition}_tokenized"]["path"]).name,
                        f"ym_cyclic_{n_particles}pt_{partition}_{count}_tok.csv",
                    )

    def test_insufficient_eligible_targets_publishes_nothing(self) -> None:
        rows = _unique_rows(2)
        rows += [rows[0], rows[0]]
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw, tokenised, _ = self._sources(directory, rows)
            output_dir = directory / "split"
            with self.assertRaises(ValueError):
                self._split(raw, tokenised, output_dir, test_size=2)
            self._assert_no_outputs(output_dir)

    def test_bad_alignment_in_either_column_publishes_nothing(self) -> None:
        for column in (0, 1):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temp_dir:
                directory = Path(temp_dir)
                raw, tokenised, token_rows = self._sources(directory, _unique_rows(6))
                bad_row = list(token_rows[-1])
                bad_row[column] = json.dumps([999])
                token_rows[-1] = tuple(bad_row)
                _write_rows(tokenised, token_rows)
                output_dir = directory / "split"
                with self.assertRaisesRegex(ValueError, "raw/token mismatch"):
                    self._split(raw, tokenised, output_dir)
                self._assert_no_outputs(output_dir)

    def test_non_integer_token_json_is_rejected_before_publication(self) -> None:
        for bad_value in ("[true]", '"not a token list"', "not json"):
            with self.subTest(tokens=bad_value), tempfile.TemporaryDirectory() as temp_dir:
                directory = Path(temp_dir)
                raw, tokenised, token_rows = self._sources(directory, _unique_rows(6))
                token_rows[0] = (bad_value, token_rows[0][1])
                _write_rows(tokenised, token_rows)
                output_dir = directory / "split"
                with self.assertRaises(ValueError):
                    self._split(raw, tokenised, output_dir)
                self._assert_no_outputs(output_dir)

    def test_unequal_source_lengths_are_rejected_before_publication(self) -> None:
        for truncate_raw in (False, True):
            with self.subTest(truncate_raw=truncate_raw), tempfile.TemporaryDirectory() as temp_dir:
                directory = Path(temp_dir)
                rows = _unique_rows(6)
                raw, tokenised, token_rows = self._sources(directory, rows)
                _write_rows(raw if truncate_raw else tokenised, (rows if truncate_raw else token_rows)[:-1])
                output_dir = directory / "split"
                with self.assertRaisesRegex(ValueError, "more rows"):
                    self._split(raw, tokenised, output_dir)
                self._assert_no_outputs(output_dir)

    def test_existing_split_is_protected_unless_overwrite_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            rows = _unique_rows(6)
            raw, tokenised, _ = self._sources(directory, rows)
            output_dir = directory / "split"
            first = self._split(raw, tokenised, output_dir)
            before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
            with self.assertRaises(FileExistsError):
                self._split(raw, tokenised, output_dir)
            self.assertEqual({path.name: path.read_bytes() for path in output_dir.iterdir()}, before)
            overwritten = self._split(raw, tokenised, output_dir, overwrite=True)
            self._assert_valid_partition(overwritten, rows)
            self.assertEqual(first, overwritten)

    def test_output_aliasing_an_input_is_rejected_even_with_overwrite(self) -> None:
        for output_name in ("train_raw", "test_raw", "manifest"):
            with self.subTest(output=output_name), tempfile.TemporaryDirectory() as temp_dir:
                directory = Path(temp_dir)
                raw, tokenised, _ = self._sources(directory, _unique_rows(6))
                destinations = splitter.output_paths(
                    directory, n_particles=4, source_rows=6, test_size=2, seed=401
                )
                aliased_raw = getattr(destinations, output_name)
                raw.rename(aliased_raw)
                before = {path.name: path.read_bytes() for path in directory.iterdir()}
                with self.assertRaisesRegex(ValueError, "replace an input"):
                    self._split(aliased_raw, tokenised, directory, overwrite=True)
                self.assertEqual(
                    {path.name: path.read_bytes() for path in directory.iterdir()}, before
                )

    def test_aligned_source_mutation_between_passes_prevents_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            rows = _unique_rows(6)
            raw, tokenised, token_rows = self._sources(directory, rows)
            output_dir = directory / "split"
            original_scan = splitter._scan_sources

            def mutate_after_scan(*args, **kwargs):
                scan = original_scan(*args, **kwargs)
                # Keep every token key and row count unchanged. The raw text
                # mutation must still invalidate the two-pass source digest.
                changed_rows = [(" " + rows[0][0], rows[0][1]), *rows[1:]]
                _write_rows(raw, changed_rows)
                _write_rows(tokenised, token_rows)
                return scan

            with mock.patch.object(splitter, "_scan_sources", side_effect=mutate_after_scan):
                with self.assertRaisesRegex(ValueError, "source rows changed"):
                    self._split(raw, tokenised, output_dir)
            self._assert_no_outputs(output_dir)

    def test_manifest_publication_failure_rolls_back_the_whole_split(self) -> None:
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite), tempfile.TemporaryDirectory() as temp_dir:
                directory = Path(temp_dir)
                raw, tokenised, _ = self._sources(directory, _unique_rows(6))
                output_dir = directory / "split"
                if overwrite:
                    self._split(raw, tokenised, output_dir)
                before = {path.name: path.read_bytes() for path in output_dir.glob("*")}
                original_publish = splitter._shared._publish

                def fail_manifest(temporary, destination, *, overwrite):
                    if destination.suffix == ".json":
                        raise OSError("simulated manifest publication failure")
                    return original_publish(temporary, destination, overwrite=overwrite)

                with mock.patch.object(
                    splitter._shared, "_publish", side_effect=fail_manifest
                ) as publish:
                    with self.assertRaisesRegex(OSError, "manifest publication failure"):
                        self._split(raw, tokenised, output_dir, overwrite=overwrite)
                self.assertEqual(publish.call_count, 5)
                self.assertEqual(
                    {path.name: path.read_bytes() for path in output_dir.glob("*")}, before
                )


if __name__ == "__main__":
    unittest.main()
