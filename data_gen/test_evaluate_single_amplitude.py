"""Regression tests for the focused first-amplitude evaluator."""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from data_testing import evaluate_single_amplitude as single_eval


class FirstAmplitudeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)

    def test_raw_pair_csv_counts_rows_and_keeps_the_first_pair(self) -> None:
        source = self.directory / "pairs.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["simple", "scrambled", "process"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "simple": "p_1 · p_2",
                    "scrambled": "p_2 · p_1",
                    "process": "4s1h",
                }
            )
            writer.writerow(
                {
                    "simple": "p_1 · p_3",
                    "scrambled": "p_3 · p_1",
                    "process": "3s2h",
                }
            )

        selected = single_eval.select_first_amplitude(source)

        self.assertEqual(selected.source_format, "raw")
        self.assertEqual(selected.evaluator_format, "raw")
        self.assertEqual(selected.total_entries, 2)
        self.assertEqual(selected.simple, "p_1 · p_2")
        self.assertEqual(selected.scrambled, "p_2 · p_1")
        self.assertEqual(selected.process, "4s1h")

        selected_path = self.directory / "selected.csv"
        single_eval.write_selected_amplitude(selected_path, selected)
        with selected_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            rows,
            [
                {
                    "simple": "p_1 · p_2",
                    "scrambled": "p_2 · p_1",
                    "process": "4s1h",
                }
            ],
        )

    def test_gzip_token_csv_counts_rows_and_keeps_first_token_list(self) -> None:
        source = self.directory / "tokens.csv.gz"
        with gzip.open(source, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "tokens", "process"],
            )
            writer.writeheader()
            writer.writerow({"id": 1, "tokens": "[4, 5, 25]", "process": "3s2h"})
            writer.writerow({"id": 2, "tokens": "[4, 6, 26]", "process": "4s1h"})

        selected = single_eval.select_first_amplitude(source)

        self.assertEqual(selected.source_format, "tokens")
        self.assertEqual(selected.total_entries, 2)
        self.assertEqual(json.loads(selected.tokens or ""), [4, 5, 25])
        self.assertEqual(selected.process, "3s2h")

    def test_standard_simple_scrambled_token_csv_uses_scrambled_tokens(self) -> None:
        source = self.directory / "dataset_tok.csv.gz"
        with gzip.open(source, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["simple", "scrambled"])
            writer.writeheader()
            writer.writerow(
                {
                    "simple": "[4, 5, 25]",
                    "scrambled": "[4, 6, 26]",
                }
            )
            writer.writerow(
                {
                    "simple": "[4, 5, 27]",
                    "scrambled": "[4, 6, 28]",
                }
            )

        selected = single_eval.select_first_amplitude(source)

        self.assertEqual(selected.source_format, "token-pair")
        self.assertEqual(selected.evaluator_format, "token-pair")
        self.assertEqual(selected.total_entries, 2)
        self.assertEqual(json.loads(selected.simple_tokens or ""), [4, 5, 25])
        self.assertEqual(json.loads(selected.tokens or ""), [4, 6, 26])

        selected_path = self.directory / "selected_pair.csv"
        single_eval.write_selected_amplitude(selected_path, selected)
        with selected_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            rows,
            [{"simple": "[4, 5, 25]", "scrambled": "[4, 6, 26]"}],
        )

    def test_headerless_feyn_csv_skips_blanks_and_uses_first_entry(self) -> None:
        source = self.directory / "feyn.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([])
            writer.writerow([1, "e_1 · e_2"])
            writer.writerow([])
            writer.writerow([2, "e_1 · p_2"])

        selected = single_eval.select_first_amplitude(source)

        self.assertEqual(selected.source_format, "feyn")
        self.assertEqual(selected.total_entries, 2)
        self.assertEqual(selected.expression, "e_1 · e_2")

    def test_headered_expression_csv_is_normalised_to_feyn_input(self) -> None:
        source = self.directory / "expressions.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "amplitude"])
            writer.writeheader()
            writer.writerow({"id": 1, "amplitude": "p_1 · p_2"})
            writer.writerow({"id": 2, "amplitude": "p_1 · p_3"})

        selected = single_eval.select_first_amplitude(source)

        self.assertEqual(selected.source_format, "expression")
        self.assertEqual(selected.evaluator_format, "feyn")
        self.assertEqual(selected.total_entries, 2)
        self.assertEqual(selected.expression, "p_1 · p_2")

    def test_empty_and_header_only_csvs_are_rejected(self) -> None:
        empty = self.directory / "empty.csv"
        empty.touch()
        with self.assertRaisesRegex(ValueError, "empty"):
            single_eval.select_first_amplitude(empty)

        header_only = self.directory / "header_only.csv"
        header_only.write_text("simple,scrambled\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no data rows"):
            single_eval.select_first_amplitude(header_only)

    def test_token_bools_are_not_accepted_as_integer_ids(self) -> None:
        source = self.directory / "bad_tokens.csv"
        source.write_text('id,tokens\n1,"[4, true, 5]"\n', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "must contain integers"):
            single_eval.select_first_amplitude(source)


class GravityProcessTests(unittest.TestCase):
    def test_selected_process_is_used_and_conflicts_are_rejected(self) -> None:
        selected = single_eval.SelectedAmplitude(
            source_format="expression",
            evaluator_format="feyn",
            total_entries=1,
            expression="p_1 · p_2",
            process="3s2h",
        )
        source = Path("amplitude.csv")

        self.assertEqual(
            single_eval.resolve_gravity_process(
                backend="gravity",
                selected=selected,
                explicit_process=None,
                metadata_path=None,
                source_path=source,
            ),
            "3s2h",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            single_eval.resolve_gravity_process(
                backend="gravity",
                selected=selected,
                explicit_process="4s1h",
                metadata_path=None,
                source_path=source,
            )

    def test_gravity_requires_a_process(self) -> None:
        selected = single_eval.SelectedAmplitude(
            source_format="feyn",
            evaluator_format="feyn",
            total_entries=1,
            expression="p_1 · p_2",
        )
        with self.assertRaisesRegex(ValueError, "needs the first amplitude's process"):
            single_eval.resolve_gravity_process(
                backend="gravity",
                selected=selected,
                explicit_process=None,
                metadata_path=None,
                source_path=Path("amplitude.csv"),
            )

    def test_metadata_expressions_must_align_with_the_selected_pair(self) -> None:
        selected = single_eval.SelectedAmplitude(
            source_format="raw",
            evaluator_format="raw",
            total_entries=2,
            simple="simple first",
            scrambled="scrambled first",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata = Path(temporary_directory) / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["simple", "scrambled", "process"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "simple": "wrong simple",
                        "scrambled": "wrong scrambled",
                        "process": "3s2h",
                    }
                )
                writer.writerow(
                    {
                        "simple": "simple first",
                        "scrambled": "scrambled first",
                        "process": "4s1h",
                    }
                )

            with self.assertRaisesRegex(ValueError, "not aligned"):
                single_eval.read_first_gravity_process(
                    metadata,
                    selected=selected,
                )

    def test_token_pair_aligns_to_expression_metadata(self) -> None:
        selected = single_eval.SelectedAmplitude(
            source_format="token-pair",
            evaluator_format="token-pair",
            total_entries=1,
            simple_tokens="[21, 25, 26]",
            tokens="[21, 26, 25]",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata = Path(temporary_directory) / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["simple", "scrambled", "process"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "simple": "p_1 · p_2",
                        "scrambled": "p_2 · p_1",
                        "process": "4s1h",
                    }
                )

            process = single_eval.read_first_gravity_process(
                metadata,
                selected=selected,
            )

        self.assertEqual(process, "4s1h")


class EvaluatorArgumentTests(unittest.TestCase):
    def test_builds_one_row_ym_beam_evaluation_arguments(self) -> None:
        parser = single_eval.build_parser()
        args = parser.parse_args(
            [
                "models/example.pt",
                "data/example.csv",
                "--numeric-backend",
                "yang-mills",
                "--n-particles",
                "5",
                "--decoding-method",
                "beam",
                "--beam-size",
                "7",
            ]
        )
        selected = single_eval.SelectedAmplitude(
            source_format="feyn",
            evaluator_format="feyn",
            total_entries=3,
            expression="e_1 · e_2",
        )
        selected_path = Path("/tmp/selected.csv")

        evaluator_args = single_eval.build_evaluator_args(
            args,
            selected_path=selected_path,
            selected=selected,
            gravity_process=None,
        )

        self.assertIn("--single-amplitude-input-csv", evaluator_args)
        self.assertEqual(
            evaluator_args[evaluator_args.index("--single-amplitude-input-csv") + 1],
            str(selected_path),
        )
        self.assertEqual(
            evaluator_args[evaluator_args.index("--numeric-backend") + 1],
            "ym",
        )
        self.assertEqual(
            evaluator_args[evaluator_args.index("--n-particles") + 1],
            "5",
        )
        self.assertEqual(
            evaluator_args[evaluator_args.index("--beam-size") + 1],
            "7",
        )
        self.assertIn("--no-plots", evaluator_args)
        self.assertIn("--rerank-numerical", evaluator_args)

    def test_token_pair_handoff_preserves_the_simple_target(self) -> None:
        args = single_eval.build_parser().parse_args(
            [
                "models/example.pt",
                "data/example_tok.csv.gz",
                "--numeric-backend",
                "sqed",
            ]
        )
        selected = single_eval.SelectedAmplitude(
            source_format="token-pair",
            evaluator_format="token-pair",
            total_entries=2,
            simple_tokens="[21, 25, 26]",
            tokens="[21, 26, 25]",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            selected_path = Path(temporary_directory) / "selected.csv"
            single_eval.write_selected_amplitude(selected_path, selected)
            evaluator_args = single_eval.build_evaluator_args(
                args,
                selected_path=selected_path,
                selected=selected,
                gravity_process=None,
            )
            with selected_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["simple"], selected.simple_tokens)
        self.assertEqual(row["scrambled"], selected.tokens)
        self.assertEqual(
            evaluator_args[
                evaluator_args.index("--single-amplitude-input-format") + 1
            ],
            "token-pair",
        )
        self.assertNotIn("--single-amplitude-expression-column", evaluator_args)

    def test_main_reports_multiple_rows_and_delegates_only_the_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "amplitudes.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([1, "e_1 · e_2"])
                writer.writerow([2, "e_1 · p_2"])

            captured: dict[str, object] = {}
            fake_evaluator = types.ModuleType("data_testing.evaluate_model")

            def fake_main(argv: list[str]) -> int:
                captured["argv"] = argv
                selected_path = Path(
                    argv[argv.index("--single-amplitude-input-csv") + 1]
                )
                with selected_path.open(newline="", encoding="utf-8") as handle:
                    captured["rows"] = list(csv.reader(handle))
                return 17

            fake_evaluator.main = fake_main  # type: ignore[attr-defined]
            terminal = io.StringIO()
            with mock.patch.dict(
                sys.modules,
                {"data_testing.evaluate_model": fake_evaluator},
            ), redirect_stdout(terminal):
                exit_code = single_eval.main(
                    [
                        "models/example.pt",
                        str(source),
                        "--numeric-backend",
                        "ym",
                    ]
                )

        self.assertEqual(exit_code, 17)
        self.assertIn("Found 2 amplitude entries", terminal.getvalue())
        self.assertIn("using the first and ignoring 1", terminal.getvalue())
        self.assertEqual(captured["rows"], [["1", "e_1 · e_2"]])


if __name__ == "__main__":
    unittest.main()
