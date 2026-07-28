from __future__ import annotations

import contextlib
import csv
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

from data_gen.filter_antisymmetry_zeros import (
    OnShellAssumptions,
    filter_csv,
)
from data_gen.verify_antisymmetry_filter import (
    VerificationError,
    main as verify_main,
    subtract_expressions,
    verify_filtered_csv,
)


class DifferenceReductionTests(unittest.TestCase):
    def test_old_minus_new_reduces_to_removed_zero_summands(self) -> None:
        old = (
            "p_1 · p_2 "
            "+ p_3 · F_4 · p_3 "
            "- Tr(F_1 · F_2 · F_1)"
        )
        new = "p_1 · p_2"
        delta = subtract_expressions(old, new)
        self.assertEqual(delta.zero_summand_count, 2)
        self.assertEqual(
            delta.reason_counts,
            {
                "antisymmetric_palindrome_chain": 1,
                "antisymmetric_cyclic_trace": 1,
            },
        )

    def test_nonzero_old_minus_new_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            VerificationError,
            "unproved nonzero summand",
        ):
            subtract_expressions("p_1 · p_2", "p_1 · p_3")


class FullCsvReconciliationTests(unittest.TestCase):
    FIELDNAMES = ["id", "simple", "scrambled", "note"]

    def _write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def _source_rows(self) -> list[dict[str, str]]:
        return [
            {
                "id": "clean",
                "simple": "p_1 · p_2",
                "scrambled": "same clean input",
                "note": "unchanged",
            },
            {
                "id": "mixed",
                "simple": "p_1 · p_2 + p_3 · F_4 · p_3",
                "scrambled": "same mixed input",
                "note": "zero summand pruned",
            },
            {
                "id": "dropped",
                "simple": "p_2 · F_5 · p_2",
                "scrambled": "expanded zero",
                "note": "compared with virtual zero",
            },
        ]

    def _cleaned_rows(self) -> list[dict[str, str]]:
        return [
            {
                "id": "clean",
                "simple": "p_1 · p_2",
                "scrambled": "same clean input",
                "note": "unchanged",
            },
            {
                "id": "mixed",
                "simple": "p_1 · p_2",
                "scrambled": "same mixed input",
                "note": "zero summand pruned",
            },
        ]

    def test_every_source_minus_cleaned_entry_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "old.csv.gz"
            cleaned = root / "new.csv.gz"
            self._write_rows(source, self._source_rows())
            self._write_rows(cleaned, self._cleaned_rows())

            stats = verify_filtered_csv(
                source,
                cleaned,
                progress_every=0,
            )
            self.assertEqual(stats.source_rows, 3)
            self.assertEqual(stats.cleaned_rows, 2)
            self.assertEqual(stats.zero_differences, 3)
            self.assertEqual(stats.clean_source_rows, 1)
            self.assertEqual(stats.mixed_source_rows, 1)
            self.assertEqual(stats.identical_pairs, 1)
            self.assertEqual(stats.zero_delta_modified_pairs, 1)
            self.assertEqual(stats.dropped_zero_rows, 1)
            self.assertEqual(stats.delta_zero_summands, 2)

    def test_non_simple_field_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "old.csv.gz"
            cleaned = root / "new.csv.gz"
            cleaned_rows = self._cleaned_rows()
            cleaned_rows[1]["scrambled"] = "changed"
            self._write_rows(source, self._source_rows())
            self._write_rows(cleaned, cleaned_rows)

            with self.assertRaisesRegex(
                VerificationError,
                "non-simple field 'scrambled'",
            ):
                verify_filtered_csv(source, cleaned, progress_every=0)

    def test_extra_cleaned_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "old.csv.gz"
            cleaned = root / "new.csv.gz"
            cleaned_rows = self._cleaned_rows()
            cleaned_rows.append(cleaned_rows[0].copy())
            self._write_rows(source, self._source_rows())
            self._write_rows(cleaned, cleaned_rows)

            with self.assertRaisesRegex(VerificationError, "extra data"):
                verify_filtered_csv(source, cleaned, progress_every=0)

    def test_reordered_or_injected_cancelling_terms_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "old.csv.gz"
            cleaned = root / "new.csv.gz"
            source_rows = [
                {
                    "id": "ordered",
                    "simple": "p_1 · p_2 + p_3 · p_4",
                    "scrambled": "same",
                    "note": "same",
                }
            ]
            self._write_rows(source, source_rows)

            invalid_simples = [
                "p_3 · p_4 + p_1 · p_2",
                "p_1 · p_2 + p_3 · p_4 + p_1 · p_3 - p_1 · p_3",
            ]
            for invalid_simple in invalid_simples:
                with self.subTest(invalid_simple=invalid_simple):
                    cleaned_rows = [source_rows[0].copy()]
                    cleaned_rows[0]["simple"] = invalid_simple
                    self._write_rows(cleaned, cleaned_rows)
                    with self.assertRaisesRegex(
                        VerificationError,
                        "unexpected cleaned expression",
                    ):
                        verify_filtered_csv(source, cleaned, progress_every=0)

    def test_all_zero_source_matches_header_only_cleaned_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "old.csv.gz"
            cleaned = root / "new.csv.gz"
            source_rows = [
                {
                    "id": "zero-1",
                    "simple": "p_1 · F_2 · p_1",
                    "scrambled": "zero",
                    "note": "dropped",
                },
                {
                    "id": "zero-2",
                    "simple": "p_3 · F_4 · p_3",
                    "scrambled": "zero",
                    "note": "dropped",
                },
            ]
            self._write_rows(source, source_rows)
            self._write_rows(cleaned, [])

            stats = verify_filtered_csv(source, cleaned, progress_every=0)
            self.assertEqual(stats.source_rows, 2)
            self.assertEqual(stats.cleaned_rows, 0)
            self.assertEqual(stats.zero_differences, 2)
            self.assertEqual(stats.dropped_zero_rows, 2)

    def test_canceling_removed_zero_terms_are_still_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "old.csv.gz"
            cleaned = root / "new.csv.gz"
            zero = "p_1 · F_2 · p_1"
            source_rows = [
                {
                    "id": "canceling-zeros",
                    "simple": f"p_3 · p_4 + {zero} - {zero}",
                    "scrambled": "same",
                    "note": "both manifestly zero terms were removed",
                }
            ]
            cleaned_rows = [source_rows[0].copy()]
            cleaned_rows[0]["simple"] = "p_3 · p_4"
            self._write_rows(source, source_rows)
            self._write_rows(cleaned, cleaned_rows)

            stats = verify_filtered_csv(source, cleaned, progress_every=0)
            self.assertEqual(stats.identical_pairs, 0)
            self.assertEqual(stats.zero_delta_modified_pairs, 1)
            self.assertEqual(stats.delta_zero_summands, 2)
            self.assertEqual(
                stats.reason_counts["antisymmetric_palindrome_chain"],
                2,
            )

    def test_filter_and_verifier_use_the_same_scoped_assumptions(self) -> None:
        sqed = OnShellAssumptions(
            massless_momenta=frozenset({2, 3}),
            transverse_field_strengths=frozenset({2, 3}),
        )
        source_rows = [
            {
                "id": "massive-left",
                "simple": "p_1 · p_1",
                "scrambled": "left",
                "note": "retain",
            },
            {
                "id": "massless",
                "simple": "p_2 · p_2",
                "scrambled": "photon",
                "note": "drop",
            },
            {
                "id": "massive-right",
                "simple": "p_4 · p_4",
                "scrambled": "right",
                "note": "retain",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sqed.csv.gz"
            cleaned = root / "sqed-clean.csv.gz"
            self._write_rows(source, source_rows)
            filter_csv(
                source,
                cleaned,
                assumptions=sqed,
                progress_every=0,
            )

            stats = verify_filtered_csv(
                source,
                cleaned,
                assumptions=sqed,
                progress_every=0,
            )
            self.assertEqual(stats.source_rows, 3)
            self.assertEqual(stats.cleaned_rows, 2)
            self.assertEqual(stats.dropped_zero_rows, 1)
            self.assertEqual(stats.zero_differences, 3)

            with self.assertRaises(VerificationError):
                verify_filtered_csv(
                    source,
                    cleaned,
                    assumptions=None,
                    progress_every=0,
                )

    def test_broader_verifier_assumptions_reject_unfiltered_rows(self) -> None:
        sqed = OnShellAssumptions(
            massless_momenta=frozenset({2, 3}),
            transverse_field_strengths=frozenset({2, 3}),
        )
        source_rows = [
            {
                "id": "massive-left",
                "simple": "p_1 · p_1",
                "scrambled": "left",
                "note": "retain",
            },
            {
                "id": "massless",
                "simple": "p_2 · p_2",
                "scrambled": "photon",
                "note": "would drop if scoped",
            },
            {
                "id": "massive-right",
                "simple": "p_4 · p_4",
                "scrambled": "right",
                "note": "retain",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv.gz"
            cleaned = root / "antisymmetry-only.csv.gz"
            self._write_rows(source, source_rows)
            filter_csv(source, cleaned, progress_every=0)

            with self.assertRaises(VerificationError):
                verify_filtered_csv(
                    source,
                    cleaned,
                    assumptions=sqed,
                    progress_every=0,
                )

    def test_verifier_json_records_the_exact_assumptions(self) -> None:
        assumptions = OnShellAssumptions(
            massless_momenta=frozenset({2, 3}),
            transverse_field_strengths=frozenset({2, 3}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv.gz"
            cleaned = root / "cleaned.csv.gz"
            report_path = root / "verification.report.json"
            source_rows = [
                {
                    "id": "massive",
                    "simple": "p_1 · p_1",
                    "scrambled": "same",
                    "note": "retained",
                },
                {
                    "id": "massless",
                    "simple": "p_2 · p_2",
                    "scrambled": "zero",
                    "note": "dropped",
                },
            ]
            self._write_rows(source, source_rows)
            filter_csv(
                source,
                cleaned,
                assumptions=assumptions,
                progress_every=0,
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = verify_main(
                    [
                        str(source),
                        str(cleaned),
                        "--massless-labels",
                        "3,2",
                        "--transverse-field-labels",
                        "2,3",
                        "--report-json",
                        str(report_path),
                        "--progress-every",
                        "0",
                    ]
                )
            self.assertEqual(status, 0)
            stdout_report = json.loads(stdout.getvalue())
            file_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(stdout_report, file_report)
            self.assertEqual(stdout_report["report_schema_version"], 1)
            self.assertEqual(
                stdout_report["on_shell_assumptions"],
                {
                    "massless_momenta": [2, 3],
                    "transverse_field_strengths": [2, 3],
                },
            )


if __name__ == "__main__":
    unittest.main()
