from __future__ import annotations

import csv
import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import data_gen.filter_antisymmetry_zeros as filter_module
from data_gen.filter_antisymmetry_zeros import (
    ExpressionSyntaxError,
    OnShellAssumptions,
    analyze_simple_expression,
    filter_csv,
    main,
    zero_factor_reasons,
)


def reason_codes(
    factor: str,
    *,
    assumptions: OnShellAssumptions | None = None,
) -> set[str]:
    return {
        reason.code
        for reason in zero_factor_reasons(
            factor,
            assumptions=assumptions,
        )
    }


class AntisymmetryRuleTests(unittest.TestCase):
    def test_module_derivation_uses_lorentzian_skew_adjointness(self) -> None:
        documentation = filter_module.__doc__ or ""
        self.assertIn("F_a^T eta = -eta F_a", documentation)
        self.assertIn(
            "p · W · q = (-1)^n q · F_{an} ... F_{a2} F_{a1} · p",
            documentation,
        )
        self.assertNotIn("``F_a^T = -F_a``", documentation)

    def test_open_chain_requires_same_endpoints_and_odd_palindrome(self) -> None:
        positives = [
            "p_1 · F_4 · p_1",
            "p_1 · F_2 · F_3 · F_2 · p_1",
            "p_12.F_2.F_3.F_4.F_3.F_2.p_12",
        ]
        for factor in positives:
            with self.subTest(factor=factor):
                self.assertIn(
                    "antisymmetric_palindrome_chain",
                    reason_codes(factor),
                )

        negatives = [
            "p_1 · F_2 · F_3 · F_4 · p_1",
            "p_1 · F_2 · F_3 · F_3 · F_2 · p_1",
            "p_1 · F_2 · F_3 · F_2 · p_4",
        ]
        for factor in negatives:
            with self.subTest(factor=factor):
                self.assertNotIn(
                    "antisymmetric_palindrome_chain",
                    reason_codes(factor),
                )

    def test_trace_requires_odd_cyclic_reflection_symmetry(self) -> None:
        positives = [
            "Tr(F_2)",
            "Tr(F_2 · F_3 · F_2)",
            "Tr(F_2 · F_3 · F_2 · F_1 · F_1)",
        ]
        for factor in positives:
            with self.subTest(factor=factor):
                self.assertIn(
                    "antisymmetric_cyclic_trace",
                    reason_codes(factor),
                )

        negatives = [
            "Tr(F_1 · F_2 · F_3)",
            "Tr(F_1 · F_2 · F_2 · F_1)",
        ]
        for factor in negatives:
            with self.subTest(factor=factor):
                self.assertNotIn(
                    "antisymmetric_cyclic_trace",
                    reason_codes(factor),
                )

    def test_ym_onshell_rules_are_opt_in_and_boundary_local(self) -> None:
        left_boundary = "p_3 · F_3 · F_2 · p_4"
        right_boundary = "p_5 · F_1 · F_3 · p_3"
        interior_match = "p_3 · F_1 · F_3 · F_2 · p_5"
        ym5 = OnShellAssumptions.all_massless_ym(5)

        self.assertFalse(reason_codes(left_boundary))
        self.assertFalse(reason_codes(right_boundary))
        self.assertIn(
            "ym_left_self_contraction",
            reason_codes(left_boundary, assumptions=ym5),
        )
        self.assertIn(
            "ym_right_self_contraction",
            reason_codes(right_boundary, assumptions=ym5),
        )
        self.assertFalse(reason_codes(interior_match, assumptions=ym5))
        self.assertIn(
            "ym_massless_momentum_square",
            reason_codes("p_4 · p_4", assumptions=ym5),
        )
        self.assertIn(
            "ym_nilpotent_field_cube",
            reason_codes(
                "p_1 · F_2 · F_2 · F_2 · p_4",
                assumptions=ym5,
            ),
        )

    def test_all_massless_constructor_scopes_exactly_labels_one_through_n(
        self,
    ) -> None:
        ym4 = OnShellAssumptions.all_massless_ym(4)
        expected = frozenset({1, 2, 3, 4})
        self.assertEqual(ym4.massless_momenta, expected)
        self.assertEqual(ym4.transverse_field_strengths, expected)
        self.assertFalse(reason_codes("p_5 · p_5", assumptions=ym4))

    def test_sqed_like_assumptions_leave_massive_endpoints_untouched(
        self,
    ) -> None:
        sqed = OnShellAssumptions(
            massless_momenta=frozenset({2, 3}),
            transverse_field_strengths=frozenset({2, 3}),
        )

        for factor in ("p_1 · p_1", "p_4 · p_4"):
            with self.subTest(factor=factor):
                self.assertNotIn(
                    "ym_massless_momentum_square",
                    reason_codes(factor, assumptions=sqed),
                )
        self.assertIn(
            "ym_massless_momentum_square",
            reason_codes("p_2 · p_2", assumptions=sqed),
        )
        self.assertIn(
            "ym_left_self_contraction",
            reason_codes("p_2 · F_2 · p_1", assumptions=sqed),
        )
        self.assertIn(
            "ym_right_self_contraction",
            reason_codes("p_1 · F_2 · p_2", assumptions=sqed),
        )

        same_endpoint_codes = reason_codes(
            "p_1 · F_2 · p_1",
            assumptions=sqed,
        )
        self.assertIn(
            "antisymmetric_palindrome_chain",
            same_endpoint_codes,
        )
        self.assertFalse(
            any(code.startswith("ym_") for code in same_endpoint_codes)
        )
        self.assertIn(
            "ym_nilpotent_field_cube",
            reason_codes(
                "p_1 · F_2 · F_2 · F_2 · p_4",
                assumptions=sqed,
            ),
        )
        self.assertNotIn(
            "ym_nilpotent_field_cube",
            reason_codes(
                "p_2 · F_1 · F_1 · F_1 · p_3",
                assumptions=sqed,
            ),
        )

    def test_field_strength_rules_require_massless_and_transverse_labels(
        self,
    ) -> None:
        massless_only = OnShellAssumptions(
            massless_momenta=frozenset({2}),
            transverse_field_strengths=frozenset(),
        )
        transverse_only = OnShellAssumptions(
            massless_momenta=frozenset(),
            transverse_field_strengths=frozenset({2}),
        )

        self.assertIn(
            "ym_massless_momentum_square",
            reason_codes("p_2 · p_2", assumptions=massless_only),
        )
        for assumptions in (massless_only, transverse_only):
            with self.subTest(assumptions=assumptions):
                self.assertNotIn(
                    "ym_left_self_contraction",
                    reason_codes(
                        "p_2 · F_2 · p_1",
                        assumptions=assumptions,
                    ),
                )
                self.assertNotIn(
                    "ym_nilpotent_field_cube",
                    reason_codes(
                        "p_1 · F_2 · F_2 · F_2 · p_3",
                        assumptions=assumptions,
                    ),
                )

    def test_cyclic_field_cube_is_scoped_to_the_repeated_label(self) -> None:
        label_two = OnShellAssumptions(
            massless_momenta=frozenset({2}),
            transverse_field_strengths=frozenset({2}),
        )
        for too_short in ("Tr(F_2)", "Tr(F_2 · F_2)"):
            with self.subTest(too_short=too_short):
                self.assertNotIn(
                    "ym_nilpotent_field_cube",
                    reason_codes(too_short, assumptions=label_two),
                )
        self.assertIn(
            "ym_nilpotent_field_cube",
            reason_codes(
                "Tr(F_2 · F_3 · F_2 · F_2)",
                assumptions=label_two,
            ),
        )
        self.assertNotIn(
            "ym_nilpotent_field_cube",
            reason_codes(
                "Tr(F_1 · F_1 · F_1 · F_2)",
                assumptions=label_two,
            ),
        )

    def test_none_and_empty_assumptions_have_identical_behavior(self) -> None:
        empty = OnShellAssumptions(frozenset(), frozenset())
        factors = (
            "p_1 · p_1",
            "p_1 · F_1 · p_2",
            "p_1 · F_2 · p_1",
            "Tr(F_1 · F_2 · F_1)",
        )
        for factor in factors:
            with self.subTest(factor=factor):
                self.assertEqual(
                    reason_codes(factor),
                    reason_codes(factor, assumptions=empty),
                )

    def test_assumptions_reject_nonpositive_and_noninteger_labels(self) -> None:
        invalid_sets = (
            frozenset({0}),
            frozenset({-1}),
            frozenset({True}),
            frozenset({"2"}),
        )
        for labels in invalid_sets:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    OnShellAssumptions(labels, frozenset())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            OnShellAssumptions.all_massless_ym(0)


class ExpressionPruningTests(unittest.TestCase):
    def test_prunes_only_zero_top_level_summands_and_preserves_signs(self) -> None:
        zero = "p_3 · F_4 · p_3"
        cases = {
            f"p_1 · p_2 + {zero} + p_2 · p_4":
                "p_1 · p_2 + p_2 · p_4",
            f"{zero} - p_2 · p_4":
                "-p_2 · p_4",
            f"-{zero} + p_2 · p_4":
                "p_2 · p_4",
            f"p_2 · p_4 - {zero}":
                "p_2 · p_4",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                analysis = analyze_simple_expression(expression)
                self.assertEqual(analysis.classification, "mixed")
                self.assertEqual(analysis.cleaned, expected)

    def test_drops_expression_when_every_summand_is_zero(self) -> None:
        expression = (
            "p_1 · F_4 · p_1 "
            "+ Tr(F_2 · F_3 · F_2)"
        )
        analysis = analyze_simple_expression(expression)
        self.assertEqual(analysis.classification, "all_zero")
        self.assertIsNone(analysis.cleaned)
        self.assertEqual(len(analysis.zero_summands), 2)

    def test_coefficient_power_and_rational_wrapping_are_supported(self) -> None:
        expression = (
            "-73*(Tr(F_1 · F_2 · F_1)*p_3 · p_4)"
            "/(p_1 · p_2*p_2 · p_3)"
        )
        analysis = analyze_simple_expression(expression)
        self.assertEqual(analysis.classification, "all_zero")

        powered = analyze_simple_expression(
            "(p_1 · F_4 · p_1)^2*p_2 · p_3"
        )
        self.assertEqual(powered.classification, "all_zero")

    def test_nested_sum_and_denominator_are_not_scanned_as_factors(self) -> None:
        expression = (
            "p_1 · p_2/(p_3 · p_4 - p_4 · p_5) "
            "+ p_6 · F_7 · p_6"
        )
        analysis = analyze_simple_expression(expression)
        self.assertEqual(
            analysis.cleaned,
            "p_1 · p_2/(p_3 · p_4 - p_4 · p_5)",
        )

        denominator_only = analyze_simple_expression(
            "p_1 · p_2/(p_3 · F_4 · p_3)"
        )
        self.assertEqual(denominator_only.classification, "clean")

    def test_malformed_expression_fails_closed(self) -> None:
        with self.assertRaises(ExpressionSyntaxError):
            analyze_simple_expression("(p_1 · F_2 · p_1")

    def test_wrapped_unary_sign_and_spaced_unary_factor_are_safe(self) -> None:
        wrapped = analyze_simple_expression("(-(p_1 · F_2 · p_1))")
        self.assertEqual(wrapped.classification, "all_zero")

        spaced_unary = analyze_simple_expression(
            "p_1 · p_2 * - p_3 · F_4 · p_3"
        )
        self.assertEqual(spaced_unary.classification, "clean")

    def test_scientific_notation_sign_is_not_split_as_a_summand(self) -> None:
        negative_exponent = analyze_simple_expression(
            "1e-3*p_1 · F_2 · p_1 + p_3 · p_4"
        )
        self.assertEqual(negative_exponent.cleaned, "p_3 · p_4")

        positive_exponent = analyze_simple_expression(
            "1E+3*p_1 · F_2 · p_1 - p_3 · p_4"
        )
        self.assertEqual(positive_exponent.cleaned, "-p_3 · p_4")


class CsvFilterTests(unittest.TestCase):
    def _write_fixture(self, path: Path) -> None:
        fieldnames = ["id", "scrambled", "simple", "note"]
        rows = [
            {
                "id": "all-zero",
                "scrambled": "expanded-zero",
                "simple": "p_1 · F_4 · p_1",
                "note": "drop me",
            },
            {
                "id": "mixed",
                "scrambled": "x" * 200_000,
                "simple": (
                    "p_1 · p_2 "
                    "- Tr(F_2 · F_3 · F_2)"
                ),
                "note": "comma, and\nnewline",
            },
            {
                "id": "clean",
                "scrambled": "unchanged",
                "simple": "p_1 · F_2 · F_3 · F_4 · p_1",
                "note": "odd but not palindromic",
            },
            {
                "id": "ym-boundary",
                "scrambled": "also unchanged",
                "simple": "p_3 · F_3 · F_2 · p_4",
                "note": "on-shell only",
            },
        ]
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_streaming_gzip_filter_preserves_schema_and_other_columns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="antisymmetry test ") as temp_dir:
            root = Path(temp_dir)
            source = root / "source data.csv.gz"
            output = root / "clean data.csv.gz"
            self._write_fixture(source)

            stats, _examples = filter_csv(
                source,
                output,
                progress_every=0,
            )
            self.assertEqual(stats.input_rows, 4)
            self.assertEqual(stats.output_rows, 3)
            self.assertEqual(stats.rows_dropped, 1)
            self.assertEqual(stats.rows_modified, 1)
            self.assertEqual(stats.zero_summands, 2)

            self.assertEqual(output.read_bytes()[:2], b"\x1f\x8b")
            with gzip.open(output, "rt", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    ["id", "scrambled", "simple", "note"],
                )
                rows = list(reader)

            self.assertEqual([row["id"] for row in rows], ["mixed", "clean", "ym-boundary"])
            mixed = rows[0]
            self.assertEqual(mixed["simple"], "p_1 · p_2")
            self.assertEqual(mixed["scrambled"], "x" * 200_000)
            self.assertEqual(mixed["note"], "comma, and\nnewline")

    def test_ym_mode_drops_boundary_only_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv.gz"
            output = root / "clean.csv.gz"
            self._write_fixture(source)
            stats, _examples = filter_csv(
                source,
                output,
                assumptions=OnShellAssumptions.all_massless_ym(5),
                progress_every=0,
            )
            self.assertEqual(stats.output_rows, 2)
            self.assertEqual(stats.rows_dropped, 2)

    def test_sqed_scoping_retains_massive_endpoint_rows_end_to_end(self) -> None:
        assumptions = OnShellAssumptions(
            massless_momenta=frozenset({2, 3}),
            transverse_field_strengths=frozenset({2, 3}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sqed.csv.gz"
            output = root / "sqed-clean.csv.gz"
            with gzip.open(
                source,
                "wt",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["simple", "scrambled"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"simple": "p_1 · p_1", "scrambled": "massive-left"},
                        {"simple": "p_2 · p_2", "scrambled": "massless"},
                        {"simple": "p_4 · p_4", "scrambled": "massive-right"},
                    ]
                )

            stats, _examples = filter_csv(
                source,
                output,
                assumptions=assumptions,
                progress_every=0,
            )
            self.assertEqual(stats.input_rows, 3)
            self.assertEqual(stats.output_rows, 2)
            self.assertEqual(stats.rows_dropped, 1)
            with gzip.open(
                output,
                "rt",
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["simple"] for row in rows],
                ["p_1 · p_1", "p_4 · p_4"],
            )
            self.assertEqual(
                [row["scrambled"] for row in rows],
                ["massive-left", "massive-right"],
            )

    def test_json_report_records_sorted_explicit_assumptions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv.gz"
            report_path = root / "filter.report.json"
            self._write_fixture(source)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        str(source),
                        "--dry-run",
                        "--massless-labels",
                        "3,2,2",
                        "--transverse-field-labels",
                        "2,3",
                        "--report-json",
                        str(report_path),
                        "--max-examples",
                        "0",
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

    def test_legacy_ym_cli_requires_explicit_particle_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv.gz"
            output = root / "output.csv.gz"
            self._write_fixture(source)

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            str(source),
                            str(output),
                            "--include-ym-onshell",
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_legacy_ym_cli_with_count_matches_preferred_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.csv.gz"
            self._write_fixture(source)

            reports: list[dict[str, object]] = []
            legacy_stderr = io.StringIO()
            for options, stderr in (
                (
                    ["--include-ym-onshell", "--n-particles", "4"],
                    legacy_stderr,
                ),
                (["--all-massless-ym", "4"], io.StringIO()),
            ):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    with contextlib.redirect_stderr(stderr):
                        status = main(
                            [
                                str(source),
                                "--dry-run",
                                "--max-examples",
                                "0",
                                "--progress-every",
                                "0",
                                *options,
                            ]
                        )
                self.assertEqual(status, 0)
                reports.append(json.loads(stdout.getvalue()))

            self.assertIn("deprecated", legacy_stderr.getvalue())
            self.assertEqual(
                reports[0]["on_shell_assumptions"],
                reports[1]["on_shell_assumptions"],
            )
            self.assertEqual(reports[0]["stats"], reports[1]["stats"])

    def test_cli_rejects_ambiguous_or_invalid_assumption_options(self) -> None:
        invalid_options = (
            ["--n-particles", "4"],
            ["--all-massless-ym", "0"],
            [
                "--all-massless-ym",
                "4",
                "--massless-labels",
                "2,3",
            ],
            [
                "--include-ym-onshell",
                "--n-particles",
                "4",
                "--transverse-field-labels",
                "2,3",
            ],
            ["--massless-labels", "2,,3"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv.gz"
            self._write_fixture(source)
            for case_number, options in enumerate(invalid_options):
                output = root / f"invalid-{case_number}.csv.gz"
                with self.subTest(options=options):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            main([str(source), str(output), *options])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertFalse(output.exists())

    def test_missing_simple_header_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "bad.csv"
            output = root / "should-not-exist.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["scrambled"])
                writer.writerow(["x"])
            with self.assertRaisesRegex(ValueError, "'simple'"):
                filter_csv(source, output, progress_every=0)
            self.assertFalse(output.exists())

    def test_report_cannot_overwrite_input_or_csv_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv.gz"
            output = root / "output.csv.gz"
            self._write_fixture(source)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stderr(stderr):
                    status = main([
                        str(source),
                        "--dry-run",
                        "--report-json",
                        str(source),
                        "--overwrite",
                    ])
            self.assertEqual(status, 1)
            self.assertEqual(source.read_bytes()[:2], b"\x1f\x8b")

            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stderr(stderr):
                    status = main([
                        str(source),
                        str(output),
                        "--report-json",
                        str(output),
                    ])
            self.assertEqual(status, 1)
            self.assertFalse(output.exists())

    def test_no_overwrite_mode_does_not_clobber_racing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv.gz"
            output = root / "output.csv.gz"
            self._write_fixture(source)

            original_analyze = filter_module.analyze_simple_expression

            def create_racing_output(expression: str, **kwargs):
                if not output.exists():
                    output.write_text("created by another process", encoding="utf-8")
                return original_analyze(expression, **kwargs)

            with mock.patch.object(
                filter_module,
                "analyze_simple_expression",
                side_effect=create_racing_output,
            ):
                with self.assertRaisesRegex(
                    FileExistsError,
                    "appeared during processing",
                ):
                    filter_csv(source, output, progress_every=0)

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "created by another process",
            )


if __name__ == "__main__":
    unittest.main()
