"""Focused tests for the verified 5PT SQED release workflow."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from . import gen_data
from . import split_sqed_5pt_release as release
from . import verify_sqed_5pt_release as verifier


def _pair(index: int) -> tuple[str, str]:
    denominator = "((p_1 · p_2)*(p_2 · p_5))"
    simple = f"{index}*(p_1 · F_2 · p_5)/{denominator}"
    scrambled = (
        f"{index}*(p_1 · p_2*e_2 · p_5 - e_2 · p_1*p_2 · p_5)"
        f"/{denominator}"
    )
    return simple, scrambled


def _write_fixture(
    raw_path: Path,
    token_path: Path,
    rows: list[tuple[str, str]],
) -> None:
    tokenizer = release.ScatteringAmplitudeTokenizer(
        max_particles=8,
        max_sequence_length=None,
    )
    with raw_path.open("w", newline="", encoding="utf-8") as raw_handle, (
        token_path.open("w", newline="", encoding="utf-8")
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
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader)) != release.CSV_HEADER:
            raise AssertionError("bad output header")
        return [(row[0], row[1]) for row in reader]


class SplitSQED5PTReleaseTests(unittest.TestCase):
    def test_profile_is_pinned_for_five_particles(self) -> None:
        self.assertEqual(release.DEFAULT_PROFILE["n_particles"], 5)
        self.assertEqual(
            release.SQED_5PT_ASSUMPTIONS.massless_momenta,
            frozenset({2, 3, 4}),
        )
        cover = release.DEFAULT_PROFILE["profile"]["sqed_cover"]
        self.assertEqual((cover["min_terms"], cover["max_terms"]), (3, 3))
        self.assertEqual(cover["old_style_probability"], 0.0)

    def test_writes_exact_target_disjoint_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "candidate.csv"
            token_path = directory / "candidate_tok.csv"
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
                    "--raw", str(raw_path),
                    "--tokenised", str(token_path),
                    "--generation-log", str(directory / "missing.log"),
                    "--output-dir", str(directory / "release"),
                    "--broad-candidate-rows", "6",
                    "--sqed-cover-candidate-rows", "4",
                    "--hard-candidate-rows", "3",
                    "--broad-release-rows", "5",
                    "--sqed-cover-release-rows", "3",
                    "--hard-release-rows", "2",
                    "--broad-test-rows", "2",
                    "--sqed-cover-test-rows", "1",
                    "--hard-test-rows", "1",
                    "--progress-every", "0",
                ]
            )
            manifest = release.build_release(args)
            paths = release.output_paths(
                args.output_dir,
                release_rows=10,
                test_rows=4,
            )

            train_rows = _read_rows(paths.train_raw)
            test_rows = _read_rows(paths.test_raw)
            self.assertEqual((len(train_rows), len(test_rows)), (6, 4))
            self.assertEqual(len(set(train_rows + test_rows)), 10)
            self.assertTrue(
                {simple for simple, _ in train_rows}.isdisjoint(
                    {simple for simple, _ in test_rows}
                )
            )
            self.assertEqual(manifest["dataset"]["n_particles"], 5)
            self.assertEqual(
                manifest["verification"]["train_test_target_overlap"],
                0,
            )
            self.assertEqual(
                manifest["generation"]["source_artifacts"]["raw"]["rows"],
                13,
            )
            self.assertEqual(
                manifest["generation"]["source_artifacts"]["raw"]["sha256"],
                release._sha256_file(raw_path),
            )
            self.assertIn("sqed_5pt", paths.train_raw.name)

            verify_args = verifier.build_parser().parse_args(
                [
                    "--manifest", str(paths.manifest),
                    "--candidate-raw", str(raw_path),
                    "--candidate-tokenised", str(token_path),
                    "--release-dir", str(args.output_dir),
                    "--full-zero-audit",
                    "--numeric-samples", "0",
                    "--progress-every", "0",
                ]
            )
            report = verifier.verify(verify_args)
            self.assertEqual(report["status"], "verified")

    def test_entire_ambiguous_input_group_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "candidate.csv"
            token_path = directory / "candidate_tok.csv"
            first_simple, shared_scrambled = _pair(1)
            rows = [
                (first_simple, shared_scrambled),
                (_pair(99)[0], shared_scrambled),
                _pair(2),
                _pair(3),
                _pair(4),
                _pair(5),
                _pair(6),
                _pair(7),
            ]
            _write_fixture(raw_path, token_path, rows)
            args = release.build_parser().parse_args(
                [
                    "--raw", str(raw_path),
                    "--tokenised", str(token_path),
                    "--generation-log", str(directory / "missing.log"),
                    "--output-dir", str(directory / "release"),
                    "--broad-candidate-rows", "4",
                    "--sqed-cover-candidate-rows", "2",
                    "--hard-candidate-rows", "2",
                    "--broad-release-rows", "2",
                    "--sqed-cover-release-rows", "2",
                    "--hard-release-rows", "2",
                    "--broad-test-rows", "1",
                    "--sqed-cover-test-rows", "1",
                    "--hard-test-rows", "1",
                    "--progress-every", "0",
                ]
            )
            manifest = release.build_release(args)
            verification = manifest["verification"]
            self.assertEqual(
                verification["full_candidate_ambiguous_scrambled_input_groups"],
                1,
            )
            self.assertEqual(
                verification["ambiguous_scrambled_inputs_removed_by_stratum"],
                {"broad": 2, "sqed_cover": 0, "hard": 0},
            )

    def test_rejects_ignored_characters_and_out_of_scope_legs(self) -> None:
        tokenizer = release.ScatteringAmplitudeTokenizer(
            max_particles=8,
            max_sequence_length=None,
        )
        for expression in ("p_1 · p_2 @", "p_1 · F_6 · p_5"):
            with self.assertRaises(ValueError):
                release._validate_5pt_lexical_scope(
                    expression,
                    tokenizer=tokenizer,
                    row_index=0,
                    column="simple",
                )

    def test_production_stratum_seed_domains_are_disjoint(self) -> None:
        stride = 1_000_003
        broad = {42 + stride * index for index in range(198)}
        cover = {
            42 + 1_000_000_007 + stride * index
            for index in range(99)
        }
        hard = {
            42 + 2_000_000_033 + stride * index
            for index in range(33)
        }
        self.assertTrue(broad.isdisjoint(cover))
        self.assertTrue(broad.isdisjoint(hard))
        self.assertTrue(cover.isdisjoint(hard))

    def test_five_point_middle_field_zero_is_rejected(self) -> None:
        expression = "p_4 · F_3 · F_4 · F_2 · p_4"
        analysis = release.analyze_simple_expression(
            expression,
            assumptions=release.SQED_5PT_ASSUMPTIONS,
        )
        self.assertEqual(analysis.classification, "all_zero")
        self.assertIn(
            "field_strength_split_chain",
            {
                reason.code
                for summand in analysis.zero_summands
                for reason in summand.reasons
            },
        )

    def test_generated_five_point_targets_are_manifestly_clean(self) -> None:
        gen_data.random.seed(50_205)
        generated: list[str] = []
        for _ in range(150):
            built = gen_data._build_base_expression(
                5,
                unit_probability=0.9,
                old_style_probability=0.0,
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
                gen_data._is_manifestly_clean_sqed_target(expression, 5)
                for expression in generated
            )
        )


if __name__ == "__main__":
    unittest.main()
