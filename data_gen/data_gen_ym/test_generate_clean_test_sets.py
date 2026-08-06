"""Tests for the combined corrected Yang--Mills test-set launcher."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import generate_clean_4pt as core
from . import generate_clean_test_sets as launcher


class CleanTestSetLauncherTests(unittest.TestCase):
    def test_defaults_request_two_distinct_200_row_test_sets(self) -> None:
        args = launcher.build_parser().parse_args([])
        self.assertEqual(args.samples, 200)
        self.assertEqual(args.only, "both")
        self.assertNotEqual(args.seed_4pt, args.seed_5pt)
        self.assertEqual(args.max_tokens_4pt, 2048)
        self.assertEqual(args.max_tokens_5pt, 4096)

    def test_output_names_are_role_and_particle_specific(self) -> None:
        directory = Path("/tmp/example")
        raw4, token4, report4 = launcher.output_paths(
            directory,
            n_particles=4,
            samples=200,
            seed=4007,
        )
        raw5, token5, report5 = launcher.output_paths(
            directory,
            n_particles=5,
            samples=200,
            seed=5007,
        )
        self.assertEqual(
            raw4.name,
            "ym_4pt_heldout_200_seed4007_canonical_nonzero.csv.gz",
        )
        self.assertEqual(
            token5.name,
            "ym_5pt_heldout_200_seed5007_canonical_nonzero_tok.csv.gz",
        )
        self.assertNotEqual(raw4, raw5)
        self.assertNotEqual(token4, token5)
        self.assertNotEqual(report4, report5)

    def test_generates_both_points_and_writes_a_combined_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = launcher.build_parser().parse_args(
                [
                    "--samples",
                    "7",
                    "--seed-4pt",
                    "4007",
                    "--seed-5pt",
                    "5007",
                    "--output-dir",
                    str(directory),
                    "--jobs",
                    "1",
                    "--progress-every",
                    "0",
                ]
            )

            calls: list[tuple[int, str, int, int, str, str]] = []

            def fake_generate(
                generation_args,
                *,
                n_particles: int,
                generator_name: str,
            ):
                calls.append(
                    (
                        n_particles,
                        generator_name,
                        generation_args.seed,
                        generation_args.max_tokens,
                        generation_args.raw_out,
                        generation_args.tok_out,
                    )
                )
                stats = core.GenerationStats(
                    requested=generation_args.samples,
                    accepted=generation_args.samples,
                )
                report = {
                    "outputs": {
                        "raw": {
                            "path": generation_args.raw_out,
                            "rows": generation_args.samples,
                            "sha256_uncompressed": f"raw-{n_particles}",
                        },
                        "tokenized": {
                            "path": generation_args.tok_out,
                            "rows": generation_args.samples,
                            "sha256_uncompressed": f"tok-{n_particles}",
                        },
                    },
                    "stats": {
                        "requested": generation_args.samples,
                        "accepted": generation_args.samples,
                    },
                }
                return stats, report

            with mock.patch.object(
                core,
                "generate_to_files",
                side_effect=fake_generate,
            ):
                manifest, manifest_path = launcher.generate_test_sets(args)

            self.assertEqual([call[0] for call in calls], [4, 5])
            self.assertEqual([call[1] for call in calls], [
                "clean_4pt_yang_mills_test",
                "clean_5pt_yang_mills_test",
            ])
            self.assertEqual([call[2] for call in calls], [4007, 5007])
            self.assertEqual([call[3] for call in calls], [2048, 4096])
            self.assertIn("ym_4pt_heldout_7_seed4007", calls[0][4])
            self.assertIn("ym_5pt_heldout_7_seed5007", calls[1][5])

            self.assertEqual(manifest["dataset_role"], "held_out_test")
            self.assertEqual(manifest["samples_per_point"], 7)
            self.assertEqual(manifest["selected_points"], [4, 5])
            self.assertEqual(set(manifest["sets"]), {"4pt", "5pt"})
            self.assertTrue(manifest_path.exists())
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                manifest,
            )

    def test_preflight_refuses_partial_overwrite_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = launcher.build_parser().parse_args(
                ["--output-dir", str(directory)]
            )
            raw4, _token4, _report4 = launcher.output_paths(
                directory,
                n_particles=4,
                samples=200,
                seed=launcher.DEFAULT_4PT_SEED,
            )
            raw4.write_text("sentinel", encoding="utf-8")

            with mock.patch.object(core, "generate_to_files") as generate_mock:
                with self.assertRaises(FileExistsError):
                    launcher.generate_test_sets(args)
            generate_mock.assert_not_called()
            self.assertEqual(raw4.read_text(encoding="utf-8"), "sentinel")


if __name__ == "__main__":
    unittest.main()
