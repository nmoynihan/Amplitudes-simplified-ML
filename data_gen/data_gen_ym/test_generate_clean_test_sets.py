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
                Path(generation_args.raw_out).write_text(
                    f"raw-{n_particles}\n",
                    encoding="utf-8",
                )
                Path(generation_args.tok_out).write_text(
                    f"tok-{n_particles}\n",
                    encoding="utf-8",
                )
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
            raw4, _token4, _report4 = launcher.output_paths(
                directory,
                n_particles=4,
                samples=7,
                seed=4007,
            )
            self.assertEqual(
                manifest["sets"]["4pt"]["raw_output"]["path"],
                str(raw4.absolute()),
            )
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                manifest,
            )

    def test_later_generation_failure_leaves_no_partial_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = launcher.build_parser().parse_args(
                [
                    "--samples",
                    "2",
                    "--output-dir",
                    str(directory),
                    "--jobs",
                    "1",
                    "--progress-every",
                    "0",
                ]
            )

            def fail_after_four_point(
                generation_args,
                *,
                n_particles: int,
                generator_name: str,
            ):
                if n_particles == 5:
                    raise RuntimeError("injected 5PT failure")
                Path(generation_args.raw_out).write_text("new raw\n", encoding="utf-8")
                Path(generation_args.tok_out).write_text("new tok\n", encoding="utf-8")
                stats = core.GenerationStats(requested=2, accepted=2)
                return stats, {
                    "outputs": {
                        "raw": {
                            "path": generation_args.raw_out,
                            "rows": 2,
                            "sha256_uncompressed": "raw-4",
                        },
                        "tokenized": {
                            "path": generation_args.tok_out,
                            "rows": 2,
                            "sha256_uncompressed": "tok-4",
                        },
                    },
                    "stats": {"requested": 2, "accepted": 2},
                }

            with mock.patch.object(
                core,
                "generate_to_files",
                side_effect=fail_after_four_point,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected 5PT failure"):
                    launcher.generate_test_sets(args)

            self.assertEqual(list(directory.iterdir()), [])

    def test_symlink_overwrite_manifest_names_published_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = launcher.build_parser().parse_args(
                [
                    "--samples",
                    "1",
                    "--only",
                    "4pt",
                    "--output-dir",
                    str(directory),
                    "--jobs",
                    "1",
                    "--progress-every",
                    "0",
                    "--overwrite",
                ]
            )
            raw_path, _token_path, report_path = launcher.output_paths(
                directory,
                n_particles=4,
                samples=1,
                seed=args.seed_4pt,
            )
            old_raw_target = directory / "old-heldout-raw"
            old_report_target = directory / "old-heldout-report"
            old_raw_target.write_text("old raw\n", encoding="utf-8")
            old_report_target.write_text("old report\n", encoding="utf-8")
            raw_path.symlink_to(old_raw_target.name)
            report_path.symlink_to(old_report_target.name)

            def fake_generate(
                generation_args,
                *,
                n_particles: int,
                generator_name: str,
            ):
                Path(generation_args.raw_out).write_text("new raw\n", encoding="utf-8")
                Path(generation_args.tok_out).write_text("new tok\n", encoding="utf-8")
                stats = core.GenerationStats(requested=1, accepted=1)
                return stats, {
                    "outputs": {
                        "raw": {
                            "path": generation_args.raw_out,
                            "rows": 1,
                            "sha256_uncompressed": "raw-4",
                        },
                        "tokenized": {
                            "path": generation_args.tok_out,
                            "rows": 1,
                            "sha256_uncompressed": "tok-4",
                        },
                    },
                    "stats": {"requested": 1, "accepted": 1},
                }

            with mock.patch.object(
                core,
                "generate_to_files",
                side_effect=fake_generate,
            ):
                manifest, _manifest_path = launcher.generate_test_sets(args)

            entry = manifest["sets"]["4pt"]
            self.assertFalse(raw_path.is_symlink())
            self.assertFalse(report_path.is_symlink())
            self.assertEqual(entry["raw_output"]["path"], str(raw_path.absolute()))
            self.assertEqual(entry["report_path"], str(report_path.absolute()))
            persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted_report["outputs"]["raw"]["path"],
                str(raw_path.absolute()),
            )
            self.assertEqual(old_raw_target.read_text(encoding="utf-8"), "old raw\n")
            self.assertEqual(
                old_report_target.read_text(encoding="utf-8"),
                "old report\n",
            )

    def test_release_failure_restores_all_existing_heldout_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            args = launcher.build_parser().parse_args(
                [
                    "--samples",
                    "2",
                    "--output-dir",
                    str(directory),
                    "--jobs",
                    "1",
                    "--progress-every",
                    "0",
                    "--overwrite",
                ]
            )
            paths_by_point = {
                n_particles: launcher.output_paths(
                    directory,
                    n_particles=n_particles,
                    samples=2,
                    seed=(args.seed_4pt if n_particles == 4 else args.seed_5pt),
                )
                for n_particles in (4, 5)
            }
            manifest_path = launcher.default_manifest_path(
                directory,
                samples=2,
                seed_4pt=args.seed_4pt,
                seed_5pt=args.seed_5pt,
            )
            destinations = [
                path
                for point_paths in paths_by_point.values()
                for path in point_paths
            ] + [manifest_path]
            expected: dict[Path, str] = {}
            for index, path in enumerate(destinations):
                content = f"old heldout output {index}\n"
                path.write_text(content, encoding="utf-8")
                expected[path] = content

            def fake_generate(
                generation_args,
                *,
                n_particles: int,
                generator_name: str,
            ):
                Path(generation_args.raw_out).write_text("new raw\n", encoding="utf-8")
                Path(generation_args.tok_out).write_text("new tok\n", encoding="utf-8")
                stats = core.GenerationStats(requested=2, accepted=2)
                return stats, {
                    "outputs": {
                        "raw": {
                            "path": generation_args.raw_out,
                            "rows": 2,
                            "sha256_uncompressed": f"raw-{n_particles}",
                        },
                        "tokenized": {
                            "path": generation_args.tok_out,
                            "rows": 2,
                            "sha256_uncompressed": f"tok-{n_particles}",
                        },
                    },
                    "stats": {"requested": 2, "accepted": 2},
                }

            real_publish_all = core._publish_all

            def fail_during_release(temporary_paths, final_paths, *, overwrite):
                calls = 0
                real_publish = core._publish

                def fail_third(temp_path, destination, *, overwrite):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise OSError("injected heldout release failure")
                    return real_publish(temp_path, destination, overwrite=overwrite)

                with mock.patch.object(core, "_publish", side_effect=fail_third):
                    return real_publish_all(
                        temporary_paths,
                        final_paths,
                        overwrite=overwrite,
                    )

            with mock.patch.object(
                core,
                "generate_to_files",
                side_effect=fake_generate,
            ), mock.patch.object(
                core,
                "_publish_all",
                side_effect=fail_during_release,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected heldout release failure",
                ):
                    launcher.generate_test_sets(args)

            for path, content in expected.items():
                self.assertEqual(path.read_text(encoding="utf-8"), content)
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                sorted(path.name for path in destinations),
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
