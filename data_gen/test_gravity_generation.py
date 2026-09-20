"""Regression tests for gravity dataset generation and held-out evaluation."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from . import gravity_generation as pipeline
from .data_gen_gravity.core import (
    BENCHMARKS,
    PROCESS_SPECS,
    _relabel,
    is_benchmark_leak,
    numerically_equivalent,
)
from .data_gen_gravity.generate import Candidate


def candidate(index: int = 1, **changes) -> Candidate:
    """Small syntactic fixtures; physical correctness is checked separately."""
    row = Candidate(
        simple=f"{index}*(p_1 · p_2)",
        scrambled=f"{index}*(p_2 · p_3)",
        process="3s2h", kind="oneshot", seed=index,
        scramble_depth=1, scramble_labels="fixture", stage="compact",
        compact_terms=1, simple_tokens=5, scrambled_tokens=5,
        relative_error=0.0, compact_origin="p_1 · p_2",
    )
    return replace(row, **changes)


def read_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class HoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = candidate(9)
        self.guard = pipeline.HoldoutGuard([self.benchmark], max_tokens=4096)

    def test_rejects_exact_token_equivalent_and_opposite_column_overlap(self) -> None:
        cases = (
            replace(self.benchmark),
            candidate(simple=" ( 9 * ( p_1 · p_2 ) ) "),
            candidate(simple=self.benchmark.scrambled),
            candidate(scrambled=self.benchmark.simple),
        )
        for row in cases:
            with self.subTest(row=row):
                self.assertEqual(self.guard.rejection_reason(row), "benchmark_expression")
        self.assertIsNone(self.guard.rejection_reason(candidate(2)))

    def test_rejects_relabelled_benchmark_origins_even_for_staged_rows(self) -> None:
        for process, compact in BENCHMARKS.items():
            spec = PROCESS_SPECS[process]
            mapping = dict(zip(spec.scalar_legs, reversed(spec.scalar_legs)))
            mapping.update(zip(spec.graviton_legs, reversed(spec.graviton_legs)))
            for kind in ("oneshot", "staged"):
                row = candidate(
                    process=process, kind=kind, stage="intermediate",
                    compact_origin=_relabel(compact, mapping),
                )
                with self.subTest(process=process, kind=kind):
                    self.assertNotEqual(row.simple, row.compact_origin)
                    self.assertEqual(self.guard.rejection_reason(row), "benchmark_family")

    def test_missing_origin_and_empty_benchmarks_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "compact origin"):
            self.guard.rejection_reason(candidate(compact_origin=""))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            pipeline.HoldoutGuard([], max_tokens=4096)

    def test_serialized_audit_checks_both_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, test = (Path(directory) / name for name in ("train.csv.gz", "test.csv.gz"))
            pipeline.tokenise([self.benchmark], test)
            for row in (
                candidate(simple=self.benchmark.scrambled),
                candidate(scrambled=self.benchmark.simple),
                candidate(simple=" ( 9 * ( p_1 · p_2 ) ) "),
            ):
                pipeline.tokenise([row], train)
                with self.assertRaisesRegex(ValueError, "expression overlap"):
                    pipeline.audit_token_files(train, test)
            pipeline.tokenise([candidate(2)], train)
            self.assertEqual(pipeline.audit_token_files(train, test), {
                "training_rows": 1, "test_rows": 1, "shared_expressions": 0,
            })

    def test_quota_refill_preserves_all_four_cells(self) -> None:
        cells = [(p, k) for p in PROCESS_SPECS for k in ("oneshot", "staged")]
        initial = [candidate(i + 1, process=p, kind=k) for i, (p, k) in enumerate(cells)]
        initial[0] = candidate(9)
        with patch.object(pipeline, "build_dataset_parallel", side_effect=[initial, [candidate(5)]]) as build:
            rows, rejected = pipeline.build_training_set(
                4, [self.benchmark], jobs=1, seed=12, max_tokens=4096,
            )
        self.assertEqual(Counter((r.process, r.kind) for r in rows), Counter(cells))
        self.assertEqual(rejected, {"benchmark_expression": 1})
        self.assertEqual(build.call_count, 2)
        refill = build.call_args
        self.assertEqual(refill.args, (1,))
        self.assertEqual((refill.kwargs["process"], refill.kwargs["kind"]), cells[0])
        self.assertNotEqual(refill.kwargs["seed"], 12)
        self.assertTrue(all(call.kwargs["validate"] for call in build.call_args_list))

    def test_padding_special_and_unknown_tokens_cannot_bypass_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, test = (Path(directory) / name for name in ("train.csv.gz", "test.csv.gz"))
            pipeline.tokenise([self.benchmark], test)
            shared = self.guard.tokenizer.encode_infix(self.benchmark.simple)
            other = self.guard.tokenizer.encode_infix(candidate(2).scrambled)
            for extra in (0, 1, 2, 3, 9, 10, 999, True):
                with self.subTest(extra=extra):
                    with gzip.open(train, "wt", encoding="utf-8", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(("simple", "scrambled"))
                        writer.writerow((json.dumps(shared + [extra]), json.dumps(other)))
                    with self.assertRaisesRegex(ValueError, "Invalid token list"):
                        pipeline.audit_token_files(train, test)

    def test_refill_attempts_are_bounded(self) -> None:
        with patch.object(pipeline, "build_dataset_parallel", return_value=[self.benchmark]) as build:
            with self.assertRaisesRegex(RuntimeError, "Could not fill training quotas"):
                pipeline.build_training_set(
                    1, [self.benchmark], jobs=1, seed=12, max_tokens=4096,
                    max_refill_rounds=2,
                )
        self.assertEqual(build.call_count, 3)

    def test_audit_rejects_token_cap_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, test = (Path(directory) / name for name in ("train.csv.gz", "test.csv.gz"))
            pipeline.tokenise([self.benchmark], test)
            pipeline.tokenise([candidate(2, simple="2*(p_1 · p_2)+3")], train)
            with self.assertRaisesRegex(ValueError, "Invalid token list"):
                pipeline.audit_token_files(train, test, max_tokens=5)


class OutputAndRoutingTests(unittest.TestCase):
    def parse(self, directory: str, *options: str):
        # The runner accepts environment overrides; tests must not inherit them.
        with patch.dict(os.environ, {}, clear=True):
            return pipeline.parse_args(["--output-dir", directory, *options])

    def test_colliding_and_aliased_paths_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shared = str(Path(directory) / "shared.csv.gz")
            args = self.parse(directory, "--tok-out", shared, "--bench-tok", shared)
            with self.assertRaisesRegex(ValueError, "must differ"):
                pipeline.dataset_paths(args)
            paths = pipeline.dataset_paths(self.parse(directory))
            paths["raw_out"].touch()
            os.link(paths["raw_out"], paths["bench_raw"])
            with self.assertRaisesRegex(ValueError, "alias the same file"):
                pipeline.dataset_paths(self.parse(directory))

    def test_existing_outputs_require_overwrite_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.parse(directory)
            paths = pipeline.dataset_paths(args)
            paths["raw_out"].write_bytes(b"original data")
            with patch.object(pipeline, "build_benchmarks") as build:
                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    pipeline.generate_datasets(args, paths)
                build.assert_not_called()
            self.assertEqual(paths["raw_out"].read_bytes(), b"original data")

    def test_evaluation_collisions_fail_before_any_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("full", "eval"):
                for flag in ("raw-out", "tok-out", "metadata-out", "bench-raw",
                             "bench-tok", "bench-metadata", "manifest-out"):
                    for filename in ("benchmark_evaluation.csv.gz", "benchmark_evaluation_summary.json"):
                        with self.subTest(mode=mode, flag=flag, filename=filename), \
                             patch.dict(os.environ, {}, clear=True), \
                             patch.object(pipeline, "generate_datasets") as generate, \
                             patch.object(pipeline, "verify_saved_datasets") as verify, \
                             patch.object(pipeline.subprocess, "run") as run:
                            with self.assertRaisesRegex(ValueError, "must differ"):
                                pipeline.main([
                                    mode, "--output-dir", directory,
                                    f"--{flag}", str(Path(directory) / filename), "--overwrite",
                                ])
                            generate.assert_not_called()
                            verify.assert_not_called()
                            run.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_evaluation_aliases_cannot_overwrite_existing_inputs(self) -> None:
        for alias_kind in ("symlink", "hardlink"):
            for mode in ("full", "eval"):
                with self.subTest(alias_kind=alias_kind, mode=mode), \
                     tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "benchmarks_raw.csv.gz"
                    source.write_bytes(b"audited benchmark data")
                    output = Path(directory) / "benchmark_evaluation.csv.gz"
                    if alias_kind == "symlink":
                        output.symlink_to(source)
                    else:
                        os.link(source, output)
                    with patch.dict(os.environ, {}, clear=True), \
                         patch.object(pipeline, "generate_datasets") as generate, \
                         patch.object(pipeline.subprocess, "run") as run:
                        with self.assertRaisesRegex(ValueError, "must differ|alias the same file"):
                            pipeline.main([mode, "--output-dir", directory, "--overwrite"])
                        generate.assert_not_called()
                        run.assert_not_called()
                    self.assertEqual(source.read_bytes(), b"audited benchmark data")
                    self.assertEqual(output.read_bytes(), source.read_bytes())

    def test_direct_evaluation_command_rejects_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.parse(directory, "--manifest-out",
                              str(Path(directory) / "benchmark_evaluation_summary.json"))
            # Data-only generation need not reserve unused evaluation paths.
            paths = pipeline.dataset_paths(args)
            with self.assertRaisesRegex(ValueError, "must differ"):
                pipeline.evaluation_command(args, paths)

    def test_directory_destinations_and_ancestor_conflicts_fail_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            existing_directory = base / "existing_directory"
            existing_directory.mkdir()
            cases = (
                (["--raw-out", str(existing_directory)], "not a regular file"),
                (["--raw-out", str(base / "parent"), "--tok-out", str(base / "parent" / "child.csv.gz")],
                 "cannot be the parent"),
            )
            for options, message in cases:
                with self.subTest(options=options), patch.dict(os.environ, {}, clear=True), \
                     patch.object(pipeline, "generate_datasets") as generate:
                    with self.assertRaisesRegex(ValueError, message):
                        pipeline.main(["data", "--output-dir", directory, *options])
                    generate.assert_not_called()

    def test_failed_serialization_preserves_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.parse(directory, "--overwrite", "--samples", "1", "--benchmark-samples", "5")
            paths = pipeline.dataset_paths(args)
            for path in paths.values():
                path.write_bytes(b"original data")
            with patch.object(pipeline, "build_benchmarks", return_value=[candidate(9)]), \
                 patch.object(pipeline, "build_training_set", return_value=([candidate(2)], {})), \
                 patch.object(pipeline, "tokenise", side_effect=ValueError("serialization failed")):
                with self.assertRaisesRegex(ValueError, "serialization failed"):
                    pipeline.generate_datasets(args, paths)
            self.assertTrue(all(path.read_bytes() == b"original data" for path in paths.values()))
            self.assertEqual(set(Path(directory).resolve().iterdir()), set(paths.values()))

    def test_failed_publication_restores_the_complete_previous_release(self) -> None:
        for phase in ("backup", "second_output", "manifest"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                args = self.parse(directory, "--samples", "4", "--benchmark-samples", "5",
                                  "--jobs", "1", "--min-scr", "1", "--max-scr", "1")
                paths = pipeline.dataset_paths(args)
                pipeline.generate_datasets(args, paths)
                saved = {key: path.read_bytes() for key, path in paths.items()}
                args.overwrite = True
                args.seed += 1
                original_replace = os.replace
                injected = []

                def fail_replace(source, destination):
                    source, destination = Path(source), Path(destination)
                    fail_here = (
                        source == paths["tok_out"] if phase == "backup" else
                        destination == paths["manifest_out" if phase == "manifest" else "tok_out"]
                    )
                    if fail_here and not injected:
                        injected.append(True)
                        raise OSError("injected publication failure")
                    return original_replace(source, destination)

                with patch.object(pipeline.os, "replace", side_effect=fail_replace):
                    with self.assertRaisesRegex(OSError, "injected publication failure"):
                        pipeline.generate_datasets(args, paths)
                self.assertTrue(injected)
                self.assertEqual(saved, {key: path.read_bytes() for key, path in paths.items()})
                self.assertEqual(set(Path(directory).resolve().iterdir()), set(paths.values()))
                pipeline.verify_saved_datasets(paths)

    def test_failed_fresh_publication_leaves_no_partial_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.parse(directory, "--samples", "4", "--benchmark-samples", "5",
                              "--jobs", "1", "--min-scr", "1", "--max-scr", "1")
            paths = pipeline.dataset_paths(args)
            original_link = os.link

            def fail_second_output(source, destination):
                if Path(destination) == paths["tok_out"]:
                    raise OSError("injected publication failure")
                return original_link(source, destination)

            with patch.object(pipeline.os, "link", side_effect=fail_second_output):
                with self.assertRaisesRegex(OSError, "injected publication failure"):
                    pipeline.generate_datasets(args, paths)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_saved_datasets_require_a_generation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = pipeline.dataset_paths(self.parse(directory))
            with self.assertRaisesRegex(ValueError, "Missing generation manifest"):
                pipeline.verify_saved_datasets(paths)

    def test_training_and_evaluation_use_only_their_respective_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.parse(directory, "--max-tokens", "800")
            paths = pipeline.dataset_paths(args)
            training = pipeline.training_command(args, paths)
            evaluation = pipeline.evaluation_command(args, paths)
            self.assertEqual(training[training.index("--data-files") + 1], str(paths["tok_out"]))
            self.assertEqual(training[training.index("--max-length") + 1], "802")
            for key in ("bench_raw", "bench_tok", "bench_metadata"):
                self.assertNotIn(str(paths[key]), training)
                self.assertIn(str(paths[key]), evaluation)
            for key in ("raw_out", "tok_out", "metadata_out"):
                self.assertNotIn(str(paths[key]), evaluation)

    def test_smoke_defaults_isolate_outputs_and_preserve_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            defaults = self.parse(directory, "smoke")
            self.assertEqual((defaults.samples, defaults.run_name), (16, "gravity_smoke"))
            paths = pipeline.dataset_paths(defaults)
            self.assertTrue(all(path.parent == Path(directory).resolve() / "smoke" for path in paths.values()))
            args = self.parse(directory, "smoke", "--samples", "24", "--run-name", "custom_smoke")
            self.assertEqual((args.samples, args.run_name), (24, "custom_smoke"))

    def test_training_rejects_truncating_cap_before_starting_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch.object(pipeline, "verify_saved_datasets", return_value={"max_tokens": 4096}), \
             patch.object(pipeline.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "prevent training truncation"):
                pipeline.main(["train", "--output-dir", directory, "--max-tokens", "4095"])
            run.assert_not_called()

    def test_modes_route_generation_training_and_evaluation(self) -> None:
        expected = {
            "data": (1, 0, 0), "train": (0, 1, 0), "eval": (0, 0, 1),
            "full": (1, 1, 1), "smoke": (1, 1, 0),
        }
        with tempfile.TemporaryDirectory() as directory:
            for mode, (generate_count, train_count, eval_count) in expected.items():
                with self.subTest(mode=mode), patch.dict(os.environ, {}, clear=True), \
                     patch.object(pipeline, "generate_datasets") as generate, \
                     patch.object(pipeline, "verify_saved_datasets", return_value={"max_tokens": 4096}) as verify, \
                     patch.object(pipeline.subprocess, "run") as run:
                    pipeline.main([mode, "--output-dir", directory])
                    self.assertEqual(generate.call_count, generate_count)
                    self.assertEqual(verify.call_count, int(bool(train_count + eval_count)))
                    commands = [call.args[0] for call in run.call_args_list]
                    self.assertEqual(sum(any("transformer_trainer.py" in part for part in c) for c in commands), train_count)
                    self.assertEqual(sum("data_gen.data_gen_gravity.evaluate" in c for c in commands), eval_count)
                    self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))

    def test_small_physical_generation_alignment_manifest_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.parse(
                directory, "--samples", "8", "--benchmark-samples", "5", "--jobs", "1",
                "--min-scr", "2", "--max-scr", "2", "--max-terms", "2", "--overwrite",
            )
            paths = pipeline.dataset_paths(args)
            # Exercise a successful replacement as well as fresh serialization.
            for path in paths.values():
                path.write_bytes(b"old output")
            pipeline.generate_datasets(args, paths)
            pipeline.verify_saved_datasets(paths)
            tokenizer = pipeline.ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=args.max_tokens)
            manifest = json.loads(paths["manifest_out"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["training"]["rows"], 8)
            self.assertEqual(manifest["test"]["rows"], 10)
            self.assertEqual(manifest["holdout"]["shared_expressions"], 0)
            self.assertEqual(manifest["training"]["counts"], {
                f"{process}/{kind}": 2 for process in PROCESS_SPECS for kind in ("oneshot", "staged")
            })
            for key, record in manifest["files"].items():
                self.assertEqual(record["path"], str(paths[key]))
                self.assertEqual(record["sha256"], hashlib.sha256(paths[key].read_bytes()).hexdigest())
            for raw_key, tok_key, meta_key, count in (
                ("raw_out", "tok_out", "metadata_out", 8),
                ("bench_raw", "bench_tok", "bench_metadata", 10),
            ):
                raw, tokens, metadata = (read_csv(paths[key]) for key in (raw_key, tok_key, meta_key))
                self.assertEqual((len(raw), len(tokens), len(metadata)), (count, count, count))
                for expression, token_pair, provenance in zip(raw, tokens, metadata):
                    for column in ("simple", "scrambled"):
                        self.assertEqual(expression[column], provenance[column])
                        self.assertEqual(json.loads(token_pair[column]), tokenizer.encode_infix(expression[column]))
                        equivalent, _ = numerically_equivalent(
                            provenance["compact_origin"], expression[column], provenance["process"], seeds=(813,),
                        )
                        self.assertTrue(equivalent)
                    self.assertTrue(math.isfinite(float(provenance["relative_error"])))
                    if raw_key == "raw_out":
                        self.assertFalse(is_benchmark_leak(provenance["compact_origin"], provenance["process"]))
                    else:
                        self.assertEqual(provenance["compact_origin"], BENCHMARKS[provenance["process"]])
            original = paths["tok_out"].read_bytes()
            paths["tok_out"].write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "Dataset changed"):
                pipeline.verify_saved_datasets(paths)


if __name__ == "__main__":
    unittest.main()
