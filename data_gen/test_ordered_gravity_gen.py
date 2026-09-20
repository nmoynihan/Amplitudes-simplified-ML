"""Dataset contract and leakage checks for the ordered gravity release.

The held-out objects are the two reconstruction components of each benchmark.
These tests include relabelings outside the selected four-scalar flavour channel:
those remain excluded from training even though they are not test targets.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import io
import itertools
import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from . import ordered_gravity_gen as pipeline
from .Tokenizer import ScatteringAmplitudeTokenizer
from .data_gen_gravity.core import (
    BENCHMARKS,
    PROCESS_SPECS,
    expand_expression,
    expression_mass_dimension,
    field_strength_counts_per_term,
    generate_target,
    numerically_equivalent,
)
from .data_gen_gravity.generate import Candidate
from .data_gen_gravity.ordered_benchmarks import reconstruction_terms


def candidate(expression: str, *, process: str = "3s2h", **changes) -> Candidate:
    row = Candidate(
        simple=expression, scrambled=expand_expression(expression), process=process,
        kind="oneshot", seed=17, scramble_depth=1, scramble_labels="fixture",
        stage="compact", compact_terms=1, simple_tokens=1, scrambled_tokens=1,
        relative_error=0.0, compact_origin=expression,
    )
    return replace(row, **changes)


def relabel(expression: str, order: tuple[int, ...]) -> str:
    # Deliberately independent of the production relabeling implementation.
    return re.sub(r"([peF])_(\d+)\b", lambda m: f"{m[1]}_{order[int(m[2]) - 1]}", expression)


def read_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class OrderedHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.held_out = [candidate(term, process=process)
                        for process in PROCESS_SPECS
                        for term in reconstruction_terms(process)]
        cls.guard = pipeline.OrderedHoldoutGuard(cls.held_out, max_tokens=4096)
        cls.safe = candidate(generate_target("3s2h", rng=random.Random(1924),
                                             min_terms=1, max_terms=1))

    def test_full_and_component_families_exclude_every_species_relabeling(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            expressions = (BENCHMARKS[process], *reconstruction_terms(process))
            for scalar_order in itertools.permutations(spec.scalar_legs):
                for graviton_order in itertools.permutations(spec.graviton_legs):
                    order = scalar_order + graviton_order
                    for expression in expressions:
                        with self.subTest(process=process, order=order, expression=expression):
                            row = candidate(relabel(expression, order), process=process)
                            self.assertIsNotNone(self.guard.rejection_reason(row))

    def test_nonzero_rational_rescaling_does_not_create_training_examples(self) -> None:
        for process in PROCESS_SPECS:
            for expression in (BENCHMARKS[process], *reconstruction_terms(process)):
                for coefficient in ("7", "-3", "2/7"):
                    with self.subTest(process=process, coefficient=coefficient):
                        row = candidate(f"({coefficient})*({expression})", process=process)
                        self.assertIsNotNone(self.guard.rejection_reason(row))

    def test_commuting_products_and_reordering_terms_cannot_bypass_holdout(self) -> None:
        # Same m3, with every scalar factor in a different order.
        m3 = (
            "-((p_2 · F_5 · p_3)*(p_1 · F_5 · p_3)"
            "*(p_1 · F_4 · p_5)*(p_1 · F_4 · p_2))"
            "/((p_4 · p_5)*(p_3 · p_5)*(p_2 · p_5)"
            "*(p_2 · p_4)*(p_1 · p_5)*(p_1 · p_4))"
        )
        # Same m4, with its second summand first and factors commuted.
        m4 = (
            "(p_3 · F_5 · p_4)*(p_1 · F_5 · p_4)"
            "/((p_4 · p_5)*(p_3 · p_5)*(p_1 · p_5)*(p_2 · p_3))"
            " + (p_2 · F_5 · p_3)*(p_1 · F_5 · p_4)"
            "/((p_3 · p_5)*(p_1 · p_5)*(p_2 · p_3)*(p_1 · p_4)*2)"
        )
        for process, expression in (("3s2h", m3), ("4s1h", m4)):
            self.assertIsNotNone(self.guard.rejection_reason(candidate(expression, process=process)))

    def test_staged_rows_are_classified_by_their_compact_origin(self) -> None:
        for row in self.held_out:
            staged = replace(row, simple=self.safe.simple, scrambled=self.safe.scrambled,
                             kind="staged", stage="scramble-2-to-1")
            self.assertIsNotNone(self.guard.rejection_reason(staged))
        with self.assertRaises(ValueError):
            self.guard.rejection_reason(replace(self.safe, compact_origin=""))

    def test_exact_overlap_is_checked_across_both_columns(self) -> None:
        self.assertIsNone(self.guard.rejection_reason(self.safe))
        for held_out in self.held_out:
            for column in ("simple", "scrambled"):
                for expression in (held_out.simple, held_out.scrambled):
                    # Preserve the unrelated compact origin to isolate the
                    # expression-overlap check from family classification.
                    row = replace(self.safe, **{column: f" ( {expression} ) "})
                    with self.subTest(column=column, expression=expression):
                        self.assertIsNotNone(self.guard.rejection_reason(row))

    def test_empty_test_set_is_not_a_valid_holdout_guard(self) -> None:
        with self.assertRaises(ValueError):
            pipeline.OrderedHoldoutGuard([], max_tokens=4096)


class OrderedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.options = dict(samples=24, test_size=8, seed=1924, split_seed=271828,
                           jobs=1, min_scr=1, max_scr=2, min_terms=1,
                           max_terms=2, max_tokens=4096)
        cls.training, cls.test, cls.report = pipeline.build_datasets(**cls.options)

    def test_exact_balanced_train_and_benchmark_quotas(self) -> None:
        self.assertEqual((len(self.training), len(self.test)), (16, 8))
        self.assertEqual(Counter((row.process, row.kind) for row in self.training),
                         Counter({(p, k): 4 for p in PROCESS_SPECS
                                  for k in ("oneshot", "staged")}))
        target_counts = Counter((row.process, row.compact_origin) for row in self.test)
        self.assertEqual(target_counts, Counter({(p, term): 2 for p in PROCESS_SPECS
                                                 for term in reconstruction_terms(p)}))
        for process, expression in target_counts:
            depths = Counter(row.scramble_depth for row in self.test
                             if row.process == process and row.compact_origin == expression)
            self.assertEqual(depths, {1: 1, 2: 1})

    def test_generated_pairs_satisfy_physics_and_have_correct_compact_degree(self) -> None:
        tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=4096)
        for row in (*self.training, *self.test):
            with self.subTest(process=row.process, kind=row.kind, seed=row.seed):
                ok, error = numerically_equivalent(row.simple, row.scrambled, row.process,
                                                   seeds=(17, 31))
                self.assertTrue(ok, f"Relative error: {error}")
                for expression in (row.simple, row.scrambled):
                    decoded = tokenizer.decode_infix(tokenizer.encode_infix(expression))
                    ok, error = numerically_equivalent(expression, decoded, row.process,
                                                       seeds=(17, 31))
                    self.assertTrue(ok, f"Token semantics changed: {error}")
                spec = PROCESS_SPECS[row.process]
                self.assertEqual(expression_mass_dimension(row.compact_origin), spec.target_dimension)
                self.assertTrue(all(count == {h: 2 for h in spec.graviton_legs}
                                    for count in field_strength_counts_per_term(row.compact_origin)))

    def test_holdout_is_disjoint_at_expression_and_family_levels(self) -> None:
        tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=4096)
        def expressions(rows: list[Candidate]) -> set[tuple[int, ...]]:
            return {tuple(tokenizer.encode_infix(expression)) for row in rows
                    for expression in (row.simple, row.scrambled)}
        self.assertFalse(expressions(self.training) & expressions(self.test))
        guard = pipeline.OrderedHoldoutGuard(self.test, max_tokens=4096)
        self.assertTrue(all(guard.rejection_reason(row) is None for row in self.training))
        for rows in (self.training, self.test):
            self.assertEqual(len({(r.simple, r.scrambled) for r in rows}), len(rows))

    def test_seeded_generation_is_reproducible(self) -> None:
        training, test, _ = pipeline.build_datasets(**self.options)
        self.assertEqual(training, self.training)
        self.assertEqual(test, self.test)

    def test_single_process_and_uneven_quotas_are_exact(self) -> None:
        training, test, _ = pipeline.build_datasets(
            samples=8, test_size=3, process="4s1h", kind="oneshot",
            seed=1924, split_seed=271828, jobs=1, min_scr=1, max_scr=2,
            min_terms=1, max_terms=1,
        )
        self.assertEqual((len(training), len(test)), (5, 3))
        self.assertTrue(all(row.process == "4s1h" for row in (*training, *test)))
        self.assertEqual(sorted(Counter(row.compact_origin for row in test).values()), [1, 2])
        self.assertTrue(all(row.kind == "oneshot" for row in training))

    def test_invalid_generation_parameters_are_rejected(self) -> None:
        for changes in (
            {"samples": 0}, {"test_size": 0}, {"test_size": 24}, {"test_size": 25},
            {"jobs": 0}, {"process": "invalid"}, {"kind": "invalid"},
            {"min_scr": 3, "max_scr": 2}, {"min_terms": 3, "max_terms": 2},
            {"max_tokens": 0}, {"max_attempts_factor": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pipeline.build_datasets(**(self.options | changes))


class OrderedOutputTests(unittest.TestCase):
    def test_half_coefficient_survives_the_legacy_tokenizer(self) -> None:
        tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=4096)
        numerator = "(p_1 · F_5 · p_4)*(p_2 · F_5 · p_3)"
        denominator = "(p_1 · p_4)*(p_2 · p_3)*(p_1 · p_5)*(p_3 · p_5)"
        expected = f"({numerator})/(2*({denominator}))"
        for coefficient in ("0.5", "(1/2)"):
            with self.subTest(coefficient=coefficient):
                original = f"{coefficient}*({numerator})/({denominator})"
                rendered = pipeline.parenthesize_for_semantic_tokenization(original)
                self.assertNotRegex(rendered, r"\d+\.\d+")
                decoded = tokenizer.decode_infix(tokenizer.encode_infix(rendered))
                ok, error = numerically_equivalent(expected, decoded, "4s1h", seeds=(17, 31))
                self.assertTrue(ok, f"Half coefficient changed by tokenization: {error}")

    def test_cli_writes_aligned_tokenized_splits_and_refuses_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            argv = ["--samples", "12", "--test-size", "4", "--jobs", "1",
                    "--seed", "1924", "--split-seed", "271828",
                    "--min-scr", "1", "--max-scr", "1", "--min-terms", "1",
                    "--max-terms", "1", "--output-dir", directory]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(argv), 0)
            tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=4096)
            files = set(Path(directory).iterdir())
            self.assertEqual({p.name for p in files}, {
                f"ordered_gravity_{split}_{kind}.csv.gz"
                for split in ("train", "test") for kind in ("raw", "tok", "metadata")
            } | {"ordered_gravity_manifest.json"})
            for split, count in (("train", 8), ("test", 4)):
                raw, tok, metadata = [read_csv(Path(directory) / f"ordered_gravity_{split}_{kind}.csv.gz")
                                      for kind in ("raw", "tok", "metadata")]
                self.assertEqual((len(raw), len(tok), len(metadata)), (count, count, count))
                for raw_row, tok_row, meta_row in zip(raw, tok, metadata):
                    for column in ("simple", "scrambled"):
                        self.assertEqual(json.loads(tok_row[column]), tokenizer.encode_infix(raw_row[column]))
                        self.assertEqual(meta_row[column], raw_row[column])
                    self.assertTrue(meta_row["compact_origin"])
                    self.assertTrue(meta_row["object_kind"])
                    self.assertTrue(meta_row["reference_order"])
            manifest = json.loads((Path(directory) / "ordered_gravity_manifest.json").read_text())
            self.assertEqual(manifest["training"]["rows"], 8)
            self.assertEqual(manifest["test"]["rows"], 4)
            self.assertEqual(manifest["audit"]["shared_expressions"], 0)
            self.assertEqual(len(manifest["files"]), 6)
            for record in manifest["files"].values():
                self.assertEqual(hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest(),
                                 record["sha256"])
            saved = {p: p.read_bytes() for p in files}
            with patch.object(pipeline, "build_datasets") as build, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertNotEqual(pipeline.main(argv), 0)
            build.assert_not_called()
            self.assertEqual(saved, {p: p.read_bytes() for p in files})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main([*argv, "--overwrite"]), 0)

    def test_direct_and_module_entrypoints_expose_the_cli(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for entrypoint in ([str(root / "data_gen" / "ordered_gravity_gen.py")],
                           ["-m", "data_gen.ordered_gravity_gen"]):
            with self.subTest(entrypoint=entrypoint):
                result = subprocess.run([sys.executable, *entrypoint, "--help"], cwd=root,
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                for option in ("--samples", "--test-size", "--output-dir", "--split-seed"):
                    self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
