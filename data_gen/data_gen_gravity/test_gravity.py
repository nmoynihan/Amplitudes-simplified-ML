"""Regression and smoke tests for the five-point gravity pipeline."""

from __future__ import annotations

import random
import unittest
from collections import Counter

import numpy as np

from ..Tokenizer import ScatteringAmplitudeTokenizer
from .core import (
    BENCHMARKS,
    PROCESS_SPECS,
    _relabel,
    count_expanded_terms,
    eval_expression,
    expand_expression,
    expression_mass_dimension,
    field_strength_counts_per_term,
    generate_target,
    is_benchmark_leak,
    numerically_equivalent,
    verify_paper_benchmarks,
)
from .generate import build_benchmarks, build_dataset
from .kinematics import generate_kinematics, mdot, with_references
from .scramble import SCRAMBLE_NAMES, scramble_trajectory


class KinematicsTests(unittest.TestCase):
    def test_massless_conserved_transverse_null(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            kin = generate_kinematics(
                seed=81, graviton_legs=spec.graviton_legs
            )
            self.assertLess(np.max(np.abs(np.sum(kin.momenta, axis=0))), 1e-9)
            for momentum in kin.momenta:
                self.assertLess(abs(mdot(momentum, momentum)), 1e-9)
            for leg in spec.graviton_legs:
                polarisation = kin.polarisations[leg]
                self.assertLess(
                    abs(mdot(polarisation, kin.momenta[leg - 1])), 1e-9
                )
                self.assertLess(abs(mdot(polarisation, polarisation)), 1e-9)

    def test_reference_and_explicit_gauge_independence(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            kin = generate_kinematics(
                seed=91, graviton_legs=spec.graviton_legs
            )
            alternate = with_references(
                kin, spec.graviton_legs, reference_mode="last"
            )
            shifted = with_references(
                kin,
                spec.graviton_legs,
                reference_mode="cyclic",
                gauge_shifts={leg: 0.3 - 0.2j for leg in spec.graviton_legs},
            )
            values = [
                eval_expression(BENCHMARKS[process], point)
                for point in (kin, alternate, shifted)
            ]
            for candidate in values[1:]:
                difference = abs(values[0] - candidate)
                tolerance = max(
                    1e-12,
                    1e-9 * max(abs(values[0]), abs(candidate)),
                )
                self.assertLessEqual(difference, tolerance)


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_exclusion_handles_relabeling_and_factor_order(self) -> None:
        for process, compact in BENCHMARKS.items():
            spec = PROCESS_SPECS[process]
            mapping = {leg: leg for leg in range(1, 6)}
            mapping.update(zip(spec.scalar_legs, reversed(spec.scalar_legs)))
            mapping.update(zip(spec.graviton_legs, reversed(spec.graviton_legs)))
            relabeled = _relabel(compact, mapping)
            self.assertTrue(is_benchmark_leak(relabeled, process))

        compact = BENCHMARKS["4s1h"]
        reordered = compact.replace(
            "(p_1 · F_5 · p_4)*(p_2 · F_5 · p_3)",
            "(p_2 · F_5 · p_3)*(p_1 · F_5 · p_4)",
        )
        self.assertNotEqual(compact, reordered)
        self.assertTrue(is_benchmark_leak(reordered, "4s1h"))

    def test_malformed_benchmark_checks_fail_closed(self) -> None:
        malformed = ("", "not_an_expression", "(p_1 · p_2", "p_1 · p_2 +", "1 2")
        for expression in malformed:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(
                    ValueError, "Cannot check benchmark exclusion"
                ):
                    is_benchmark_leak(expression, "4s1h")

    def test_paper_formulas_and_expansion_sizes(self) -> None:
        errors = verify_paper_benchmarks()
        self.assertLess(max(errors.values()), 1e-9)
        self.assertEqual(count_expanded_terms(BENCHMARKS["3s2h"]), 32)
        self.assertEqual(count_expanded_terms(BENCHMARKS["4s1h"]), 12)

    def test_benchmark_dimensions_and_multiplicities(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            self.assertEqual(
                expression_mass_dimension(BENCHMARKS[process]),
                spec.target_dimension,
            )
            expected = {leg: 2 for leg in spec.graviton_legs}
            self.assertTrue(
                all(
                    counts == expected
                    for counts in field_strength_counts_per_term(
                        BENCHMARKS[process]
                    )
                )
            )

    def test_small_heldout_set_is_unique_and_depth_balanced(self) -> None:
        rows = build_benchmarks(
            scrambles_per_amplitude=5, seed=1234, validate=True
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row.scrambled for row in rows}), 10)
        self.assertTrue(
            all(row.compact_origin == BENCHMARKS[row.process] for row in rows)
        )
        self.assertEqual(
            Counter((row.process, row.scramble_depth) for row in rows),
            Counter(
                (process, depth)
                for process in PROCESS_SPECS
                for depth in range(1, 6)
            ),
        )


class GeneratorTests(unittest.TestCase):
    def test_staged_rows_preserve_their_original_compact_target(self) -> None:
        rows = build_dataset(
            4,
            kind="staged",
            seed=991,
            min_scr=2,
            max_scr=2,
            min_terms=1,
            max_terms=2,
            validate=True,
        )
        self.assertTrue(any(row.simple != row.compact_origin for row in rows))
        for row in rows:
            compact = generate_target(
                row.process,
                rng=random.Random(row.seed),
                min_terms=1,
                max_terms=2,
            )
            self.assertEqual(row.compact_origin, compact)
            self.assertFalse(is_benchmark_leak(row.compact_origin, row.process))
            equivalent, _ = numerically_equivalent(
                row.simple, row.compact_origin, row.process, seeds=(201,)
            )
            self.assertTrue(equivalent)

    def test_compact_targets_obey_manifest(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            rng = random.Random(123)
            for _ in range(8):
                target = generate_target(process, rng=rng)
                self.assertFalse(is_benchmark_leak(target, process))
                self.assertEqual(
                    expression_mass_dimension(target), spec.target_dimension
                )
                expected = {leg: 2 for leg in spec.graviton_legs}
                self.assertTrue(
                    all(
                        counts == expected
                        for counts in field_strength_counts_per_term(target)
                    )
                )
                expanded = expand_expression(target)
                equivalent, _ = numerically_equivalent(
                    target, expanded, process, seeds=(201,)
                )
                self.assertTrue(equivalent)

    def test_each_scrambler_round_trips(self) -> None:
        for process, spec in PROCESS_SPECS.items():
            target = generate_target(
                process,
                rng=random.Random(51),
                min_terms=2,
                max_terms=2,
            )
            expanded = expand_expression(target)
            for index, name in enumerate(SCRAMBLE_NAMES):
                trajectory = scramble_trajectory(
                    expanded,
                    spec,
                    rng=random.Random(600 + index),
                    depth=1,
                    names=(name,),
                )
                self.assertEqual(len(trajectory), 1, name)
                equivalent, _ = numerically_equivalent(
                    expanded,
                    trajectory[-1].expression,
                    process,
                    seeds=(301,),
                )
                self.assertTrue(equivalent, name)

    def test_balanced_deterministic_deduplicated_tokenisable_dataset(self) -> None:
        kwargs = dict(
            process="mixed",
            kind="mixed",
            seed=991,
            min_scr=1,
            max_scr=2,
            min_terms=1,
            max_terms=2,
            max_tokens=4096,
            validate=True,
        )
        first = build_dataset(8, **kwargs)
        second = build_dataset(8, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter((row.process, row.kind) for row in first),
            Counter(
                {
                    ("3s2h", "oneshot"): 2,
                    ("3s2h", "staged"): 2,
                    ("4s1h", "oneshot"): 2,
                    ("4s1h", "staged"): 2,
                }
            ),
        )
        self.assertEqual(
            len({(row.simple, row.scrambled) for row in first}), len(first)
        )
        tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8, max_sequence_length=4096
        )
        for row in first:
            self.assertNotIn(
                tokenizer.vocab["<UNK>"], tokenizer.encode_infix(row.simple)
            )
            self.assertNotIn(
                tokenizer.vocab["<UNK>"], tokenizer.encode_infix(row.scrambled)
            )


if __name__ == "__main__":
    unittest.main()
