"""Regression test for independently scrambled gluon-amplitude round trips.

Run from the repository root with:

    python3 -m unittest data_gen.data_gen_ym.test_tokenizer_roundtrip -v

The full terminal report is also written to:

    Yang_Mills_Generation_Testing.txt
"""

from __future__ import annotations

import math
import random
import unittest
from pathlib import Path

from ..Tokenizer import ScatteringAmplitudeTokenizer
from .algebra import simplify_to_lowest_terms
from .expr_model import _build_base_expression
from .kinematics import generate_kinematics
from .notation import (
    DENOM_REPEAT_PROBABILITY,
    OLD_STYLE_PROBABILITY,
    SCALAR_POWER_PROBABILITY,
    UNIT_PROBABILITY,
)
from .numerics import eval_infix_numeric
from .scramble import scramble


class GluonAmplitudeRoundTripTests(unittest.TestCase):
    """Check that scrambling and token round trips preserve an amplitude."""

    N_PARTICLES = (4, 5)
    REPETITIONS = 3
    KINEMATIC_SEEDS = (7001, 7002, 7003)
    POLARISATION_MODES = ("coulomb", "covariant")
    ABS_TOL = 1e-10
    REL_TOL = 1e-8
    TOKENIZER_MAX_PARTICLES = 8
    MAX_SEQUENCE_LENGTH = 2048
    REPORT_PATH = (
        Path(__file__).resolve().parents[2]
        / "Yang_Mills_Generation_Testing.txt"
    )

    def _generate_amplitude(
        self,
        n_particles: int,
        seed: int,
    ) -> tuple[str, str]:
        """Generate one deterministic compact/expanded gluon amplitude."""
        random.seed(seed)
        for _ in range(40):
            generated = _build_base_expression(
                n_particles,
                unit_probability=UNIT_PROBABILITY,
                old_style_probability=OLD_STYLE_PROBABILITY,
                denom_repeat_probability=DENOM_REPEAT_PROBABILITY,
                scalar_power_probability=SCALAR_POWER_PROBABILITY,
                use_denominators=True,
                min_terms=1,
                max_terms=1,
            )
            if generated is not None:
                return generated
        self.fail(f"Could not generate a {n_particles}-gluon amplitude")

    def _distinct_tokenizable_scramble(
        self,
        expression: str,
        n_particles: int,
        seed: int,
        tokenizer: ScatteringAmplitudeTokenizer,
        *,
        excluded_raw: set[str],
        excluded_simplified: set[str],
    ) -> tuple[str, str, list[int]]:
        """Produce a distinct deterministic scramble within the token budget."""
        for offset in range(80):
            random.seed(seed + offset)
            raw_candidate = scramble(
                expression,
                n_particles,
                min_scr=2,
                max_scr=5,
                full_expand=True,
            )
            if raw_candidate in excluded_raw:
                continue

            simplified_candidate = simplify_to_lowest_terms(raw_candidate)
            if (
                simplified_candidate.strip() == "0"
                or simplified_candidate in excluded_simplified
            ):
                continue

            try:
                tokens = tokenizer.encode_infix(simplified_candidate)
            except ValueError:
                # This mirrors the token-length filtering in build_dataset.
                continue

            return raw_candidate, simplified_candidate, tokens

        self.fail(
            "Could not produce a distinct, tokenizable "
            f"{n_particles}-gluon scramble from seed {seed}"
        )

    def _equivalent_at_point(
        self,
        reference_value: float,
        candidate_value: float,
    ) -> tuple[bool, float, float]:
        """Return agreement plus the absolute difference and allowed tolerance."""
        if not (
            math.isfinite(reference_value) and math.isfinite(candidate_value)
        ):
            return False, math.inf, 0.0

        difference = abs(reference_value - candidate_value)
        tolerance = max(
            self.ABS_TOL,
            self.REL_TOL
            * max(abs(reference_value), abs(candidate_value)),
        )
        return difference <= tolerance, difference, tolerance

    def test_near_zero_comparison_has_no_unit_scale_floor(self) -> None:
        """Independently guard the regression that made 1e-9 equal to zero."""
        cases = (
            (0.0, 0.5e-10, True),
            (1.0, 1.0 + 0.5e-8, True),
            (0.0, 1e-9, False),
            (0.0, 1e-8, False),
        )
        for reference, candidate, expected in cases:
            with self.subTest(reference=reference, candidate=candidate):
                # math.isclose is independent of the production comparator and
                # implements the documented abs/rel tolerance convention.
                independent = math.isclose(
                    reference,
                    candidate,
                    abs_tol=self.ABS_TOL,
                    rel_tol=self.REL_TOL,
                )
                self.assertEqual(independent, expected)

                actual, difference, tolerance = self._equivalent_at_point(
                    reference,
                    candidate,
                )
                self.assertEqual(actual, independent)
                self.assertEqual(difference, abs(reference - candidate))
                self.assertEqual(
                    tolerance,
                    max(
                        self.ABS_TOL,
                        self.REL_TOL
                        * max(abs(reference), abs(candidate)),
                    ),
                )

        reference = 2.0 * self.ABS_TOL
        self.assertGreater(reference, self.ABS_TOL)
        self.assertFalse(self._equivalent_at_point(reference, 0.0)[0])

    def test_independent_scrambles_survive_tokenizer_round_trip(self) -> None:
        random_state = random.getstate()
        report_sections = [
            "Yang-Mills generation, scrambling, and tokenizer round-trip test"
        ]

        try:
            for n_particles in self.N_PARTICLES:
                tokenizer = ScatteringAmplitudeTokenizer(
                    max_particles=self.TOKENIZER_MAX_PARTICLES,
                    max_sequence_length=self.MAX_SEQUENCE_LENGTH,
                )

                for repetition in range(self.REPETITIONS):
                    with self.subTest(
                        n_particles=n_particles,
                        repetition=repetition,
                    ):
                        base_seed = 1000 * n_particles + repetition
                        compact, expanded = self._generate_amplitude(
                            n_particles,
                            base_seed,
                        )

                        # Start with two genuinely identical copies, then
                        # scramble them using independent random streams.
                        amplitude_a = expanded
                        amplitude_b = expanded
                        self.assertEqual(amplitude_a, amplitude_b)

                        (
                            raw_scramble_a,
                            scrambled_a,
                            tokens_a,
                        ) = self._distinct_tokenizable_scramble(
                            amplitude_a,
                            n_particles,
                            seed=base_seed + 100,
                            tokenizer=tokenizer,
                            excluded_raw={expanded},
                            excluded_simplified=set(),
                        )
                        (
                            raw_scramble_b,
                            scrambled_b,
                            tokens_b,
                        ) = self._distinct_tokenizable_scramble(
                            amplitude_b,
                            n_particles,
                            seed=base_seed + 200,
                            tokenizer=tokenizer,
                            excluded_raw={expanded, raw_scramble_a},
                            excluded_simplified={scrambled_a},
                        )

                        self.assertNotEqual(scrambled_a.strip(), "0")
                        self.assertNotEqual(scrambled_b.strip(), "0")
                        self.assertNotEqual(scrambled_a, scrambled_b)
                        self.assertNotEqual(tokens_a, tokens_b)

                        decoded_a = tokenizer.decode_infix(tokens_a)
                        decoded_b = tokenizer.decode_infix(tokens_b)

                        # Associative expressions need not recover the exact
                        # prefix tree, but both decoded forms must be accepted
                        # by the tokenizer and remain numerically equivalent.
                        tokenizer.encode_infix(decoded_a)
                        tokenizer.encode_infix(decoded_b)

                        expressions = {
                            "compact": compact,
                            "expanded": expanded,
                            "raw_scramble_a": raw_scramble_a,
                            "raw_scramble_b": raw_scramble_b,
                            "simplified_scramble_a": scrambled_a,
                            "simplified_scramble_b": scrambled_b,
                            "decoded_a": decoded_a,
                            "decoded_b": decoded_b,
                        }
                        numerical_matches = {
                            label: True
                            for label in expressions
                            if label != "compact"
                        }
                        decoded_pair_matches = True
                        numerical_failures: list[str] = []
                        reference_was_nonzero = False

                        for mode in self.POLARISATION_MODES:
                            for kinematic_seed in self.KINEMATIC_SEEDS:
                                momenta, polarisations = generate_kinematics(
                                    n_particles,
                                    E_scale=2.0,
                                    pol_mode=mode,
                                    seed=kinematic_seed + 10 * repetition,
                                )
                                values = {
                                    label: eval_infix_numeric(
                                        expression,
                                        momenta,
                                        polarisations,
                                        strict=True,
                                    )
                                    for label, expression in expressions.items()
                                }

                                reference = values["compact"]
                                self.assertTrue(
                                    math.isfinite(reference),
                                    "compact amplitude is non-finite",
                                )
                                reference_was_nonzero |= (
                                    abs(reference) > self.ABS_TOL
                                )

                                for label, value in values.items():
                                    if label == "compact":
                                        continue
                                    passed, difference, tolerance = (
                                        self._equivalent_at_point(
                                            reference,
                                            value,
                                        )
                                    )
                                    numerical_matches[label] &= passed
                                    if not passed:
                                        numerical_failures.append(
                                            f"{mode}, seed={kinematic_seed}: "
                                            f"{label} changed numerically "
                                            f"(difference={difference:.3e}, "
                                            f"tolerance={tolerance:.3e})"
                                        )

                                pair_passed, difference, tolerance = (
                                    self._equivalent_at_point(
                                        values["decoded_a"],
                                        values["decoded_b"],
                                    )
                                )
                                decoded_pair_matches &= pair_passed
                                if not pair_passed:
                                    numerical_failures.append(
                                        f"{mode}, seed={kinematic_seed}: "
                                        "decoded_a and decoded_b differ "
                                        f"(difference={difference:.3e}, "
                                        f"tolerance={tolerance:.3e})"
                                    )

                        def result(passed: bool) -> str:
                            return "PASS" if passed else "FAIL"

                        case_report = (
                            "\n"
                            f"=== {n_particles}-gluon round trip "
                            f"{repetition + 1}/{self.REPETITIONS} ===\n"
                            "Initial expanded gluon amplitude "
                            "(copied twice before scrambling):\n"
                            f"{expanded}\n\n"
                            "Scrambled A after tokenize -> detokenize "
                            f"({len(tokens_a)} tokens):\n"
                            f"{decoded_a}\n\n"
                            "Scrambled B after tokenize -> detokenize "
                            f"({len(tokens_b)} tokens):\n"
                            f"{decoded_b}\n\n"
                            "Numerical equivalence "
                            f"({len(self.KINEMATIC_SEEDS)} seeds x "
                            f"{len(self.POLARISATION_MODES)} gauges):\n"
                            "  initial == scrambled A: "
                            f"{result(numerical_matches['decoded_a'])}\n"
                            "  initial == scrambled B: "
                            f"{result(numerical_matches['decoded_b'])}\n"
                            "  scrambled A == scrambled B: "
                            f"{result(decoded_pair_matches)}\n"
                            "  non-zero reference observed: "
                            f"{result(reference_was_nonzero)}"
                        )
                        print(case_report, flush=True)
                        report_sections.append(case_report)

                        self.assertFalse(
                            numerical_failures,
                            "\n".join(numerical_failures[:10]),
                        )

                        # Prevent an all-zero evaluation from making the
                        # equivalence assertions pass vacuously.
                        self.assertTrue(
                            reference_was_nonzero,
                            "generated amplitude evaluated to zero at every "
                            "sampled kinematic point",
                        )
        finally:
            random.setstate(random_state)
            self.REPORT_PATH.write_text(
                "\n\n".join(report_sections) + "\n",
                encoding="utf-8",
            )
            print(
                f"\nSaved test output to {self.REPORT_PATH}",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
