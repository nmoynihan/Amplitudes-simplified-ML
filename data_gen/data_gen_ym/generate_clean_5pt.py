#!/usr/bin/env python3
"""Generate corrected, zero-free five-point Yang--Mills training pairs.

This is the five-point entry point for the strict generation engine implemented
in :mod:`generate_clean_4pt`.  The engine is particle-count aware: five-point
data receives all generic Yang--Mills identities and numerical safeguards, but
does *not* use the special four-point complementary scalar-product identity.

Recommended invocation from the repository root::

    python -m data_gen.data_gen_ym.generate_clean_5pt \
        --samples 500000 --seed 42 --jobs auto

The command writes aligned raw and prefix-tokenized CSV files plus a JSON audit
report.  Existing files are never replaced unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

from . import generate_clean_4pt as _core


N_PARTICLES = 5

# Re-export the shared result types and numerical helpers for callers that use
# the four-point module programmatically.
CanonicalizationResult = _core.CanonicalizationResult
GenerationStats = _core.GenerationStats
KinematicPoint = _core.KinematicPoint
PreparedPair = _core.PreparedPair
ExpressionSyntaxError = _core.ExpressionSyntaxError
TokenizationError = _core.TokenizationError
parenthesize_for_semantic_tokenization = (
    _core.parenthesize_for_semantic_tokenization
)
evaluate_expression = _core.evaluate_expression
numerically_equivalent = _core.numerically_equivalent
numerically_zero = _core.numerically_zero
remove_numerically_zero_subsets = _core.remove_numerically_zero_subsets


def canonicalize_factor(factor: str):
    """Canonicalize one compact factor using five-point label rules."""

    return _core.canonicalize_factor(factor, n_particles=N_PARTICLES)


def canonicalize_simple_expression(expression: str) -> CanonicalizationResult:
    """Canonicalize a compact five-point Yang--Mills expression exactly."""

    return _core.canonicalize_simple_expression(
        expression,
        n_particles=N_PARTICLES,
    )


def build_kinematic_points(
    *,
    base_seed: int,
    checks_per_mode: int,
    energy_scale: float,
    pol_modes: Sequence[str] = _core.DEFAULT_POL_MODES,
) -> tuple[KinematicPoint, ...]:
    """Build deterministic all-massless five-point validation points."""

    return _core.build_kinematic_points(
        base_seed=base_seed,
        checks_per_mode=checks_per_mode,
        energy_scale=energy_scale,
        pol_modes=pol_modes,
        n_particles=N_PARTICLES,
    )


def prepare_pair(*args: Any, **kwargs: Any) -> PreparedPair | None:
    """Clean and validate one candidate with five-point assumptions."""

    kwargs["n_particles"] = N_PARTICLES
    return _core.prepare_pair(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    return _core.build_parser(n_particles=N_PARTICLES)


def generate_to_files(
    args: argparse.Namespace,
) -> tuple[GenerationStats, dict[str, Any]]:
    return _core.generate_to_files(
        args,
        n_particles=N_PARTICLES,
        generator_name="clean_5pt_yang_mills",
    )


def default_paths(samples: int):
    return _core._default_paths(samples, n_particles=N_PARTICLES)


def main(argv: Sequence[str] | None = None) -> int:
    return _core.main_for_particles(N_PARTICLES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
