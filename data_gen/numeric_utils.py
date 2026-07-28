"""Shared numerical-comparison helpers for scientific validation."""

from __future__ import annotations

import math


def numeric_values_close(
    value_a: float | complex,
    value_b: float | complex,
    *,
    tol_abs: float,
    tol_rel: float,
) -> bool:
    """Return whether two finite values agree at the requested tolerances.

    This is the finite-only analogue of :func:`math.isclose`.  In particular,
    relative tolerance is scaled only by the magnitudes being compared; there
    is no implicit unit-scale floor near zero.
    """
    if not (math.isfinite(tol_abs) and tol_abs >= 0):
        raise ValueError("tol_abs must be finite and non-negative")
    if not (math.isfinite(tol_rel) and tol_rel >= 0):
        raise ValueError("tol_rel must be finite and non-negative")
    # ``abs`` maps real and complex scalar values to a real magnitude, so the
    # finite-only contract works for both SQED/YM and complex gravity values.
    if not (
        math.isfinite(abs(value_a))
        and math.isfinite(abs(value_b))
    ):
        return False

    return abs(value_a - value_b) <= max(
        tol_abs,
        tol_rel * max(abs(value_a), abs(value_b)),
    )


__all__ = ["numeric_values_close"]
