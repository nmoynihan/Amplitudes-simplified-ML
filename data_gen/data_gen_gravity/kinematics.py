"""Complex massless five-point kinematics and positive-helicity polarisations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def mdot(a: np.ndarray, b: np.ndarray) -> complex:
    """Mostly-minus Minkowski dot product, without complex conjugation."""
    return complex(a[0] * b[0] - np.dot(a[1:], b[1:]))


def angle(a: np.ndarray, b: np.ndarray) -> complex:
    return complex(a[0] * b[1] - a[1] * b[0])


def square(a: np.ndarray, b: np.ndarray) -> complex:
    return complex(a[0] * b[1] - a[1] * b[0])


def bispinor_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Map ``v_{a dot a}`` to ``(v0,v1,v2,v3)``."""
    return np.asarray(
        [
            (matrix[0, 0] + matrix[1, 1]) / 2,
            (matrix[0, 1] + matrix[1, 0]) / 2,
            (matrix[1, 0] - matrix[0, 1]) / (2j),
            (matrix[0, 0] - matrix[1, 1]) / 2,
        ],
        dtype=np.complex128,
    )


def spinor_to_momentum(lam: np.ndarray, tilde: np.ndarray) -> np.ndarray:
    return bispinor_to_vector(np.outer(lam, tilde))


def positive_polarisation(
    lam: np.ndarray,
    tilde: np.ndarray,
    reference_lam: np.ndarray,
) -> np.ndarray:
    """Standard positive-helicity vector ``sqrt(2)|q><p|/<q p>``."""
    denominator = angle(reference_lam, lam)
    if abs(denominator) < 1e-12:
        raise ValueError("Reference spinor is collinear with the momentum spinor")
    return bispinor_to_vector(
        np.sqrt(2.0) * np.outer(reference_lam, tilde) / denominator
    )


@dataclass(frozen=True)
class SpinorKinematics:
    """One complex five-point phase-space point."""

    lambdas: np.ndarray
    tildes: np.ndarray
    momenta: np.ndarray
    polarisations: Mapping[int, np.ndarray]
    references: Mapping[int, int]

    def angle(self, i: int, j: int) -> complex:
        return angle(self.lambdas[i - 1], self.lambdas[j - 1])

    def square(self, i: int, j: int) -> complex:
        return square(self.tildes[i - 1], self.tildes[j - 1])


def _random_spinor(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=2) + 1j * rng.normal(size=2)


def _reference_map(
    lambdas: np.ndarray,
    graviton_legs: Sequence[int],
    rng: np.random.Generator,
    mode: str,
) -> dict[int, int]:
    refs: dict[int, int] = {}
    for leg in graviton_legs:
        candidates = [
            j
            for j in range(1, 6)
            if j != leg and abs(angle(lambdas[j - 1], lambdas[leg - 1])) > 1e-8
        ]
        if not candidates:
            raise ValueError("No non-collinear reference spinor")
        if mode == "first":
            refs[leg] = candidates[0]
        elif mode == "last":
            refs[leg] = candidates[-1]
        elif mode == "cyclic":
            refs[leg] = min(candidates, key=lambda j: (j - leg) % 5)
        elif mode == "random":
            refs[leg] = int(rng.choice(candidates))
        else:
            raise ValueError(f"Unknown reference mode: {mode}")
    return refs


def _is_well_conditioned(
    lambdas: np.ndarray,
    tildes: np.ndarray,
    momenta: np.ndarray,
    min_invariant: float,
) -> bool:
    if np.max(np.abs(np.sum(momenta, axis=0))) > 1e-8:
        return False
    if max(abs(mdot(p, p)) for p in momenta) > 1e-8:
        return False
    for i in range(5):
        for j in range(i + 1, 5):
            if abs(mdot(momenta[i], momenta[j])) < min_invariant:
                return False
            if abs(angle(lambdas[i], lambdas[j])) < 1e-7:
                return False
            if abs(square(tildes[i], tildes[j])) < 1e-7:
                return False
    return True


def generate_kinematics(
    *,
    seed: int | None = None,
    graviton_legs: Sequence[int] = (4, 5),
    reference_mode: str = "cyclic",
    gauge_shifts: Mapping[int, complex] | None = None,
    min_invariant: float = 2e-3,
    max_attempts: int = 500,
) -> SpinorKinematics:
    """Generate generic complex massless five-point kinematics.

    Three pairs of spinors are sampled freely.  The anti-holomorphic spinors
    of legs 4 and 5 are then solved for exactly from momentum conservation.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        lambdas = np.stack([_random_spinor(rng) for _ in range(5)])
        tildes = np.empty((5, 2), dtype=np.complex128)
        tildes[:3] = np.stack([_random_spinor(rng) for _ in range(3)])

        a54 = angle(lambdas[4], lambdas[3])
        a45 = angle(lambdas[3], lambdas[4])
        if abs(a54) < 1e-5:
            continue
        tildes[3] = -sum(
            angle(lambdas[4], lambdas[i]) * tildes[i] for i in range(3)
        ) / a54
        tildes[4] = -sum(
            angle(lambdas[3], lambdas[i]) * tildes[i] for i in range(3)
        ) / a45
        momenta = np.stack(
            [spinor_to_momentum(lambdas[i], tildes[i]) for i in range(5)]
        )
        if not _is_well_conditioned(
            lambdas, tildes, momenta, min_invariant=min_invariant
        ):
            continue

        try:
            references = _reference_map(
                lambdas, tuple(graviton_legs), rng, reference_mode
            )
            pols: dict[int, np.ndarray] = {}
            for leg in graviton_legs:
                pol = positive_polarisation(
                    lambdas[leg - 1],
                    tildes[leg - 1],
                    lambdas[references[leg] - 1],
                )
                if gauge_shifts and leg in gauge_shifts:
                    pol = pol + gauge_shifts[leg] * momenta[leg - 1]
                pols[leg] = pol
        except ValueError:
            continue

        if any(abs(mdot(pols[i], momenta[i - 1])) > 1e-8 for i in graviton_legs):
            continue
        if any(abs(mdot(pols[i], pols[i])) > 1e-8 for i in graviton_legs):
            continue
        return SpinorKinematics(lambdas, tildes, momenta, pols, references)
    raise RuntimeError("Could not generate a well-conditioned five-point point")


def with_references(
    kin: SpinorKinematics,
    graviton_legs: Sequence[int],
    *,
    reference_mode: str,
    seed: int = 0,
    gauge_shifts: Mapping[int, complex] | None = None,
) -> SpinorKinematics:
    """Rebuild polarisations at fixed momenta in another reference gauge."""
    rng = np.random.default_rng(seed)
    references = _reference_map(
        kin.lambdas, tuple(graviton_legs), rng, reference_mode
    )
    pols: dict[int, np.ndarray] = {}
    for leg in graviton_legs:
        pol = positive_polarisation(
            kin.lambdas[leg - 1],
            kin.tildes[leg - 1],
            kin.lambdas[references[leg] - 1],
        )
        if gauge_shifts and leg in gauge_shifts:
            pol = pol + gauge_shifts[leg] * kin.momenta[leg - 1]
        pols[leg] = pol
    return SpinorKinematics(
        kin.lambdas, kin.tildes, kin.momenta, pols, references
    )
