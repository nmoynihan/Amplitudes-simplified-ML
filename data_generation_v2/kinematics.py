#!/usr/bin/env python3
"""
kinematics.py — Random N-point phase-space generator.

    p_1, p_N        : massive scalars (mass M, positive energy)
    p_2 … p_{N-1}   : massless photons

All momenta are outgoing: Σ_i p_i = 0.   Signature (+,−,−,−).
Each photon carries a transverse polarisation ε_i with ε·k = 0, ε² = −1.

Two polarisation modes:
    'coulomb'   — ε^0 = 0, purely spatial  (default, fast)
    'covariant' — Lorentz-boosted from Coulomb gauge so ε^0 ≠ 0 in general;
                  exercises the full covariant structure in numerical tests.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm


# ── Helpers ──────────────────────────────────────────────────────────

def _random_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / norm(v)


def _lorentz_boost(vec4: np.ndarray, beta_vec: np.ndarray) -> np.ndarray:
    """Boost a 4-vector by velocity +β in (+,−,−,−)."""
    beta2 = np.dot(beta_vec, beta_vec)
    if beta2 < 1e-18:
        return vec4.copy()
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    bp = np.dot(beta_vec, vec4[1:])
    t_lab = gamma * (vec4[0] + bp)
    x_lab = vec4[1:] + ((gamma - 1.0) * bp / beta2 + gamma * vec4[0]) * beta_vec
    return np.concatenate(([t_lab], x_lab))


def _transverse_pol_coulomb(k4: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Coulomb-gauge polarisation: ε^0 = 0, ε · k = 0, ε² = −1."""
    k = k4[1:]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(k, ref) / norm(k)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    eps_spatial = np.cross(ref, k)
    eps_spatial /= norm(eps_spatial)
    return np.concatenate(([0.0], eps_spatial))


def _transverse_pol_covariant(k4: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Covariant polarisation using the residual gauge freedom ε → ε + α k.

    For massless k (k² = 0), adding α k^μ to any transverse ε preserves
    both ε · k = 0 and ε² = −1.  This gives ε^0 = α ω ≠ 0 in general,
    and F_μν = p_μ ε_ν − ε_μ p_ν is unchanged (gauge-invariant).
    """
    eps_coulomb = _transverse_pol_coulomb(k4, rng)
    alpha = rng.uniform(-0.5, 0.5)
    return eps_coulomb + alpha * k4


# ── Main generator ───────────────────────────────────────────────────

def generate_kinematics(
    N: int,
    M: float = 1.0,
    *,
    E_min: float | None = None,
    E_max: float | None = None,
    pol_mode: str = "coulomb",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one random N-point phase-space configuration.

    Returns
    -------
    momenta       : (N, 4) array   [p_1 … p_N]
    polarisations : (N-2, 4) array [ε_2 … ε_{N-1}]
    """
    if E_min is None:
        E_min = 0.3 * M
    if E_max is None:
        E_max = 1.5 * M
    assert E_max > E_min > 0

    pol_fn = _transverse_pol_covariant if pol_mode == "covariant" else _transverse_pol_coulomb
    rng = np.random.default_rng(seed)
    n_phot = N - 2

    # 1) Sample photon momenta until √s > 2M
    while True:
        energies = rng.uniform(E_min, E_max, size=n_phot)
        photons = np.stack([
            np.concatenate(([E], E * _random_unit_vec(rng)))
            for E in energies
        ])
        K = photons.sum(axis=0)
        s = K[0] ** 2 - np.dot(K[1:], K[1:])
        if s > 4.0 * M ** 2:
            break

    # 2) Two-body decay in the rest frame of the parent
    sqrt_s = np.sqrt(s)
    E_star = sqrt_s / 2.0
    p_mag = np.sqrt(E_star ** 2 - M ** 2)
    n_hat = _random_unit_vec(rng)

    p1_rest = np.array([E_star, *(p_mag * n_hat)])
    pN_rest = np.array([E_star, *(-p_mag * n_hat)])

    # 3) Boost scalars so total momentum is +K (balances flipped photons)
    beta = K[1:] / K[0]
    p1 = _lorentz_boost(p1_rest, beta)
    pN = _lorentz_boost(pN_rest, beta)

    # 4) Flip photons → outgoing convention
    photons = -photons

    # 5) Polarisations
    pols = np.stack([pol_fn(k4, rng) for k4 in photons])

    # 6) Assemble
    momenta = np.vstack((p1, photons, pN))
    return momenta, pols


# ── Minkowski inner product (public, reused by evaluator) ────────────

def mdot(a: np.ndarray, b: np.ndarray) -> float:
    """Minkowski inner product with (+,−,−,−) signature."""
    return float(a[0] * b[0] - np.dot(a[1:], b[1:]))


# ── Self-tests ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    np.set_printoptions(precision=6, suppress=True)

    def _check(N, M, pol_mode="coulomb"):
        mom, pol = generate_kinematics(N, M, pol_mode=pol_mode)
        msq = lambda v: v[0] ** 2 - np.dot(v[1:], v[1:])

        assert np.allclose(msq(mom[0]), M ** 2, atol=1e-10)
        assert np.allclose(msq(mom[-1]), M ** 2, atol=1e-10)
        for v in mom[1:-1]:
            assert abs(msq(v)) < 1e-10

        tot = mom.sum(axis=0)
        assert np.allclose(tot, 0.0, atol=1e-10), f"momentum not conserved: {tot}"

        for k, eps in zip(mom[1:-1], pol):
            assert abs(mdot(eps, k)) < 1e-10, "ε · k ≠ 0"
            assert abs(mdot(eps, eps) + 1.0) < 1e-10, "ε² ≠ −1"

    t0 = time.perf_counter()
    for _ in range(200):
        for N in (5, 6, 7):
            _check(N, 1.7, "coulomb")
            _check(N, 2.3, "covariant")
    dt = time.perf_counter() - t0
    print(f"All kinematic tests passed ({dt:.2f}s)")
