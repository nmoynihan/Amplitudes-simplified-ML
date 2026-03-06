#!/usr/bin/env python3
# kinematics.py
#
# Generate one random N‑point phase‑space point:
#   p1 , pN   : massive scalars (mass = M, positive energy)
#   p2…p_{N-1}: massless photons
#
# All momenta are outgoing and obey Σ_i p_i = 0 in (+,−,−,−) signature.
# Each photon has a real transverse polarisation ε_i with ε·k = 0 and ε² = −1.
#
# Includes extended self‑tests when run as __main__.

from __future__ import annotations
import numpy as np
from numpy.linalg import norm
import random, time
try:
    import colorama
    _HAS_COLORAMA = True
except ImportError:
    colorama = None
    _HAS_COLORAMA = False

# ╭──────────────────────────────────────────────────────────────────╮
# │  Helpers                                                        │
# ╰──────────────────────────────────────────────────────────────────╯
def random_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / norm(v)

def lorentz_boost(vec4: np.ndarray, beta_vec: np.ndarray) -> np.ndarray:
    """
    Forward‐boost a 4‑vector from its rest frame to the lab frame moving with
    velocity +β⃗ in (+,−,−,−) signature:
      t  = γ ( t' + β·x' )
      x  = x' + (γ−1)(β·x')β/β² + γ t' β
    """
    beta2 = np.dot(beta_vec, beta_vec)
    if beta2 < 1e-18:
        return vec4.copy()
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    bp    = np.dot(beta_vec, vec4[1:])
    # time component
    t_lab = gamma * ( vec4[0] + bp )
    # spatial component
    factor1 = (gamma - 1.0) * bp / beta2
    factor2 = gamma * vec4[0]
    x_lab = vec4[1:] + factor1 * beta_vec + factor2 * beta_vec
    return np.concatenate(([t_lab], x_lab))

def transverse_pol(k4: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Real transverse polarisation ε for a massless 4‑vector k4.
    ε·k = 0 ,  ε² = -1  in (+,−,−,−).
    """
    k = k4[1:]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(k, ref)/norm(k)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    eps_spatial = np.cross(ref, k)
    eps_spatial /= norm(eps_spatial)
    return np.concatenate(([0.0], eps_spatial))

# ╭──────────────────────────────────────────────────────────────────╮
# │  Main kinematics generator                                       │
# ╰──────────────────────────────────────────────────────────────────╯
def generate_kinematics(N: int,
                        M: float = 1.0,
                        *,
                        E_min: float | None = None,
                        E_max: float | None = None,
                        seed: int | None = None):
    """
    Returns
      momenta      : (N,4) array of 4‑vectors [p1 … pN]
      polarisations: (N-2,4) array [ε2 … ε_{N-1}]
    """
    # default photon energies scaled to M
    if E_min is None: E_min = 0.3 * M
    if E_max is None: E_max = 1.5 * M
    assert E_max > E_min > 0

    rng = np.random.default_rng(seed)
    n_phot = N - 2

    # 1) Sample photon momenta until √s > 2M
    while True:
        photons = np.stack([
            np.concatenate(([E := rng.uniform(E_min, E_max)],
                            E * random_unit_vec(rng)))
            for _ in range(n_phot)
        ])
        K = photons.sum(axis=0)
        s = K[0]**2 - np.dot(K[1:], K[1:])
        if s > 4.0 * M**2:
            break

    # 2) Two-body decay of a particle with momentum -K and mass √s
    # In the rest frame of the parent: 
    sqrt_s = np.sqrt(s)
    E_star = sqrt_s / 2.0
    p_mag = np.sqrt(E_star**2 - M**2)
    n_hat = random_unit_vec(rng)

    p1_rest = np.array([E_star, *(p_mag * n_hat)])
    p2_rest = np.array([E_star, *(-p_mag * n_hat)])

    # 3) The scalars should have total momentum +K to balance -K from flipped photons
    beta = K[1:] / K[0]
    
    p1 = lorentz_boost(p1_rest, beta)
    pN = lorentz_boost(p2_rest, beta)

    # 4) Flip photon momenta so total momentum is zero
    photons = -photons

    # 5) Photon polarisations (use final photon momenta)
    pols = np.stack([transverse_pol(k4, rng) for k4 in photons])

    # 6) Assemble
    momenta = np.vstack((p1, photons, pN))
    return momenta, pols

# ╭──────────────────────────────────────────────────────────────────╮
# │  Extended self‑tests (only if run as script)                     │
# ╰──────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    if _HAS_COLORAMA:
        colorama.init(autoreset=True)
    np.set_printoptions(precision=6, suppress=True)

    def check_point(N, M):
        mom, pol = generate_kinematics(N, M)
        # on‑shell mass squares
        msq = lambda v: v[0]**2 - np.dot(v[1:], v[1:])
        assert np.allclose(msq(mom[0]),  M**2, atol=1e-10)
        assert np.allclose(msq(mom[-1]), M**2, atol=1e-10)
        for v in mom[1:-1]:
            assert abs(msq(v)) < 1e-10

        # momentum conservation
        tot = mom.sum(axis=0)
        if not np.allclose(tot, 0.0, atol=1e-10):
            _red = colorama.Fore.RED if _HAS_COLORAMA else ""
            print(_red + f" Error: total momentum = {tot}")
        assert np.allclose(tot, 0.0, atol=1e-10)

        # polarisation checks
        for i,(k,eps) in enumerate(zip(mom[1:-1], pol), start=2):
            assert abs(np.dot(eps, k))   < 1e-12  # ε·p = 0
            val = eps[0]**2 - np.dot(eps[1:], eps[1:])
            assert abs(val + 1.0)        < 1e-12  # ε² = -1

    print(f"\nRunning extended kinematic tests...")
    t0 = time.perf_counter()
    for _ in range(100):
        check_point(N=5, M=1.7)
        check_point(N=6, M=2.3)
        check_point(N=7, M=5)
    dt = time.perf_counter() - t0
    _green = colorama.Fore.GREEN if _HAS_COLORAMA else ""
    print(_green + f"all kinematic tests passed ({dt:.2f}s)")
