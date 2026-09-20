"""Independent five-point CHY validation of fixed scalar orderings.

This is a numerical verifier for the two four-dimensional benchmark sectors,
not a general gravity amplitude or training-data generator. Scalar orderings
are alpha=beta=(1,2,3), H=(4,5), and alpha=(1,2,3,4), beta=(1,3,2,4),
H=(5,). The gravitons in H are spectators of the scalar ordering and have
positive helicity; their polarization tensor is epsilon_mu epsilon_nu.

Conventions: d_ab=p_a.p_b, E_a=sum_{b!=a} d_ab/(z_a-z_b)=0,
PT(alpha)=1/product(z_alpha_i-z_alpha_(i+1)). Psi_H has blocks
[A,-C.T; C,B], with A_ab=d_ab/z_ab, B_ab=e_a.e_b/z_ab,
C_ab=e_a.p_b/z_ab and C_aa=-sum_{j!=a} e_a.p_j/z_aj. Diagonals of A,B
vanish. The repo-normalized block is -sum PT(alpha)PT(beta)Pf(Psi_H)^2
/det'(Phi), where Phi_ab=d_ab/z_ab^2 (a!=b), Phi_aa=-sum_{b!=a}Phi_ab.
The global minus is the residue-orientation convention fixed here.

We fix z_1=0,z_2=1,z_5=-1, so det'(Phi) is the minor deleting 1,2,5
divided by (z_12 z_25 z_51)^2. Both five-point scattering-equation solutions
are evaluated. The resulting blocks equal Eq. (4.7), and twice Eq. (4.8),
of arXiv:2408.04720 in the repository's sqrt(2) polarization convention.
The CHY/transmutation construction follows arXiv:1808.07451, Eq. (4.10).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import BENCHMARKS, eval_expression, paper_spinor_value
from .kinematics import SpinorKinematics, generate_kinematics, mdot, with_references

ORDERINGS = {
    "3s2h": ((1, 2, 3), (1, 2, 3), (4, 5)),
    "4s1h": ((1, 2, 3, 4), (1, 3, 2, 4), (5,)),
}


def _parke_taylor(order: Sequence[int], z: np.ndarray) -> complex:
    return complex(1 / np.prod([
        z[order[k] - 1] - z[order[(k + 1) % len(order)] - 1]
        for k in range(len(order))
    ]))


def ordered_chy_value(
    kin: SpinorKinematics,
    alpha: Sequence[int],
    beta: Sequence[int],
    graviton_legs: Sequence[int],
) -> tuple[complex, dict[str, float]]:
    """Evaluate the selected five-point CHY block and equation residuals.

    The two solutions use angle and square spinors respectively. Numerical
    verification requires generic momenta, away from coalescing solutions and
    singular puncture gauges. ``generate_kinematics`` provides generic samples.
    """
    alpha, beta, gravitons = tuple(alpha), tuple(beta), tuple(graviton_legs)
    if (len(alpha) not in (3, 4) or len(beta) != len(alpha)
            or len(set(alpha)) != len(alpha) or len(set(beta)) != len(beta)
            or set(alpha) != set(beta) or len(set(gravitons)) != len(gravitons)
            or len(alpha) + len(gravitons) != 5
            or set(alpha) & set(gravitons)
            or set(alpha) | set(gravitons) != set(range(1, 6))):
        raise ValueError("Expected two scalar orders and one or two spectator gravitons covering 1..5")
    p, e = kin.momenta, kin.polarisations
    d = np.asarray([[mdot(a, b) for b in p] for a in p])
    total = 0j
    max_absolute_residual = max_scaled_residual = 0.0
    for bracket in (kin.angle, kin.square):
        z = np.asarray([
            bracket(i, 1) * bracket(2, 5)
            / (2 * bracket(i, 5) * bracket(2, 1)
               - bracket(i, 1) * bracket(2, 5))
            for i in range(1, 6)
        ])
        phi = np.zeros((5, 5), complex)
        for i in range(5):
            terms = [d[i, j] / (z[i] - z[j]) for j in range(5) if i != j]
            residual = abs(sum(terms))
            max_absolute_residual = max(max_absolute_residual, residual)
            max_scaled_residual = max(
                max_scaled_residual, residual / max(1.0, sum(abs(t) for t in terms))
            )
            for j in range(5):
                if i != j:
                    phi[i, j] = d[i, j] / (z[i] - z[j]) ** 2
            phi[i, i] = -sum(phi[i, :])
        jacobian = np.linalg.det(phi[np.ix_([2, 3], [2, 3])]) / (
            (z[0] - z[1]) * (z[1] - z[4]) * (z[4] - z[0])
        ) ** 2
        c = np.zeros((len(gravitons), len(gravitons)), complex)
        for i, a in enumerate(gravitons):
            for j, b in enumerate(gravitons):
                c[i, j] = (
                    mdot(e[a], p[b - 1]) / (z[a - 1] - z[b - 1]) if a != b
                    else -sum(mdot(e[a], p[l - 1]) / (z[a - 1] - z[l - 1])
                              for l in range(1, 6) if l != a)
                )
        if len(gravitons) == 1:
            pfaffian = -c[0, 0]
        else:
            a, b = gravitons
            pfaffian = (d[a - 1, b - 1] * mdot(e[a], e[b]) / (z[a - 1] - z[b - 1]) ** 2
                        - c[0, 0] * c[1, 1] + c[0, 1] * c[1, 0])
        total += _parke_taylor(alpha, z) * _parke_taylor(beta, z) * pfaffian ** 2 / jacobian
    if not np.isfinite(abs(total)) or max_scaled_residual > 1e-9:
        raise ArithmeticError("Ill-conditioned five-point CHY evaluation")
    return -complex(total), {
        "absolute": float(max_absolute_residual), "scaled": float(max_scaled_residual),
    }


def _swap_gravitons(kin: SpinorKinematics) -> SpinorKinematics:
    order = [0, 1, 2, 4, 3]
    relabel = {1: 1, 2: 2, 3: 3, 4: 5, 5: 4}
    return replace(
        kin, lambdas=kin.lambdas[order], tildes=kin.tildes[order], momenta=kin.momenta[order],
        polarisations={relabel[i]: value for i, value in kin.polarisations.items()},
        references={relabel[i]: relabel[j] for i, j in kin.references.items()},
    )


def verify_chy(seeds: Sequence[int] = (17, 31, 73)) -> dict:
    """Check paper, fixtures, auxiliary reconstruction and ordered symmetries."""
    from .ordered_benchmarks import reconstruct_benchmark

    if not seeds:
        raise ValueError("At least one kinematic seed is required")
    report = {
        "definition": "five-point double scalar ordering with positive-helicity spectator gravitons",
        "sources": ["https://arxiv.org/pdf/1808.07451", "https://arxiv.org/html/2408.04720v2"],
        "normalization": "minus CHY residue sum with d_ab=p_a.dot(p_b) and sqrt(2) polarizations",
        "scope": "numerical checks in four dimensions; not a general generator or proof for other sectors",
        "checks": {},
    }
    for process, (alpha, beta, gravitons) in ORDERINGS.items():
        auxiliary = reconstruct_benchmark(process)
        worst = dict.fromkeys(("chy_vs_fixture", "chy_vs_paper", "chy_vs_auxiliary_sum",
                               "gauge_invariance", "independent_scalar_cyclic_rotations"), 0.0)
        if process == "3s2h":
            worst["graviton_label_swap"] = 0.0
        residuals = {"absolute": 0.0, "scaled": 0.0}
        configurations = cyclic_checks = 0

        def record(name, actual, expected, context):
            scale = max(abs(actual), abs(expected))
            difference = abs(actual - expected)
            error = difference / scale if scale else 0.0
            if not np.isfinite(error) or difference > 1e-10 + 5e-8 * scale:
                raise AssertionError(f"{process}, {context}, {name}: {error}")
            worst[name] = max(worst[name], float(error))

        for seed in seeds:
            base = generate_kinematics(seed=int(seed), graviton_legs=gravitons)
            base_value, _ = ordered_chy_value(base, alpha, beta, gravitons)
            for mode in ("cyclic", "first", "last", "random", "shifted"):
                kin = with_references(
                    base, gravitons, reference_mode="cyclic" if mode == "shifted" else mode,
                    seed=int(seed) + 911,
                    gauge_shifts=({h: .73 + .27j for h in gravitons} if mode == "shifted" else None),
                )
                actual, current_residuals = ordered_chy_value(kin, alpha, beta, gravitons)
                context = f"seed {seed}, gauge {mode}"
                record("chy_vs_fixture", actual, eval_expression(BENCHMARKS[process], kin), context)
                record("chy_vs_paper", actual, paper_spinor_value(process, kin), context)
                record("chy_vs_auxiliary_sum", actual, eval_expression(auxiliary, kin), context)
                record("gauge_invariance", actual, base_value, context)
                for name in residuals:
                    residuals[name] = max(residuals[name], current_residuals[name])
                for i in range(len(alpha)):
                    for j in range(len(beta)):
                        cyclic_value, _ = ordered_chy_value(
                            kin, alpha[i:] + alpha[:i], beta[j:] + beta[:j], gravitons,
                        )
                        record("independent_scalar_cyclic_rotations", cyclic_value, actual, context)
                        cyclic_checks += 1
                if process == "3s2h":
                    swapped, _ = ordered_chy_value(_swap_gravitons(kin), alpha, beta, gravitons)
                    record("graviton_label_swap", swapped, actual, context)
                configurations += 1
        report["checks"][process] = {
            "alpha": list(alpha), "beta": list(beta), "spectator_gravitons": list(gravitons),
            # Multiply the repository value by this factor to obtain the paper value.
            "repository_to_paper_factor": 1 if process == "3s2h" else 0.5,
            "kinematic_seeds": [int(seed) for seed in seeds], "configurations": configurations,
            "cyclic_rotation_checks": cyclic_checks,
            "maximum_relative_errors": worst, "maximum_scattering_equation_residuals": residuals,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30, help="number of kinematic points, with seeds 0..N-1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    text = json.dumps(verify_chy(tuple(range(args.seeds))), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
