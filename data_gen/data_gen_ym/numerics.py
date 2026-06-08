"""numerics — extracted from gen_data.py (scaffold, verbatim)."""

from __future__ import annotations
import math
from typing import Sequence

from .notation import *
from .algebra import *
from .kinematics import generate_kinematics, mdot

DEFAULT_VALIDATION_POL_MODES = ("coulomb", "covariant")


def eval_infix_numeric(expr: str, momenta, pols) -> float:
    """Evaluate an infix expression on explicit kinematics."""
    N = len(momenta)
    # All N legs are gluons: every leg carries a polarisation (fix 1).
    P = {f"p_{i}": momenta[i - 1] for i in range(1, N + 1)}
    E = {f"e_{i}": pols[i - 1] for i in range(1, N + 1)}

    def _eval(node):
        if isinstance(node, _Num):
            return node.value
        if isinstance(node, _BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            if node.op == "/":
                return left / right
            if node.op == "**":
                return left ** right
            raise ValueError(f"Unknown operator {node.op}")
        if isinstance(node, _UnaryOp):
            value = _eval(node.operand)
            return -value if node.op == "-" else value
        if isinstance(node, tuple) and node[0] == _DOT_TAG:
            lhs, rhs = node[1], node[2]
            if lhs[0] == _VEC_TAG and rhs[0] == _VEC_TAG:
                va = P.get(f"p_{lhs[2]}") if lhs[1] == "p" else E.get(f"e_{lhs[2]}")
                vb = P.get(f"p_{rhs[2]}") if rhs[1] == "p" else E.get(f"e_{rhs[2]}")
                # Fix 2: a missing vector is a bug, not a zero. Scream, don't lie.
                if va is None or vb is None:
                    missing = []
                    if va is None:
                        missing.append(f"{lhs[1]}_{lhs[2]}")
                    if vb is None:
                        missing.append(f"{rhs[1]}_{rhs[2]}")
                    raise KeyError(f"eval_infix_numeric: unknown vector(s) {missing} (N={N})")
                return mdot(va, vb)
            return 0.0
        return 0.0

    tree = _Parser(_tokenize(expr)).parse()
    expanded = _expand_ast(tree)
    return float(_eval(expanded))


def _validate_pair(
    expr_a: str,
    expr_b: str,
    N: int,
    M: float,
    *,
    n_checks: int = 3,
    tol_rel: float = 1e-8,
    tol_abs: float = 1e-10,
    pol_modes: Sequence[str] = DEFAULT_VALIDATION_POL_MODES,
) -> tuple[bool, str]:
    for pol_mode in pol_modes:
        for _ in range(n_checks):
            mom, pol = generate_kinematics(N, E_scale=M, pol_mode=pol_mode)
            try:
                va = eval_infix_numeric(expr_a, mom, pol)
                vb = eval_infix_numeric(expr_b, mom, pol)
            except Exception as exc:
                return False, f"{pol_mode}:exception:{exc}"
            if not (math.isfinite(va) and math.isfinite(vb)):
                return False, f"{pol_mode}:non-finite"
            diff = abs(va - vb)
            if diff > max(tol_abs, tol_rel * max(1.0, abs(va), abs(vb))):
                return False, f"{pol_mode}:mismatch:{diff:.3e}"
    return True, ""

__all__ = [
    'DEFAULT_VALIDATION_POL_MODES',
    'eval_infix_numeric',
    '_validate_pair',
]
