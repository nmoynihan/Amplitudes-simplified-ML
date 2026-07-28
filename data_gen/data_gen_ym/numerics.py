"""Numerical evaluation for all-massless Yang–Mills expressions."""

from __future__ import annotations
import math
from typing import Sequence

from .notation import *
from .algebra import *
from .kinematics import generate_kinematics, mdot

try:
    from ..numeric_utils import numeric_values_close
except ImportError:  # Preserve imports when data_gen is placed directly on sys.path.
    from numeric_utils import numeric_values_close

DEFAULT_VALIDATION_POL_MODES = ("coulomb", "covariant")


def _strict_tokenize(expr: str) -> list[str]:
    """Tokenize without silently skipping unsupported characters or symbols."""
    text = expr.replace("^", "**")
    pos = 0
    tokens: list[str] = []
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            if not text[pos:].strip():
                break
            excerpt = text[pos : pos + 24]
            raise ValueError(
                f"eval_infix_numeric: unsupported syntax near {excerpt!r}"
            )
        tokens.append(match.group(1))
        pos = match.end()
    if not tokens:
        raise ValueError("eval_infix_numeric: empty expression")
    return tokens


def _validate_strict_token_sequence(tokens: list[str]) -> None:
    """Catch missing operands that the legacy parser represents as numeric zero."""
    multiplicative_ops = {"*", "/", "**"}
    bad_neighbor = {")", "*", "/", "**", "·", ".", ","}
    for index, token in enumerate(tokens):
        previous = tokens[index - 1] if index else None
        following = tokens[index + 1] if index + 1 < len(tokens) else None

        if token in {"+", "-"}:
            if following is None or following in bad_neighbor:
                raise ValueError(
                    f"eval_infix_numeric: operator {token!r} has no right operand"
                )
        elif token in multiplicative_ops:
            if (
                previous is None
                or previous in {"(", "+", "-", "*", "/", "**", "·", ".", ",", "Tr"}
                or following is None
                or following in bad_neighbor
            ):
                raise ValueError(
                    f"eval_infix_numeric: operator {token!r} has a missing operand"
                )
        elif token == "Tr" and following != "(":
            raise ValueError("eval_infix_numeric: Tr must be followed by '('")
        elif token == "(" and following == ")":
            raise ValueError("eval_infix_numeric: empty parentheses are not valid")
        elif token in {"·", ".", ","} and (
            previous is None
            or previous in {"(", "+", "-", "*", "/", "**", "·", ".", ",", "Tr"}
            or following is None
            or following in {")", "+", "-", "*", "/", "**", "·", ".", ","}
        ):
            raise ValueError(
                f"eval_infix_numeric: separator {token!r} has a missing operand"
            )


def _validate_source_ast(node, N: int) -> None:
    """Require a scalar Yang–Mills expression before expanding F-blocks."""
    if isinstance(node, _Num):
        return
    if isinstance(node, _UnaryOp):
        if node.op != "-":
            raise ValueError(f"eval_infix_numeric: unsupported unary operator {node.op!r}")
        _validate_source_ast(node.operand, N)
        return
    if isinstance(node, _BinOp):
        if node.op not in {"+", "-", "*", "/", "**"}:
            raise ValueError(f"eval_infix_numeric: unsupported operator {node.op!r}")
        _validate_source_ast(node.left, N)
        _validate_source_ast(node.right, N)
        return
    if isinstance(node, _Vec):
        raise ValueError(
            f"eval_infix_numeric: free vector {node.tag}_{node.idx} is not a scalar"
        )
    if isinstance(node, _DotChain):
        parts = node.parts
        is_trace = bool(parts) and parts[-1] is _DotChain._TR
        vectors = parts[:-1] if is_trace else parts
        if not vectors or any(not isinstance(vec, _Vec) for vec in vectors):
            raise ValueError("eval_infix_numeric: malformed vector chain")
        for vec in vectors:
            if vec.idx < 1 or vec.idx > N:
                raise KeyError(
                    f"eval_infix_numeric: unknown vector {vec.tag}_{vec.idx} (N={N})"
                )
        if is_trace:
            if len(vectors) < 2 or any(vec.tag != "F" for vec in vectors):
                raise ValueError(
                    "eval_infix_numeric: Tr requires a chain of at least two F_i vectors"
                )
            return
        if any(vec.tag == "F" for vec in vectors):
            valid_f_chain = (
                len(vectors) >= 3
                and vectors[0].tag == "p"
                and vectors[-1].tag == "p"
                and all(vec.tag == "F" for vec in vectors[1:-1])
            )
            if not valid_f_chain:
                raise ValueError(
                    "eval_infix_numeric: F_i is only valid inside p·F...·p or Tr(F...)"
                )
            return
        if len(vectors) != 2 or any(vec.tag not in {"p", "e"} for vec in vectors):
            raise ValueError(
                "eval_infix_numeric: a p/e dot product must contain exactly two vectors"
            )
        return
    raise ValueError(
        f"eval_infix_numeric: unsupported expression node {type(node).__name__}"
    )


def _validate_expanded_ast(node, N: int) -> None:
    """Ensure F expansion produced only numeric scalar operations and p/e dots."""
    if isinstance(node, _Num):
        return
    if isinstance(node, _UnaryOp):
        _validate_expanded_ast(node.operand, N)
        return
    if isinstance(node, _BinOp):
        _validate_expanded_ast(node.left, N)
        _validate_expanded_ast(node.right, N)
        return
    if isinstance(node, tuple) and len(node) == 3 and node[0] == _DOT_TAG:
        for vec in node[1:]:
            if (
                not isinstance(vec, tuple)
                or len(vec) != 3
                or vec[0] != _VEC_TAG
                or vec[1] not in {"p", "e"}
                or not isinstance(vec[2], int)
                or vec[2] < 1
                or vec[2] > N
            ):
                raise ValueError("eval_infix_numeric: malformed expanded dot product")
        return
    raise ValueError(
        f"eval_infix_numeric: non-scalar expanded node {type(node).__name__}"
    )


def eval_infix_numeric(expr: str, momenta, pols, *, strict: bool = False) -> float:
    """Evaluate an infix expression on explicit all-gluon kinematics.

    With ``strict=True``, reject unsupported symbols, malformed syntax, free
    vectors, invalid F-blocks, and out-of-range leg indices instead of mapping
    them to numeric zero.
    """
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

    tokens = _strict_tokenize(expr) if strict else _tokenize(expr)
    if strict:
        _validate_strict_token_sequence(tokens)
        depth = 0
        for token in tokens:
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("eval_infix_numeric: unmatched closing parenthesis")
        if depth:
            raise ValueError("eval_infix_numeric: unmatched opening parenthesis")

    parser = _Parser(tokens)
    tree = parser.parse()
    if strict:
        if parser.i != len(tokens):
            raise ValueError(
                "eval_infix_numeric: expression was not fully consumed "
                f"({parser.i}/{len(tokens)} tokens)"
            )
        _validate_source_ast(tree, N)
    expanded = _expand_ast(tree)
    if strict:
        _validate_expanded_ast(expanded, N)
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
                va = eval_infix_numeric(expr_a, mom, pol, strict=True)
                vb = eval_infix_numeric(expr_b, mom, pol, strict=True)
            except Exception as exc:
                return False, f"{pol_mode}:exception:{exc}"
            if not (math.isfinite(va) and math.isfinite(vb)):
                return False, f"{pol_mode}:non-finite"
            diff = abs(va - vb)
            if not numeric_values_close(
                va,
                vb,
                tol_abs=tol_abs,
                tol_rel=tol_rel,
            ):
                return False, f"{pol_mode}:mismatch:{diff:.3e}"
    return True, ""

__all__ = [
    'DEFAULT_VALIDATION_POL_MODES',
    'eval_infix_numeric',
    '_validate_pair',
]
