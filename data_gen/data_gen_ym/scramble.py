"""scramble — extracted from gen_data.py (scaffold, verbatim)."""

from __future__ import annotations
import random
import re
from typing import Sequence

from notation import *
from algebra import *

SCRAMBLE_MULTIPLY_ONE = "multiply_one"


SCRAMBLE_WARD = "ward"


SCRAMBLE_MOMENTUM = "momentum"


SCRAMBLE_COMMUTE_DOT = "commute_dot"


SCRAMBLE_RATIO = "ratio"


SCRAMBLE_PARTIAL_FRACTION = "partial_fraction"


SCRAMBLE_WARD_ALL = "ward_all"


SCRAMBLE_POLARISATION_ZERO = "polarisation_zero"


SCRAMBLE_TERM_REORDER = "term_reorder"


DEFAULT_SCRAMBLES = (
    SCRAMBLE_MULTIPLY_ONE,
    SCRAMBLE_WARD,
    SCRAMBLE_MOMENTUM,
    SCRAMBLE_COMMUTE_DOT,
    SCRAMBLE_RATIO,
    SCRAMBLE_PARTIAL_FRACTION,
    SCRAMBLE_WARD_ALL,
    SCRAMBLE_POLARISATION_ZERO,
    SCRAMBLE_TERM_REORDER,
)


DEFAULT_MIN_SCR = 1


DEFAULT_MAX_SCR = 5


DEFAULT_MAX_SCRAMBLED_LEN = 4000


def scr_mul_by_one(expr: str, N: int) -> str:
    i, j = random.sample(range(1, N + 1), 2)
    one = f"({dot(p(i), p(j))})/({dot(p(i), p(j))})"
    return f"({expr})*{one}"


def scr_ward_substitute(expr: str, N: int) -> str:
    photon = random.randint(1, N)  # all N legs are gluons
    leg = random.randint(1, N)
    target = re.escape(dot(e(photon), p(leg)))
    repl = "-(" + " + ".join(dot(e(photon), p(s)) for s in range(1, N + 1) if s != leg) + ")"
    return re.sub(target, repl, expr, count=1)


def scr_momentum_substitute(expr: str, N: int) -> str:
    matches = list(_RE_pp.finditer(expr))
    if not matches:
        return expr
    match = random.choice(matches)
    a = int(match.group(1))
    b = int(match.group(2))
    repl = "-(" + " + ".join(dot(p(s), p(b)) for s in range(1, N + 1) if s != a) + ")"
    return expr[: match.start()] + repl + expr[match.end() :]


def scr_commute_dot(expr: str) -> str:
    matches = list(_RE_DOT.finditer(expr))
    if not matches:
        return expr
    match = random.choice(matches)
    return expr[: match.start()] + dot(match.group(2), match.group(1)) + expr[match.end() :]


def scr_mul_by_ratio(expr: str, N: int) -> str:
    legs = random.sample(range(1, N + 1), min(4, N))
    if len(legs) < 4:
        return scr_mul_by_one(expr, N)
    i, j, k, l = legs
    denom = f"({dot(p(i), p(j))} + {dot(p(k), p(l))})"
    return f"({expr})*{denom}/{denom}"


def _random_context_factor(expr: str, N: int, max_factors: int = 2) -> str:
    factors: list[str] = []
    existing_dots = [m.group(0) for m in _RE_DOT.finditer(expr)]
    n_factors = random.randint(0, max_factors)

    for _ in range(n_factors):
        if existing_dots and random.random() < 0.7:
            factors.append(random.choice(existing_dots))
            continue
        if gluon_legs(N) and random.random() < 0.5:
            factors.append(dot(e(random.choice(gluon_legs(N))), p(random.randint(1, N))))
        else:
            i, j = random.sample(range(1, N + 1), 2)
            factors.append(dot(p(i), p(j)))

    return "*".join(factors)


def _find_denom_blocks(expr: str) -> list[tuple[int, int, int]]:
    blocks: list[tuple[int, int, int]] = []
    i = 0
    while i < len(expr) - 1:
        if expr[i] == "/" and expr[i + 1] == "(":
            depth = 0
            for j in range(i + 1, len(expr)):
                if expr[j] == "(":
                    depth += 1
                elif expr[j] == ")":
                    depth -= 1
                if depth == 0:
                    blocks.append((i, i + 1, j))
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1
    return blocks


def _find_paren_block_ending_at(expr: str, end: int) -> int | None:
    if end <= 0 or expr[end - 1] != ")":
        return None
    depth = 0
    for j in range(end - 1, -1, -1):
        if expr[j] == ")":
            depth += 1
        elif expr[j] == "(":
            depth -= 1
        if depth == 0:
            return j
    return None


def scr_partial_fraction(expr: str) -> str:
    denom_blocks = _find_denom_blocks(expr)
    viable: list[tuple[int, int, int, list[str], list[int], int]] = []
    for slash_pos, dopen, dclose in denom_blocks:
        denom = expr[dopen + 1 : dclose]
        factors = _split_top_level(denom, "*")
        pp_idx = [i for i, factor in enumerate(factors) if _RE_pp.fullmatch(factor.strip())]
        if len(pp_idx) < 2:
            continue
        num_start = _find_paren_block_ending_at(expr, slash_pos)
        if num_start is None:
            continue
        viable.append((slash_pos, dopen, dclose, factors, pp_idx, num_start))
    if not viable:
        return expr

    slash_pos, _dopen, dclose, factors, pp_idx, num_start = random.choice(viable)
    ia, ib = random.sample(pp_idx, 2)
    Da = factors[ia].strip()
    Db = factors[ib].strip()
    rest = [f.strip() for i, f in enumerate(factors) if i not in (ia, ib)]
    numerator = expr[num_start:slash_pos]
    diff = f"({Da} - {Db})"
    den1 = f"({diff}*{Db}" + (f"*{'*'.join(rest)})" if rest else ")")
    den2 = f"({diff}*{Da}" + (f"*{'*'.join(rest)})" if rest else ")")
    prefix = expr[:num_start]
    suffix = expr[dclose + 1 :]
    return f"{prefix}({numerator}/{den1} - {numerator}/{den2}){suffix}"


def scr_ward_substitute_all(expr: str, N: int) -> str:
    """Like scr_ward_substitute but replaces every occurrence of the chosen e_j·p_k."""
    photon = random.randint(1, N)  # all N legs are gluons
    leg = random.randint(1, N)
    target = re.escape(dot(e(photon), p(leg)))
    repl = "-(" + " + ".join(dot(e(photon), p(s)) for s in range(1, N + 1) if s != leg) + ")"
    return re.sub(target, repl, expr)


def scr_add_polarisation_zero(expr: str, N: int) -> str:
    """Add a multiple of the Ward-identity zero sum_s e_j·p_s = 0 to the expression."""
    photon = random.randint(1, N)  # all N legs are gluons
    ward_sum = " + ".join(dot(e(photon), p(s)) for s in range(1, N + 1))
    zero = f"({ward_sum})"

    context = _random_context_factor(expr, N)
    zero_term = zero if not context else f"{zero}*{context}"

    denom_blocks = _find_denom_blocks(expr)
    if denom_blocks and random.random() < 0.5:
        _slash_pos, dopen, dclose = random.choice(denom_blocks)
        denom = expr[dopen + 1 : dclose]
        zero_term = f"({zero_term})/({denom})"

    sign = "+" if random.random() < 0.5 else "-"
    return f"({expr}) {sign} {zero_term}"


def _split_signed_terms(expr: str) -> list[tuple[str, str]]:
    """Split a flat additive expression into (sign, body) pairs.

    Each sign is '+' or '-'.  The function respects parenthesis depth so that
    operators inside sub-expressions are not treated as term separators.
    """
    expr = expr.strip()
    terms: list[tuple[str, str]] = []
    depth = 0
    current: list[str] = []
    sign = "+"

    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif depth == 0 and ch in "+-" and i > 0 and expr[i - 1] not in "*/^(":
            # Binary additive operator: flush the current term.
            body = "".join(current).strip()
            if body:
                terms.append((sign, body))
            sign = ch
            current = []
        elif i == 0 and ch in "+-":
            # Leading sign of the very first term.
            sign = ch
        else:
            current.append(ch)

    body = "".join(current).strip()
    if body:
        terms.append((sign, body))
    return terms


def _join_signed_terms(terms: list[tuple[str, str]]) -> str:
    """Reconstruct a flat additive expression from (sign, body) pairs."""
    if not terms:
        return "0"
    pieces: list[str] = []
    for i, (sign, body) in enumerate(terms):
        if i == 0:
            pieces.append(f"-{body}" if sign == "-" else body)
        else:
            pieces.append(f" {sign} {body}")
    return "".join(pieces)


def scr_term_reorder(expr: str) -> str:
    """Shuffle the top-level additive terms of a fully-expanded expression."""
    terms = _split_signed_terms(expr)
    if len(terms) <= 1:
        return expr
    random.shuffle(terms)
    return _join_signed_terms(terms)


_SCRAMBLER_BY_NAME = {
    SCRAMBLE_MULTIPLY_ONE: lambda expr, N: scr_mul_by_one(expr, N),
    SCRAMBLE_WARD: lambda expr, N: scr_ward_substitute(expr, N),
    SCRAMBLE_MOMENTUM: lambda expr, N: scr_momentum_substitute(expr, N),
    SCRAMBLE_COMMUTE_DOT: lambda expr, N: scr_commute_dot(expr),
    SCRAMBLE_RATIO: lambda expr, N: scr_mul_by_ratio(expr, N),
    SCRAMBLE_PARTIAL_FRACTION: lambda expr, N: scr_partial_fraction(expr),
    SCRAMBLE_WARD_ALL: lambda expr, N: scr_ward_substitute_all(expr, N),
    SCRAMBLE_POLARISATION_ZERO: lambda expr, N: scr_add_polarisation_zero(expr, N),
    SCRAMBLE_TERM_REORDER: lambda expr, N: scr_term_reorder(expr),
}


def normalise_scramble_names(scramble_names: Sequence[str] | None = None) -> tuple[str, ...]:
    if scramble_names is None:
        return DEFAULT_SCRAMBLES
    names: list[str] = []
    for item in scramble_names:
        for name in str(item).split(","):
            name = name.strip()
            if not name:
                continue
            if name == "all":
                names.extend(_SCRAMBLER_BY_NAME)
                continue
            if name == "none":
                continue
            if name not in _SCRAMBLER_BY_NAME:
                allowed = ", ".join(("all", "none", *_SCRAMBLER_BY_NAME))
                raise ValueError(f"Unknown scramble {name!r}. Allowed: {allowed}")
            names.append(name)

    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _active_scramblers(scramble_names: Sequence[str] | None = None):
    return tuple(_SCRAMBLER_BY_NAME[name] for name in normalise_scramble_names(scramble_names))


def scramble(
    expr: str,
    N: int,
    *,
    min_scr: int = DEFAULT_MIN_SCR,
    max_scr: int = DEFAULT_MAX_SCR,
    max_len: int = DEFAULT_MAX_SCRAMBLED_LEN,
    full_expand: bool = False,
    scramble_names: Sequence[str] | None = None,
) -> str:
    min_scr = max(0, int(min_scr))
    max_scr = max(min_scr, int(max_scr))
    active_scramblers = _active_scramblers(scramble_names)
    out = full_expand_expression(expr) if full_expand else expr
    n_steps = random.randint(min_scr, max_scr) if max_scr > 0 and active_scramblers else 0
    for _ in range(n_steps):
        cand = random.choice(active_scramblers)(out, N)
        if full_expand:
            cand = full_expand_expression(cand)
        if len(cand) <= max_len:
            out = cand
    return full_expand_expression(out) if full_expand else out

__all__ = [
    'SCRAMBLE_MULTIPLY_ONE',
    'SCRAMBLE_WARD',
    'SCRAMBLE_MOMENTUM',
    'SCRAMBLE_COMMUTE_DOT',
    'SCRAMBLE_RATIO',
    'SCRAMBLE_PARTIAL_FRACTION',
    'SCRAMBLE_WARD_ALL',
    'SCRAMBLE_POLARISATION_ZERO',
    'SCRAMBLE_TERM_REORDER',
    'DEFAULT_SCRAMBLES',
    'DEFAULT_MIN_SCR',
    'DEFAULT_MAX_SCR',
    'DEFAULT_MAX_SCRAMBLED_LEN',
    'scr_mul_by_one',
    'scr_ward_substitute',
    'scr_momentum_substitute',
    'scr_commute_dot',
    'scr_mul_by_ratio',
    '_random_context_factor',
    '_find_denom_blocks',
    '_find_paren_block_ending_at',
    'scr_partial_fraction',
    'scr_ward_substitute_all',
    'scr_add_polarisation_zero',
    '_split_signed_terms',
    '_join_signed_terms',
    'scr_term_reorder',
    '_SCRAMBLER_BY_NAME',
    'normalise_scramble_names',
    '_active_scramblers',
    'scramble',
]
