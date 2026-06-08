"""notation — extracted from gen_data.py (scaffold, verbatim)."""

from __future__ import annotations
import re
from typing import Sequence

DOT = "·"


DEFAULT_MIN_TERMS = 1


DEFAULT_MAX_TERMS = 2


DEFAULT_USE_DENOMINATORS = True


DEFAULT_MAX_ATTEMPTS_FACTOR = 12


UNIT_PROBABILITY = 0.6


# SQED-specific knobs, neutralised for the colour-ordered gluon generator.
# (code paths kept dormant; can be ripped out once the gluon model is settled)
OLD_STYLE_PROBABILITY = 0.0


SPURIOUS_REPEAT_PROBABILITY = 0.0


DENOM_REPEAT_PROBABILITY = SPURIOUS_REPEAT_PROBABILITY  # backwards-compatible alias


SCALAR_POWER_PROBABILITY = 0.0


SCALAR_COEFF_POOL = [c for c in range(-9, 10) if c != 0]


TERM_COEFF_POOL = [c for c in range(-100, 101) if c != 0]


N4_BLOCK_WEIGHTS = {
    "singleF": 8,
    "doubleF": 2,
    "tr2": 1,
}


OLD_STYLE_N4_BLOCK_WEIGHTS = {
    "singleF": 8,
    "doubleF": 2,
    "tr2": 1,
}


GENERAL_BLOCK_WEIGHTS = {
    "singleF": 1,
    "tr2": 1,
    "doubleF": 2,
    "tr3": 1,
    "tripleF": 1,
    "tr4": 1,
}


def dot(a: str, b: str) -> str:
    return f"{a} {DOT} {b}"


def p(i: int) -> str:
    return f"p_{i}"


def e(i: int) -> str:
    return f"e_{i}"


def F(i: int) -> str:
    return f"F_{i}"


def Tr(*Fs: str) -> str:
    return "Tr(" + f" {DOT} ".join(Fs) + ")"


def _vec(tag: str, idx: int) -> str:
    return f"{tag}_{idx}"


def gluon_legs(N: int) -> list[int]:
    """All N external legs are colour-ordered gluons (each owns e_i and F_i)."""
    return list(range(1, N + 1))


_RE_pp = re.compile(r"p_(\d+)\s*·\s*p_(\d+)")


_RE_pFchainp = re.compile(r"p_(\d+)((?:\s*·\s*F_\d+)+)\s*·\s*p_(\d+)")


_RE_TrN = re.compile(r"Tr\((F_\d+(?:\s*·\s*F_\d+)*)\)")


_RE_DOT = re.compile(r"(p_\d+|e_\d+)\s*·\s*(p_\d+|e_\d+)")


_TOKEN_RE = re.compile(
    r"\s*(\d+\.\d+|\d+|p_\d+|e_\d+|F_\d+|Tr\b|\*\*|\^|\+|\-|\*|/|\(|\)|\.|·|,)"
)


def _format_signed_sum(terms: list[tuple[int, str]]) -> str:
    pieces: list[str] = []
    for sign, term in terms:
        if not pieces:
            pieces.append(term if sign > 0 else f"-{term}")
        else:
            pieces.append(("+ " if sign > 0 else "- ") + term)
    return "(" + " ".join(pieces) + ")"


def _split_top_level(s: str, sep: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _format_poly(terms: Sequence[str], coeffs: Sequence[int]) -> str:
    pieces: list[str] = []
    for i, (term, coeff) in enumerate(zip(terms, coeffs)):
        abs_coeff = abs(coeff)
        sign = "-" if coeff < 0 else "+"
        core = term if abs_coeff == 1 else f"{abs_coeff}*{term}"
        if i == 0:
            pieces.append(f"-{core}" if sign == "-" else core)
        else:
            pieces.append(f" {sign} {core}")
    return "".join(pieces)


def _strip_matched_outer_parens(s: str) -> str:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return s
    return s[1:-1].strip()

__all__ = [
    'DOT',
    'dot',
    'p',
    'e',
    'F',
    'Tr',
    '_vec',
    'gluon_legs',
    '_RE_pp',
    '_RE_pFchainp',
    '_RE_TrN',
    '_RE_DOT',
    '_TOKEN_RE',
    '_format_signed_sum',
    '_format_poly',
    '_split_top_level',
    '_strip_matched_outer_parens',
    'UNIT_PROBABILITY',
    'OLD_STYLE_PROBABILITY',
    'SPURIOUS_REPEAT_PROBABILITY',
    'DENOM_REPEAT_PROBABILITY',
    'SCALAR_POWER_PROBABILITY',
    'SCALAR_COEFF_POOL',
    'TERM_COEFF_POOL',
    'N4_BLOCK_WEIGHTS',
    'OLD_STYLE_N4_BLOCK_WEIGHTS',
    'GENERAL_BLOCK_WEIGHTS',
    'DEFAULT_MIN_TERMS',
    'DEFAULT_MAX_TERMS',
    'DEFAULT_USE_DENOMINATORS',
    'DEFAULT_MAX_ATTEMPTS_FACTOR',
]
