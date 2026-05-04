#!/usr/bin/env python3
"""
gen_data.py — Build (simple, scrambled) training pairs for N-point amplitudes.

Conventions
-----------
    p_1, p_N          : massive scalars
    p_2 … p_{N-1}     : massless photons (each owns e_i, F_i)

"simple"    = a small sum of gauge-invariant terms built from
              Tr(F·…·F), p·F·…·F·p, and optional p·p factors.
"scrambled" = the same expression expanded into e·p / e·e / p·p factors,
              then hit by algebraic identities that preserve numerical
              equality.

The implementation is aimed first at 4pt and 5pt scalar-QED-like data, but
the core monomial generation and evaluator work for generic N.
"""
from __future__ import annotations

import argparse
import ast as _ast
import csv
import json
import math
import multiprocessing as mp
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from kinematics import generate_kinematics, mdot

DOT = "·"

# ─────────────────────────────────────────────────────────────────────────────
# Editable defaults
# ─────────────────────────────────────────────────────────────────────────────
# The command-line interface uses these values as defaults.  For day-to-day
# generation you can usually edit this block only, then run
#
#     python3 gen_data.py
#
# with no extra flags.  CLI flags still exist and override these values when
# provided.
def knum(num: int):
        if num > 1000:
            return f"{num//1000}k"
        else:
            return f"{num}"
# Output / run defaults
DEFAULT_N_PARTICLES = 4
DEFAULT_SAMPLES = 500000
DEFAULT_SEED = 42
DEFAULT_MASS = 2.0
# Number of samples expressed in thousands (integer)
NSAMPS = DEFAULT_SAMPLES // 1000
DEFAULT_RAW_OUT_TEMPLATE = "gi_{N}pt_{NSAMPS}k.csv"
DEFAULT_TOK_OUT_TEMPLATE = "gi_{N}pt_tok_{NSAMPS}k.csv"
DEFAULT_LOG_OUT_TEMPLATE = "gen_data_{N}pt_{NSAMPS}k.log"
DEFAULT_VALIDATE = True
DEFAULT_TOKENISE = True
DEFAULT_FULL_EXPAND_SCRAMBLED = True
DEFAULT_OVERSAMPLE_FACTOR = 1.2

# Shape of generated expressions
DEFAULT_MIN_TERMS = 1
DEFAULT_MAX_TERMS = 5
DEFAULT_USE_DENOMINATORS = True
DEFAULT_MAX_ATTEMPTS_FACTOR = 12

# Scrambling defaults
SCRAMBLE_MULTIPLY_ONE = "multiply_one"
SCRAMBLE_WARD = "ward"
SCRAMBLE_MOMENTUM = "momentum"
SCRAMBLE_COMMUTE_DOT = "commute_dot"
SCRAMBLE_RATIO = "ratio"
SCRAMBLE_PARTIAL_FRACTION = "partial_fraction"
DEFAULT_SCRAMBLES = (
    SCRAMBLE_MULTIPLY_ONE,
    SCRAMBLE_WARD,
    SCRAMBLE_MOMENTUM,
    SCRAMBLE_COMMUTE_DOT,
    SCRAMBLE_RATIO,
    SCRAMBLE_PARTIAL_FRACTION,
)
DEFAULT_MIN_SCR = 1
DEFAULT_MAX_SCR = 5
DEFAULT_MAX_SCRAMBLED_LEN = 4000

# Distribution knobs.
# UNIT_PROBABILITY chooses top-level coefficients from {-1,+1}.
# OLD_STYLE_PROBABILITY biases a sample toward the older two-term / single-F
# dominated style without making the whole dataset old-style.
# SPURIOUS_REPEAT_PROBABILITY controls hidden p·F-expansion spurious repeated
# denominators such as (p_i·p_j)^2.  Keep this nonzero but subdominant.
# SCALAR_POWER_PROBABILITY controls explicit numerator scalar powers such as
# (p_i·p_j)^2.
UNIT_PROBABILITY = 0.2
OLD_STYLE_PROBABILITY = 0.25
SPURIOUS_REPEAT_PROBABILITY = 0.15
DENOM_REPEAT_PROBABILITY = SPURIOUS_REPEAT_PROBABILITY  # backwards-compatible alias
SCALAR_POWER_PROBABILITY = 0.15

# Coefficient pools used outside unit-coefficient mode.
SCALAR_COEFF_POOL = [c for c in range(-9, 10) if c != 0]
TERM_COEFF_POOL = [c for c in range(-100, 101) if c != 0]

# Block-choice weights.  These are intentionally editable at the top because
# they control how often traces and multi-F chains appear.
# At four points, lower tr2 relative to singleF to avoid trace-heavy data.
N4_BLOCK_WEIGHTS = {
    "singleF": 5,
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

# Parallel generation defaults
DEFAULT_BATCH_SIZE = 1000
DEFAULT_JOBS = "auto"  # integer as string, or "auto"
DEFAULT_PROGRESS = True

# Tokenisation defaults
DEFAULT_TOKENIZER_MAX_PARTICLES = 8


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


def photon_legs(N: int) -> list[int]:
    return list(range(2, N))


def scalar_legs(N: int) -> tuple[int, int]:
    return (1, N)


@dataclass(frozen=True)
class BlockSpec:
    kind: str
    photons: tuple[int, ...]
    left: int | None = None
    right: int | None = None


@dataclass(frozen=True)
class MonomialSpec:
    numerator: str
    blocks: tuple[BlockSpec, ...]
    scalar_pairs: int
    numerator_mass_dim: int


@dataclass(frozen=True)
class BatchJob:
    N: int
    num_samples: int
    max_scr: int
    min_scr: int
    seed: int | None
    unit_probability: float
    old_style_probability: float
    denom_repeat_probability: float
    scalar_power_probability: float
    use_denominators: bool
    validate: bool
    M: float
    min_terms: int
    max_terms: int
    max_attempts_factor: int
    full_expand_scrambled: bool


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


def _rw_pFchainp(*idxs: int) -> str:
    """Expand p_i · F_{j1} · … · F_{jn} · p_k into dot products."""
    i, *Fs, k = idxs
    terms: list[tuple[int, str]] = []
    for mask in range(1 << len(Fs)):
        sign = 1 if bin(mask).count("1") % 2 == 0 else -1
        factors: list[str] = []
        prev = ("p", i)
        for bit, j in enumerate(Fs):
            swap = (mask >> bit) & 1
            left = ("e", j) if swap else ("p", j)
            right = ("p", j) if swap else ("e", j)
            factors.append(f"({dot(_vec(*prev), _vec(*left))})")
            prev = right
        factors.append(f"({dot(_vec(*prev), p(k))})")
        terms.append((sign, "*".join(factors)))
    return _format_signed_sum(terms)


def _rw_TrN(*js: int) -> str:
    """Expand Tr(F_{j1} · … · F_{jn}) into dot products."""
    terms: list[tuple[int, str]] = []
    for mask in range(1 << len(js)):
        sign = 1 if bin(mask).count("1") % 2 == 0 else -1
        pairs: list[tuple[str, str]] = []
        for bit, j in enumerate(js):
            if (mask >> bit) & 1:
                pairs.append((e(j), p(j)))
            else:
                pairs.append((p(j), e(j)))
        factors = [
            f"({dot(pairs[i][1], pairs[(i + 1) % len(js)][0])})"
            for i in range(len(js))
        ]
        terms.append((sign, "*".join(factors)))
    return _format_signed_sum(terms)


def rewrite_gi(block: str) -> str:
    block = block.strip()

    m = _RE_TrN.fullmatch(block)
    if m:
        js = [int(x) for x in re.findall(r"F_(\d+)", m.group(1))]
        return _rw_TrN(*js)

    m = _RE_pFchainp.fullmatch(block)
    if m:
        i = int(m.group(1))
        Fs = [int(x) for x in re.findall(r"F_(\d+)", m.group(2))]
        k = int(m.group(3))
        return _rw_pFchainp(i, *Fs, k)

    if _RE_pp.fullmatch(block):
        return block
    return block


def _canon_pp(term: str) -> str:
    m = _RE_pp.fullmatch(term)
    if not m:
        return term
    i, j = sorted(map(int, m.groups()))
    return dot(p(i), p(j))


def _canon_TrN(term: str) -> str:
    m = _RE_TrN.fullmatch(term)
    if not m:
        return term
    js = [int(x) for x in re.findall(r"F_(\d+)", m.group(1))]
    best = js[:]
    for shift in range(1, len(js)):
        rot = js[shift:] + js[:shift]
        if rot < best:
            best = rot
    return Tr(*(F(j) for j in best))


def _factor_sort_key(term: str) -> tuple:
    term = term.strip()
    if _RE_TrN.fullmatch(term):
        js = re.findall(r"F_(\d+)", term)
        return (0, len(js), _canon_TrN(term))
    if _RE_pFchainp.fullmatch(term):
        js = re.findall(r"F_(\d+)", term)
        return (1, len(js), term)
    if _RE_pp.fullmatch(term):
        return (2, 0, _canon_pp(term))
    return (3, 0, term)


def canonicalise_gi_product(prod: str) -> str:
    factors = [f.strip() for f in prod.split("*") if f.strip()]
    canon: list[str] = []
    for factor in factors:
        if _RE_TrN.fullmatch(factor):
            factor = _canon_TrN(factor)
        elif _RE_pp.fullmatch(factor):
            factor = _canon_pp(factor)
        canon.append(factor)
    canon.sort(key=_factor_sort_key)
    
    # Combine duplicate factors into powers
    factor_counts: dict[str, int] = {}
    for factor in canon:
        factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    result_parts: list[str] = []
    for factor in sorted(factor_counts.keys(), key=_factor_sort_key):
        count = factor_counts[factor]
        if count == 1:
            result_parts.append(factor)
        else:
            result_parts.append(f"({factor})^{count}")
    
    return "*".join(result_parts)


def canonicalise_denominator(den: str) -> str:
    if not den:
        return den
    factors = [_canon_pp(f.strip()) for f in den.split("*") if f.strip()]
    
    # Combine duplicate factors into powers
    factor_counts: dict[str, int] = {}
    for factor in factors:
        factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    result_parts: list[str] = []
    for factor in sorted(factor_counts.keys()):
        count = factor_counts[factor]
        if count == 1:
            result_parts.append(factor)
        else:
            result_parts.append(f"({factor})^{count}")
    
    return "*".join(result_parts)


def expand_simple_term(simple_term: str) -> str:
    """Expand one GI term into e·p / e·e / p·p form."""
    simple_term = simple_term.strip()
    if "/" in simple_term:
        num, den = simple_term.split("/", 1)
        num = num.strip()
        den = den.strip()
        if num.startswith("(") and num.endswith(")"):
            num = num[1:-1]
        if den.startswith("(") and den.endswith(")"):
            den = den[1:-1]
        exp_num = "*".join(rewrite_gi(f) for f in num.split("*"))
        return f"({exp_num})/({den})"
    return "*".join(rewrite_gi(f) for f in simple_term.split("*"))


def expand_simple_expression(simple_expr: str) -> str:
    """Expand a whole polynomial by expanding its GI terms term-by-term."""
    toks = _tokenize(simple_expr)
    tree = _Parser(toks).parse()
    return _ast_to_infix(_expand_ast(tree))


@dataclass
class _RatTerm:
    coeff: float
    numerators: list[str]
    denominators: list[str]


def _is_zero_coeff(value: float) -> bool:
    return abs(value) < 1e-14


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return repr(value)


def _format_factor(expr: str) -> str:
    expr = _strip_matched_outer_parens(expr.strip())
    if not expr:
        return expr
    if any(op in expr for op in (" + ", " - ")):
        return f"({expr})"
    return expr


def _node_to_factor(node) -> str:
    return _format_factor(_ast_to_infix(node))


def _negate_terms(terms: list[_RatTerm]) -> list[_RatTerm]:
    return [_RatTerm(-term.coeff, term.numerators[:], term.denominators[:]) for term in terms]


def _mul_terms(left: list[_RatTerm], right: list[_RatTerm]) -> list[_RatTerm]:
    out: list[_RatTerm] = []
    for a in left:
        for b in right:
            coeff = a.coeff * b.coeff
            if _is_zero_coeff(coeff):
                continue
            out.append(
                _RatTerm(
                    coeff,
                    a.numerators[:] + b.numerators[:],
                    a.denominators[:] + b.denominators[:],
                )
            )
    return out


def _divide_terms(left: list[_RatTerm], right_node) -> list[_RatTerm]:
    right_terms = _rational_terms(right_node)
    out: list[_RatTerm] = []

    if len(right_terms) == 1:
        denom = right_terms[0]
        if _is_zero_coeff(denom.coeff):
            return []
        for term in left:
            out.append(
                _RatTerm(
                    term.coeff / denom.coeff,
                    term.numerators[:] + denom.denominators[:],
                    term.denominators[:] + denom.numerators[:],
                )
            )
        return out

    denom_factor = _node_to_factor(right_node)
    for term in left:
        out.append(
            _RatTerm(
                term.coeff,
                term.numerators[:],
                term.denominators[:] + [denom_factor],
            )
        )
    return out


def _rational_terms(node) -> list[_RatTerm]:
    """Convert an expanded AST to a flat sum of rational monomial terms."""
    if isinstance(node, _Num):
        if _is_zero_coeff(node.value):
            return []
        return [_RatTerm(node.value, [], [])]

    if isinstance(node, tuple) and node[0] == _DOT_TAG:
        return [_RatTerm(1.0, [_node_to_factor(node)], [])]

    if isinstance(node, _Vec):
        return [_RatTerm(1.0, [_node_to_factor(node)], [])]

    if isinstance(node, _UnaryOp):
        terms = _rational_terms(node.operand)
        return _negate_terms(terms) if node.op == "-" else terms

    if isinstance(node, _BinOp):
        if node.op == "+":
            return _rational_terms(node.left) + _rational_terms(node.right)
        if node.op == "-":
            return _rational_terms(node.left) + _negate_terms(_rational_terms(node.right))
        if node.op == "*":
            return _mul_terms(_rational_terms(node.left), _rational_terms(node.right))
        if node.op == "/":
            return _divide_terms(_rational_terms(node.left), node.right)
        if node.op == "**":
            return [_RatTerm(1.0, [_node_to_factor(node)], [])]

    return [_RatTerm(1.0, [_node_to_factor(node)], [])]


def _format_rational_term(term: _RatTerm) -> tuple[str, str]:
    sign = "-" if term.coeff < 0 else "+"
    abs_coeff = abs(term.coeff)
    factors = [_format_factor(factor) for factor in term.numerators if factor]
    if abs(abs_coeff - 1.0) > 1e-12 or not factors:
        factors.insert(0, _format_number(abs_coeff))

    numerator = "*".join(factors)
    if term.denominators:
        denominator = "*".join(_format_factor(factor) for factor in term.denominators if factor)
        body = f"({numerator})/({denominator})"
    else:
        body = numerator
    return sign, body


def full_expand_expression(expr: str) -> str:
    """Fully distribute an expression into a flat sum of rational dot-product terms.

    This keeps the expression in p/e dot-product form; it does not recover F-blocks.
    """
    toks = _tokenize(expr)
    tree = _Parser(toks).parse()
    expanded = _expand_ast(tree)
    terms = [term for term in _rational_terms(expanded) if not _is_zero_coeff(term.coeff)]
    if not terms:
        return "0"

    pieces: list[str] = []
    for term in terms:
        sign, body = _format_rational_term(term)
        if not pieces:
            pieces.append(body if sign == "+" else f"-{body}")
        else:
            pieces.append(f" {sign} {body}")
    return "".join(pieces)


class _Num:
    __slots__ = ("value",)

    def __init__(self, value: float):
        self.value = float(value)


class _Vec:
    __slots__ = ("tag", "idx")

    def __init__(self, tag: str, idx: int):
        self.tag = tag
        self.idx = idx


class _DotChain:
    __slots__ = ("parts",)
    _TR = object()

    def __init__(self, parts: list):
        self.parts = parts


class _BinOp:
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


class _UnaryOp:
    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand):
        self.op = op
        self.operand = operand


_DOT_TAG = "DOT"
_VEC_TAG = "VEC"


class _Parser:
    """Recursive-descent parser for amplitude expressions."""

    def __init__(self, toks: list[str]):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def pop(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        return self._expr()

    def _expr(self):
        node = self._term()
        while self.peek() in ("+", "-"):
            node = _BinOp(self.pop(), node, self._term())
        return node

    def _term(self):
        node = self._factor()
        while self.peek() in ("*", "/"):
            node = _BinOp(self.pop(), node, self._factor())
        return node

    def _factor(self):
        if self.peek() == "+":
            self.pop()
            return self._factor()
        if self.peek() == "-":
            self.pop()
            return _UnaryOp("-", self._factor())
        return self._power()

    def _power(self):
        node = self._primary()
        while self.peek() == "**":
            self.pop()
            node = _BinOp("**", node, self._primary())
        return node

    def _primary(self):
        tok = self.peek()
        if tok == "(":
            self.pop()
            node = self._expr()
            if self.peek() == ")":
                self.pop()
            return node

        if tok == "Tr":
            self.pop()
            if self.peek() == "(":
                self.pop()
                parts = self._parse_F_chain()
                if self.peek() == ")":
                    self.pop()
                vecs = [_Vec("F", int(re.search(r"\d+", t).group())) for t in parts]
                return _DotChain(vecs + [_DotChain._TR])
            return _Num(0.0)

        parts: list[_Vec] = []
        while True:
            current = self.peek()
            if current and re.match(r"(p_|e_|F_)\d+", str(current)):
                m = re.match(r"(p_|e_|F_)(\d+)", current)
                parts.append(_Vec(m.group(1)[0], int(m.group(2))))
                self.pop()
                if self.peek() in ("·", "."):
                    self.pop()
                    continue
                break
            break
        if parts:
            return parts[0] if len(parts) == 1 else _DotChain(parts)

        if tok and re.match(r"\d+(?:\.\d+)?", str(tok)):
            self.pop()
            return _Num(float(tok))

        if tok is not None:
            self.pop()
        return _Num(0.0)

    def _parse_F_chain(self) -> list[str]:
        out: list[str] = []
        while True:
            tok = self.peek()
            if tok and re.match(r"F_\d+", str(tok)):
                out.append(self.pop())
                if self.peek() in ("·", ".", ","):
                    self.pop()
                    continue
                break
            break
        return out


def _tokenize(expr: str) -> list[str]:
    expr = expr.replace("^", "**")
    pos = 0
    toks: list[str] = []
    while pos < len(expr):
        match = _TOKEN_RE.match(expr, pos)
        if match:
            toks.append(match.group(1))
            pos = match.end()
            continue
        match = re.match(r"\s*([A-Za-z_]\w*)", expr[pos:])
        if match:
            toks.append(match.group(1))
            pos += match.end()
        else:
            pos += 1
    return toks


def _mk_dot(lhs, rhs):
    return (_DOT_TAG, lhs, rhs)


def _vn(tag: str, idx: int):
    return (_VEC_TAG, tag, idx)


def _ast_add(a, b):
    return _BinOp("+", a, b)


def _ast_sub(a, b):
    return _BinOp("-", a, b)


def _ast_mul(a, b):
    return _BinOp("*", a, b)


def _expand_dotchain(dc: _DotChain):
    parts = dc.parts

    if parts and parts[-1] is _DotChain._TR:
        Fs = [v for v in parts[:-1] if isinstance(v, _Vec)]
        labels = [v.idx for v in Fs]
        acc = None
        for mask in range(1 << len(labels)):
            sign = 1 if bin(mask).count("1") % 2 == 0 else -1
            pairs: list[tuple[str, int, str, int]] = []
            for bit, lab in enumerate(labels):
                if (mask >> bit) & 1:
                    pairs.append(("e", lab, "p", lab))
                else:
                    pairs.append(("p", lab, "e", lab))
            factors = [
                _mk_dot(
                    _vn(pairs[i][2], pairs[i][3]),
                    _vn(pairs[(i + 1) % len(labels)][0], pairs[(i + 1) % len(labels)][1]),
                )
                for i in range(len(labels))
            ]
            term = factors[0]
            for factor in factors[1:]:
                term = _ast_mul(term, factor)
            if acc is None:
                acc = term if sign > 0 else _UnaryOp("-", term)
            else:
                acc = _ast_add(acc, term) if sign > 0 else _ast_sub(acc, term)
        return acc

    F_vecs = [v for v in parts if isinstance(v, _Vec) and v.tag == "F"]
    if F_vecs:
        if not (
            isinstance(parts[0], _Vec)
            and parts[0].tag == "p"
            and isinstance(parts[-1], _Vec)
            and parts[-1].tag == "p"
        ):
            return _Num(0.0)
        left_idx = parts[0].idx
        right_idx = parts[-1].idx
        labels = [v.idx for v in F_vecs]
        acc = None
        for mask in range(1 << len(labels)):
            sign = 1 if bin(mask).count("1") % 2 == 0 else -1
            prev = ("p", left_idx)
            factors = []
            for bit, lab in enumerate(labels):
                swap = (mask >> bit) & 1
                lhs = ("e", lab) if swap else ("p", lab)
                rhs = ("p", lab) if swap else ("e", lab)
                factors.append(_mk_dot(_vn(*prev), _vn(*lhs)))
                prev = rhs
            factors.append(_mk_dot(_vn(*prev), _vn("p", right_idx)))
            term = factors[0]
            for factor in factors[1:]:
                term = _ast_mul(term, factor)
            if acc is None:
                acc = term if sign > 0 else _UnaryOp("-", term)
            else:
                acc = _ast_add(acc, term) if sign > 0 else _ast_sub(acc, term)
        return acc

    acc = None
    for left, right in zip(parts, parts[1:]):
        term = _mk_dot(_vn(left.tag, left.idx), _vn(right.tag, right.idx))
        acc = term if acc is None else _ast_mul(acc, term)
    return acc if acc is not None else _Num(0.0)


def _expand_ast(node):
    if isinstance(node, (_Num, _Vec)):
        return node
    if isinstance(node, _DotChain):
        return _expand_dotchain(node)
    if isinstance(node, _UnaryOp):
        return _UnaryOp(node.op, _expand_ast(node.operand))
    if isinstance(node, _BinOp):
        return _BinOp(node.op, _expand_ast(node.left), _expand_ast(node.right))
    if isinstance(node, tuple):
        return node
    return node


def eval_infix_numeric(expr: str, momenta, pols) -> float:
    """Evaluate an infix expression on explicit kinematics."""
    N = len(momenta)
    P = {f"p_{i}": momenta[i - 1] for i in range(1, N + 1)}
    E = {f"e_{i}": pols[i - 2] for i in range(2, N)}

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
                if va is not None and vb is not None:
                    return mdot(va, vb)
            return 0.0
        return 0.0

    tree = _Parser(_tokenize(expr)).parse()
    expanded = _expand_ast(tree)
    return float(_eval(expanded))


def _prec(node) -> int:
    if isinstance(node, _Num):
        return 100
    if isinstance(node, tuple) and node[0] == _DOT_TAG:
        return 90
    if isinstance(node, _UnaryOp):
        return 80
    if isinstance(node, _BinOp):
        return {"+": 10, "-": 10, "*": 20, "/": 20, "**": 30}.get(node.op, 0)
    return 100


def _vec_name(node) -> str:
    return f"{node[1]}_{node[2]}"


def _ast_to_infix(node, parent_prec: int = 0, is_right: bool = False) -> str:
    if isinstance(node, _Num):
        if abs(node.value - round(node.value)) < 1e-12:
            return str(int(round(node.value)))
        return repr(node.value)
    if isinstance(node, tuple) and node[0] == _DOT_TAG:
        return f"{_vec_name(node[1])} {DOT} {_vec_name(node[2])}"
    if isinstance(node, _UnaryOp):
        inner = _ast_to_infix(node.operand, _prec(node))
        if _prec(node.operand) < _prec(node):
            inner = f"({inner})"
        return f"-{inner}"
    if isinstance(node, _BinOp):
        cur = _prec(node)
        left = _ast_to_infix(node.left, cur, False)
        right = _ast_to_infix(node.right, cur, True)
        if _prec(node.left) < cur:
            left = f"({left})"
        right_needs = _prec(node.right) < cur or (
            is_right and node.op in ("-", "/", "**") and _prec(node.right) == cur
        )
        if node.op in ("-", "/", "**") and _prec(node.right) == cur:
            right_needs = True
        if right_needs:
            right = f"({right})"
        join = f" {node.op} " if node.op in ("+", "-") else node.op
        out = f"{left}{join}{right}"
        if cur < parent_prec:
            out = f"({out})"
        return out
    return "0"


def _chain_endpoints(Fs: Sequence[int], N: int) -> tuple[int, int]:
    """Choose endpoints for p·F...F·p chains without over-restricting them.

    Gauge invariance only forces the immediately adjacent contractions
    p_j·F_j and F_j·p_j to vanish for a massless transverse photon j.  Older
    data sources also allow endpoints to be photons appearing elsewhere inside
    the chain, and allow the two endpoints to be the same momentum.  Therefore
    we exclude only the first photon from the left endpoint and only the last
    photon from the right endpoint.
    """
    if not Fs:
        raise ValueError("F-chain must contain at least one photon")
    first, last = Fs[0], Fs[-1]
    left_pool = [x for x in range(1, N + 1) if x != first]
    right_pool = [x for x in range(1, N + 1) if x != last]
    return random.choice(left_pool), random.choice(right_pool)


def _singleF_block(j: int, N: int) -> tuple[str, BlockSpec]:
    left, right = _chain_endpoints((j,), N)
    return (
        f"{p(left)} {DOT} {F(j)} {DOT} {p(right)}",
        BlockSpec("chain", (j,), left, right),
    )


def _doubleF_block(j: int, k: int, N: int) -> tuple[str, BlockSpec]:
    left, right = _chain_endpoints((j, k), N)
    return (
        f"{p(left)} {DOT} {F(j)} {DOT} {F(k)} {DOT} {p(right)}",
        BlockSpec("chain", (j, k), left, right),
    )


def _tripleF_block(j: int, k: int, l: int, N: int) -> tuple[str, BlockSpec]:
    left, right = _chain_endpoints((j, k, l), N)
    return (
        f"{p(left)} {DOT} {F(j)} {DOT} {F(k)} {DOT} {F(l)} {DOT} {p(right)}",
        BlockSpec("chain", (j, k, l), left, right),
    )


def _tr2_block(j: int, k: int) -> tuple[str, BlockSpec]:
    return Tr(F(j), F(k)), BlockSpec("trace", (j, k))


def _tr3_block(j: int, k: int, l: int) -> tuple[str, BlockSpec]:
    return Tr(F(j), F(k), F(l)), BlockSpec("trace", (j, k, l))


def _tr4_block(j: int, k: int, l: int, m: int) -> tuple[str, BlockSpec]:
    return Tr(F(j), F(k), F(l), F(m)), BlockSpec("trace", (j, k, l, m))


def _scalar_pp_factor(N: int) -> str:
    i, j = random.sample(range(1, N + 1), 2)
    return dot(p(i), p(j))


def _block_mass_dimension(block: BlockSpec) -> int:
    """Mass dimension with [p]=1 and [F]=1, matching the paper's counting."""
    if block.kind == "trace":
        return len(block.photons)
    if block.kind == "chain":
        return len(block.photons) + 2
    raise ValueError(f"Unknown block kind {block.kind}")


def _all_physical_poles(N: int) -> list[str]:
    pool: list[str] = []
    seen: set[str] = set()
    for photon in photon_legs(N):
        for scalar_leg in scalar_legs(N):
            term = _canon_pp(dot(p(photon), p(scalar_leg)))
            if term not in seen:
                seen.add(term)
                pool.append(term)
        for other in photon_legs(N):
            if other == photon:
                continue
            term = _canon_pp(dot(p(photon), p(other)))
            if term not in seen:
                seen.add(term)
                pool.append(term)
    return pool


def _required_denominator_count(numerator_mass_dim: int, N: int) -> int | None:
    target_dim = 4 - N
    delta = numerator_mass_dim - target_dim
    if delta < 0 or delta % 2 != 0:
        return None
    return delta // 2


def _weighted_choice(weight_map: dict[str, int]) -> str:
    choices: list[str] = []
    for key, weight in weight_map.items():
        if weight > 0:
            choices.extend([key] * int(weight))
    if not choices:
        raise ValueError("At least one block-choice weight must be positive")
    return random.choice(choices)


def _block_choice_weights(N: int, remaining_count: int, *, old_style_blocks: bool) -> dict[str, int]:
    """Return editable block weights compatible with the remaining photons."""
    if N == 4:
        base = OLD_STYLE_N4_BLOCK_WEIGHTS if old_style_blocks else N4_BLOCK_WEIGHTS
    else:
        base = GENERAL_BLOCK_WEIGHTS

    allowed = {"singleF"}
    if remaining_count >= 2:
        allowed.update({"tr2", "doubleF"})
    if remaining_count >= 3:
        allowed.update({"tr3", "tripleF"})
    if remaining_count >= 4:
        allowed.add("tr4")
    return {kind: int(weight) for kind, weight in base.items() if kind in allowed and int(weight) > 0}


def _generate_gi_monomial_spec(
    N: int,
    *,
    old_style_blocks: bool = False,
    scalar_power_probability: float = SCALAR_POWER_PROBABILITY,
) -> MonomialSpec:
    remaining = photon_legs(N)
    random.shuffle(remaining)
    factors: list[str] = []
    blocks: list[BlockSpec] = []

    while remaining:
        r = len(remaining)
        kind = _weighted_choice(_block_choice_weights(N, r, old_style_blocks=old_style_blocks))

        if kind == "tr4":
            chosen = random.sample(remaining, 4)
            block_str, spec = _tr4_block(*chosen)
        elif kind == "tr3":
            chosen = random.sample(remaining, 3)
            block_str, spec = _tr3_block(*chosen)
        elif kind == "tripleF":
            chosen = random.sample(remaining, 3)
            block_str, spec = _tripleF_block(*chosen, N)
        elif kind == "tr2":
            chosen = random.sample(remaining, 2)
            block_str, spec = _tr2_block(*chosen)
        elif kind == "doubleF":
            chosen = random.sample(remaining, 2)
            block_str, spec = _doubleF_block(*chosen, N)
        else:
            chosen = [remaining[-1]]
            block_str, spec = _singleF_block(chosen[0], N)

        factors.append(block_str)
        blocks.append(spec)
        remaining = [x for x in remaining if x not in chosen]

    base_mass_dim = sum(_block_mass_dimension(block) for block in blocks)
    max_scalar_pairs = 2 if N <= 5 else 4
    min_chain_poles = sum(len(block.photons) for block in blocks if block.kind == "chain")
    pole_pool_size = len(_all_physical_poles(N))
    candidates: list[tuple[int, int]] = []
    for scalar_pairs in range(max_scalar_pairs + 1):
        numerator_mass_dim = base_mass_dim + 2 * scalar_pairs
        denom_count = _required_denominator_count(numerator_mass_dim, N)
        if denom_count is None:
            continue
        if denom_count < min_chain_poles:
            continue
        if denom_count > pole_pool_size:
            continue
        candidates.append((scalar_pairs, numerator_mass_dim))
    if not candidates:
        raise ValueError(f"Could not realise manifest dimension 4-{N} with current ansatz")

    scalar_pairs, numerator_mass_dim = random.choice(candidates)

    # Add optional scalar p·p numerator factors.  With nonzero
    # scalar_power_probability, preferentially repeat an existing physical pole
    # factor.  After canonicalisation this deliberately creates numerator powers
    # such as (p_2 · p_4)^2, which can then support a spurious repeated
    # denominator factor without creating a physical double pole.
    scalar_power_probability = max(0.0, min(1.0, scalar_power_probability))
    scalar_factors: list[str] = []
    physical_scalar_pool = _all_physical_poles(N)
    for _ in range(scalar_pairs):
        repeatable = [term for term in scalar_factors if term in physical_scalar_pool]
        if repeatable and random.random() < scalar_power_probability:
            scalar_factors.append(random.choice(repeatable))
            continue
        if physical_scalar_pool and random.random() < 0.75:
            scalar_factors.append(random.choice(physical_scalar_pool))
        else:
            scalar_factors.append(_canon_pp(_scalar_pp_factor(N)))
    factors.extend(scalar_factors)

    random.shuffle(factors)
    return MonomialSpec(
        numerator=canonicalise_gi_product("*".join(factors)),
        blocks=tuple(blocks),
        scalar_pairs=scalar_pairs,
        numerator_mass_dim=numerator_mass_dim,
    )


def strict_gi_monomial(N: int) -> str:
    """Public helper retained for tests/backwards compatibility."""
    return _generate_gi_monomial_spec(N).numerator


def _photon_pole_choices(
    photon: int,
    N: int,
    *,
    preferred: Sequence[int] = (),
) -> list[str]:
    choices: list[str] = []
    seen: set[str] = set()

    def add(a: int, b: int) -> None:
        if a == b:
            return
        term = _canon_pp(dot(p(a), p(b)))
        if term not in seen:
            seen.add(term)
            choices.append(term)

    for leg in preferred:
        if leg != photon:
            add(photon, leg)

    for scalar_leg in scalar_legs(N):
        if scalar_leg != photon:
            add(photon, scalar_leg)

    for other in photon_legs(N):
        if other != photon:
            add(photon, other)
    return choices


def _explicit_scalar_pp_counts(product: str) -> dict[str, int]:
    """Count manifest scalar p·p factors in a canonical GI product.

    Only these factors can cancel an extra repeated denominator factor.  We do
    *not* treat p·p factors that merely appear somewhere after expanding a
    p·F...F·p chain as cancellations, because those are not guaranteed to be
    common factors of the whole numerator.
    """
    counts: dict[str, int] = {}
    for raw in _split_top_level(product, "*"):
        factor = raw.strip()
        if not factor:
            continue

        power_match = re.fullmatch(r"\((p_\d+\s*·\s*p_\d+)\)\^(\d+)", factor)
        if power_match:
            term = _canon_pp(power_match.group(1))
            counts[term] = counts.get(term, 0) + int(power_match.group(2))
            continue

        stripped = _strip_matched_outer_parens(factor)
        if stripped != factor and "*" in stripped:
            for term, multiplicity in _explicit_scalar_pp_counts(stripped).items():
                counts[term] = counts.get(term, 0) + multiplicity
            continue

        if _RE_pp.fullmatch(stripped):
            term = _canon_pp(stripped)
            counts[term] = counts.get(term, 0) + 1
    return counts


def _chain_expansion_spurious_pp_counts(spec: MonomialSpec, N: int) -> dict[str, int]:
    """Count denominator poles that can be spurious after expanding F chains.

    For a chain ``p_a · F_j · ...`` the expansion contains a branch with the
    adjacent factor ``p_a · p_j``; in a gauge with ``p_a · e_j = 0`` this is the
    visible scalar factor.  Similarly, ``... · F_k · p_b`` exposes
    ``p_k · p_b`` in the corresponding right-end gauge.  A repeated denominator
    copy of such an adjacent endpoint pole is therefore treated as spurious in
    the sense relevant for the training data: it is not inserted because of an
    explicit scalar numerator factor, but because it is hidden inside the
    expanded gauge-invariant block.

    The count is capped later so that generated denominators have at most one
    extra copy of any such pole: ``D^2`` means one physical simple pole and one
    expansion-spurious copy, not a physical double pole.
    """
    physical_pool = set(_all_physical_poles(N))
    counts: dict[str, int] = {}

    def add(a: int | None, b: int | None) -> None:
        if a is None or b is None or a == b:
            return
        term = _canon_pp(dot(p(a), p(b)))
        if term not in physical_pool:
            return
        counts[term] = counts.get(term, 0) + 1

    for block in spec.blocks:
        if block.kind != "chain" or not block.photons:
            continue
        add(block.left, block.photons[0])
        add(block.right, block.photons[-1])

    return counts


def _physical_denominator_factors(
    spec: MonomialSpec,
    N: int,
    *,
    repeat_probability: float = DENOM_REPEAT_PROBABILITY,
) -> list[str]:
    """Build scalar-QED-like denominator poles with expansion-spurious repeats.

    The total number of denominator factors is fixed by the target manifest
    mass dimension.  Repeated denominator factors are generated only for poles
    that are hidden in adjacent p·F-chain expansions.  Concretely, if a term
    contains a block ``p_a · F_j · ...`` then the F_j expansion contains a branch
    proportional to ``p_a · p_j``; in a gauge with ``p_a · e_j = 0`` that branch
    makes the would-be pole look cancellable.  The right endpoint works
    similarly for ``... · F_k · p_b``.

    We therefore allow at most one extra denominator copy for such adjacent
    endpoint poles.  Thus ``D^2`` means one simple physical pole and one
    expansion-spurious copy.  Manifest scalar numerator factors are *not* used
    to justify repeated denominators here, and repeated denominators whose pole
    already appears as a manifest scalar numerator factor are suppressed, because
    those would just generate trivial factors like ``D^2/D^2``.
    """
    factors: list[str] = []
    counts: dict[str, int] = {}
    spurious_counts = _chain_expansion_spurious_pp_counts(spec, N)
    manifest_scalar_counts = _explicit_scalar_pp_counts(spec.numerator)

    def canon(term: str) -> str:
        return _canon_pp(term)

    def multiplicity(term: str) -> int:
        return counts.get(canon(term), 0)

    def spurious_budget(term: str) -> int:
        # Cap at one extra copy.  This is enough to make D^2 denominators while
        # preventing genuine double-physical-pole training examples.  Do not use
        # a manifest scalar numerator factor to justify the repeat, since that
        # would create trivial D/D cancellations rather than the desired hidden
        # p·F-expansion spurious pole.
        cterm = canon(term)
        if manifest_scalar_counts.get(cterm, 0) > 0:
            return 0
        return 1 if spurious_counts.get(cterm, 0) > 0 else 0

    def max_allowed(term: str) -> int:
        # One genuine physical pole may be present.  Additional copies require
        # the p·F-chain expansion-spurious budget above.
        return 1 + spurious_budget(term)

    def can_add(term: str) -> bool:
        cterm = canon(term)
        return counts.get(cterm, 0) < max_allowed(cterm)

    def add(term: str) -> None:
        cterm = canon(term)
        if not can_add(cterm):
            raise ValueError(f"Would create a physical double pole in {cterm}")
        factors.append(cterm)
        counts[cterm] = counts.get(cterm, 0) + 1

    target = _required_denominator_count(spec.numerator_mass_dim, N)
    if target is None:
        raise ValueError(
            f"Numerator mass dimension {spec.numerator_mass_dim} cannot give 4-{N}"
        )

    chain_blocks = [blk for blk in spec.blocks if blk.kind == "chain"]
    trace_blocks = [blk for blk in spec.blocks if blk.kind == "trace"]

    # Mandatory support: each photon appearing in a p·F...F·p block gets at
    # least one pole involving its momentum.  We prefer endpoint-adjacent poles
    # when available, because those are precisely the poles whose repeated copy
    # can be expansion-spurious.
    for block in chain_blocks:
        preferred = [x for x in (block.left, block.right) if x is not None]
        for photon in block.photons:
            choices = [canon(x) for x in _photon_pole_choices(photon, N, preferred=preferred)]
            viable = [x for x in choices if can_add(x)]
            if not viable:
                raise ValueError(f"No viable simple-pole denominator for photon {photon}")
            adjacent = [x for x in viable if spurious_budget(x) > 0]
            if adjacent and random.random() < 0.75:
                add(random.choice(adjacent))
            else:
                add(random.choice(viable))

    if len(factors) > target:
        raise ValueError(
            f"Mandatory chain poles ({len(factors)}) exceed target denominator count ({target})"
        )

    # Traces do not require a pole.  Add at most one trace-associated simple
    # pole if there is still room.
    if trace_blocks and len(factors) < target and random.random() < 0.55:
        block = random.choice(trace_blocks)
        photon = random.choice(block.photons)
        choices = [canon(x) for x in _photon_pole_choices(photon, N)]
        viable = [x for x in choices if can_add(x)]
        if viable:
            add(random.choice(viable))

    pool = [canon(term) for term in _all_physical_poles(N)]
    pool = list(dict.fromkeys(pool))
    if not pool and target:
        raise ValueError("No physical denominator poles available")

    repeat_probability = max(0.0, min(1.0, repeat_probability))
    while len(factors) < target:
        remaining = target - len(factors)

        # Insert D^2 directly when D is an expansion-spurious endpoint pole and
        # no copy has yet been used.  This produces old-style apparent repeated
        # poles without relying on explicit numerator factors.
        pair_candidates = [
            term
            for term in pool
            if spurious_budget(term) > 0 and multiplicity(term) == 0 and remaining >= 2
        ]
        if pair_candidates and random.random() < repeat_probability:
            term = random.choice(pair_candidates)
            add(term)
            add(term)
            continue

        # Or, if a simple physical copy is already present, add exactly one
        # expansion-spurious copy.
        repeat_candidates = [
            term
            for term in set(factors)
            if spurious_budget(term) > 0 and can_add(term)
        ]
        if repeat_candidates and random.random() < repeat_probability:
            add(random.choice(repeat_candidates))
            continue

        viable_pool = [term for term in pool if can_add(term)]
        if not viable_pool:
            raise ValueError("No denominator factors left without creating physical double poles")
        add(random.choice(viable_pool))

    if len(factors) != target:
        raise ValueError(
            f"Failed to build {target} denominator factors from physical pole pool"
        )
    return factors

def _term_signature(spec: MonomialSpec, denom_factors: Sequence[str]) -> tuple:
    trace_lengths = sorted(len(block.photons) for block in spec.blocks if block.kind == "trace")
    chain_lengths = sorted(len(block.photons) for block in spec.blocks if block.kind == "chain")
    denom_support = sorted(
        tuple(sorted(map(int, re.findall(r"\d+", factor)))) for factor in denom_factors
    )
    return (
        tuple(trace_lengths),
        tuple(chain_lengths),
        spec.scalar_pairs,
        spec.numerator_mass_dim,
        tuple(denom_support),
    )


def _generate_term(
    N: int,
    *,
    use_denominators: bool,
    old_style_blocks: bool = False,
    denom_repeat_probability: float = DENOM_REPEAT_PROBABILITY,
    scalar_power_probability: float = SCALAR_POWER_PROBABILITY,
) -> tuple[str, str, tuple]:
    spec = _generate_gi_monomial_spec(
        N,
        old_style_blocks=old_style_blocks,
        scalar_power_probability=scalar_power_probability,
    )
    den_factors = (
        _physical_denominator_factors(
            spec,
            N,
            repeat_probability=denom_repeat_probability,
        )
        if use_denominators
        else []
    )

    denominator = canonicalise_denominator("*".join(den_factors))
    simple_num = spec.numerator
    expanded_num = "*".join(rewrite_gi(f) for f in simple_num.split("*"))

    if denominator:
        simple_term = f"({simple_num})/({denominator})"
        expanded_term = f"({expanded_num})/({denominator})"
    else:
        simple_term = simple_num
        expanded_term = expanded_num
    return simple_term, expanded_term, _term_signature(spec, den_factors)


def _has_supported_physical_poles(simple_term: str) -> bool:
    if "/" not in simple_term:
        return True
    num, den = simple_term.split("/", 1)
    den_factors = {
        _canon_pp(f.strip())
        for f in den.strip()[1:-1].split("*")
        if f.strip()
    }
    blocks = [f.strip() for f in num.strip()[1:-1].split("*") if f.strip()]
    for block in blocks:
        match = _RE_pFchainp.fullmatch(block)
        if not match:
            continue
        photons = [int(x) for x in re.findall(r"F_(\d+)", match.group(2))]
        for photon in photons:
            if not any(re.search(fr"\bp_{photon}\b", factor) for factor in den_factors):
                return False
    return True


def manifest_mass_dimension(simple_term: str) -> int:
    """Return the manifest mass dimension of a simple GI expression.

    Integer coefficients are treated as dimensionless. For sums, every term must
    have the same manifest dimension; otherwise a ValueError is raised.
    """

    def split_top_level_sum(expr: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and ch in "+-" and i > 0:
                prev = expr[i - 1]
                if prev not in "*/^(+":
                    parts.append("".join(current).strip())
                    current = [ch]
                    continue
            current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return [part for part in parts if part]

    def count_product_dim(expr: str) -> int:
        expr = _strip_matched_outer_parens(expr)
        factors = [f.strip() for f in _split_top_level(expr, "*") if f.strip()]
        total = 0
        for factor in factors:
            factor = _strip_matched_outer_parens(factor)
            
            # Handle powers: (base)^n
            power_match = re.fullmatch(r"\((.+)\)\^(\d+)", factor)
            if power_match:
                base = power_match.group(1)
                power = int(power_match.group(2))
                # Count dimension of base and multiply by power
                base_dim = count_product_dim(base)
                total += base_dim * power
                continue
            
            if re.fullmatch(r"-?\d+(?:\.\d+)?", factor):
                continue
            match = _RE_TrN.fullmatch(factor)
            if match:
                total += len(re.findall(r"F_\d+", match.group(1)))
                continue
            match = _RE_pFchainp.fullmatch(factor)
            if match:
                total += len(re.findall(r"F_\d+", match.group(2))) + 2
                continue
            if _RE_pp.fullmatch(factor):
                total += 2
                continue
            if "*" in factor:
                total += count_product_dim(factor)
                continue
            raise ValueError(f"Unrecognised factor in manifest dimension count: {factor}")
        return total

    term_dims: list[int] = []
    for term in split_top_level_sum(simple_term.strip()):
        expr = term.lstrip("+-").strip()
        if "/" in expr:
            num, den = expr.split("/", 1)
            num_dim = count_product_dim(num)
            den_body = _strip_matched_outer_parens(den)
            denom_dim = count_product_dim(den_body)
        else:
            num_dim = count_product_dim(expr)
            denom_dim = 0
        term_dims.append(num_dim - denom_dim)

    if not term_dims:
        raise ValueError("Empty expression")
    if len(set(term_dims)) != 1:
        raise ValueError(f"Inconsistent term dimensions: {term_dims}")
    return term_dims[0]


def scr_mul_by_one(expr: str, N: int) -> str:
    i, j = random.sample(range(1, N + 1), 2)
    one = f"({dot(p(i), p(j))})/({dot(p(i), p(j))})"
    return f"({expr})*{one}"


def scr_ward_substitute(expr: str, Ngamma: int, N: int) -> str:
    photon = random.randint(2, Ngamma + 1)
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


_SCRAMBLER_BY_NAME = {
    SCRAMBLE_MULTIPLY_ONE: lambda expr, Ng, N: scr_mul_by_one(expr, N),
    SCRAMBLE_WARD: lambda expr, Ng, N: scr_ward_substitute(expr, Ng, N),
    SCRAMBLE_MOMENTUM: lambda expr, Ng, N: scr_momentum_substitute(expr, N),
    SCRAMBLE_COMMUTE_DOT: lambda expr, Ng, N: scr_commute_dot(expr),
    SCRAMBLE_RATIO: lambda expr, Ng, N: scr_mul_by_ratio(expr, N),
    SCRAMBLE_PARTIAL_FRACTION: lambda expr, Ng, N: scr_partial_fraction(expr),
}


def _active_scramblers(scramble_names: Sequence[str] | None = None):
    names = DEFAULT_SCRAMBLES if scramble_names is None else tuple(scramble_names)
    out = []
    unknown = []
    for name in names:
        if name in _SCRAMBLER_BY_NAME:
            out.append(_SCRAMBLER_BY_NAME[name])
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"Unknown scramble labels: {unknown}")
    return tuple(out)


def scramble(
    expr: str,
    Ngamma: int,
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
        cand = random.choice(active_scramblers)(out, Ngamma, N)
        if full_expand:
            cand = full_expand_expression(cand)
        if len(cand) <= max_len:
            out = cand
    return full_expand_expression(out) if full_expand else out


def _validate_pair(
    expr_a: str,
    expr_b: str,
    N: int,
    M: float,
    *,
    n_checks: int = 3,
    tol_rel: float = 1e-8,
    tol_abs: float = 1e-10,
) -> tuple[bool, str]:
    for _ in range(n_checks):
        mom, pol = generate_kinematics(N, M=M)
        try:
            va = eval_infix_numeric(expr_a, mom, pol)
            vb = eval_infix_numeric(expr_b, mom, pol)
        except Exception as exc:
            return False, f"exception:{exc}"
        if not (math.isfinite(va) and math.isfinite(vb)):
            return False, "non-finite"
        diff = abs(va - vb)
        if diff > max(tol_abs, tol_rel * max(1.0, abs(va), abs(vb))):
            return False, f"mismatch:{diff:.3e}"
    return True, ""


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


def build_dataset(
    N: int,
    num_samples: int,
    *,
    max_scr: int = DEFAULT_MAX_SCR,
    min_scr: int = DEFAULT_MIN_SCR,
    seed: int | None = None,
    unit_probability: float = UNIT_PROBABILITY,
    old_style_probability: float = OLD_STYLE_PROBABILITY,
    denom_repeat_probability: float = DENOM_REPEAT_PROBABILITY,
    scalar_power_probability: float = SCALAR_POWER_PROBABILITY,
    use_denominators: bool = DEFAULT_USE_DENOMINATORS,
    validate: bool = DEFAULT_VALIDATE,
    M: float = DEFAULT_MASS,
    min_terms: int = DEFAULT_MIN_TERMS,
    max_terms: int = DEFAULT_MAX_TERMS,
    log_path: str | None = None,
    max_attempts_factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR,
    full_expand_scrambled: bool = DEFAULT_FULL_EXPAND_SCRAMBLED,
) -> list[tuple[str, str]]:
    """Build a dataset of (simple, scrambled) pairs."""
    min_terms = max(1, int(min_terms))
    max_terms = max(min_terms, int(max_terms))
    if seed is not None:
        random.seed(seed)

    Ngamma = N - 2
    data: list[tuple[str, str]] = []
    stats = {
        "attempts": 0,
        "parity_fail": 0,
        "scramble_fail": 0,
        "pole_fail": 0,
        "dimension_fail": 0,
    }
    max_attempts = max(1, num_samples * max_attempts_factor)

    if log_path:
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(
                f"# gen_data log N={N} target={num_samples} "
                f"terms=[{min_terms},{max_terms}] scr=[{min_scr},{max_scr}] "
                f"unit_probability={unit_probability} "
                f"old_style_probability={old_style_probability} "
                f"spurious_repeat_probability={denom_repeat_probability} "
                f"scalar_power_probability={scalar_power_probability} "
                f"full_expand_scrambled={full_expand_scrambled} seed={seed}\n"
            )

    while len(data) < num_samples and stats["attempts"] < max_attempts:
        stats["attempts"] += 1
        use_old_style = random.random() < max(0.0, min(1.0, old_style_probability))
        if use_old_style:
            n_terms = 2
            use_unit_coeffs = True
        else:
            n_terms = random.randint(min_terms, max_terms)
            use_unit_coeffs = random.random() < max(0.0, min(1.0, unit_probability))

        first_simple, first_expanded, _signature = _generate_term(
            N,
            use_denominators=use_denominators,
            old_style_blocks=use_old_style,
            denom_repeat_probability=denom_repeat_probability,
            scalar_power_probability=scalar_power_probability,
        )
        if not _has_supported_physical_poles(first_simple):
            stats["pole_fail"] += 1
            continue
        if manifest_mass_dimension(first_simple) != 4 - N:
            stats["dimension_fail"] += 1
            continue

        simple_terms = [first_simple]
        expanded_terms = [first_expanded]
        coeffs = [random.choice((-1, 1)) if use_unit_coeffs else random.choice(SCALAR_COEFF_POOL)]

        ok_build = True
        for _ in range(n_terms - 1):
            for _attempt in range(80):
                cand_simple, cand_expanded, _cand_signature = _generate_term(
                    N,
                    use_denominators=use_denominators,
                    old_style_blocks=use_old_style,
                    denom_repeat_probability=denom_repeat_probability,
                    scalar_power_probability=scalar_power_probability,
                )
                if not _has_supported_physical_poles(cand_simple):
                    continue
                if manifest_mass_dimension(cand_simple) != 4 - N:
                    continue
                simple_terms.append(cand_simple)
                expanded_terms.append(cand_expanded)
                coeffs.append(random.choice((-1, 1)) if use_unit_coeffs else random.choice(TERM_COEFF_POOL))
                break
            else:
                ok_build = False
                break
        if not ok_build:
            continue

        simple_expr = _format_poly(simple_terms, coeffs)
        expanded_expr = _format_poly(expanded_terms, coeffs)

        if validate:
            ok, _ = _validate_pair(simple_expr, expanded_expr, N, M)
            if not ok:
                stats["parity_fail"] += 1
                continue

        scrambled = scramble(
            expanded_expr,
            Ngamma,
            N,
            min_scr=min_scr,
            max_scr=max_scr,
            full_expand=full_expand_scrambled,
        )

        if validate:
            ok, _ = _validate_pair(expanded_expr, scrambled, N, M)
            if not ok:
                stats["scramble_fail"] += 1
                continue

        data.append((simple_expr, scrambled))

    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(
                "# SUMMARY "
                f"accepted={len(data)} parity_fail={stats['parity_fail']} "
                f"scramble_fail={stats['scramble_fail']} pole_fail={stats['pole_fail']} "
                f"dimension_fail={stats['dimension_fail']} attempts={stats['attempts']}\n"
            )
    return data


def _batch_sizes(total: int, batch_size: int) -> list[int]:
    total = max(0, int(total))
    batch_size = max(1, int(batch_size))
    out: list[int] = []
    remaining = total
    while remaining > 0:
        take = min(batch_size, remaining)
        out.append(take)
        remaining -= take
    return out


def _progress(iterable, *, total: int, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="batch")
    except Exception:
        return iterable


def _worker_build_dataset(job: BatchJob) -> list[tuple[str, str]]:
    return build_dataset(
        job.N,
        job.num_samples,
        max_scr=job.max_scr,
        min_scr=job.min_scr,
        seed=job.seed,
        unit_probability=job.unit_probability,
        old_style_probability=job.old_style_probability,
        denom_repeat_probability=job.denom_repeat_probability,
        scalar_power_probability=job.scalar_power_probability,
        use_denominators=job.use_denominators,
        validate=job.validate,
        M=job.M,
        min_terms=job.min_terms,
        max_terms=job.max_terms,
        log_path=None,
        max_attempts_factor=job.max_attempts_factor,
        full_expand_scrambled=job.full_expand_scrambled,
    )


def build_dataset_batched(
    N: int,
    num_samples: int,
    *,
    max_scr: int = DEFAULT_MAX_SCR,
    min_scr: int = DEFAULT_MIN_SCR,
    seed: int | None = None,
    unit_probability: float = UNIT_PROBABILITY,
    old_style_probability: float = OLD_STYLE_PROBABILITY,
    denom_repeat_probability: float = DENOM_REPEAT_PROBABILITY,
    scalar_power_probability: float = SCALAR_POWER_PROBABILITY,
    use_denominators: bool = DEFAULT_USE_DENOMINATORS,
    validate: bool = DEFAULT_VALIDATE,
    M: float = DEFAULT_MASS,
    min_terms: int = DEFAULT_MIN_TERMS,
    max_terms: int = DEFAULT_MAX_TERMS,
    log_path: str | None = None,
    max_attempts_factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR,
    full_expand_scrambled: bool = DEFAULT_FULL_EXPAND_SCRAMBLED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    jobs: int | str = DEFAULT_JOBS,
    progress: bool = DEFAULT_PROGRESS,
) -> list[tuple[str, str]]:
    """Build a dataset in independent batches, optionally using multiple CPUs."""
    batch_counts = _batch_sizes(num_samples, batch_size)
    if not batch_counts:
        return []

    jobs = max(1, int(jobs))
    base_seed = seed if seed is not None else random.randrange(1, 2**31 - 1)
    seeds = [base_seed + 1000003 * i for i in range(len(batch_counts))]
    job_specs = [
        BatchJob(
            N=N,
            num_samples=count,
            max_scr=max_scr,
            min_scr=min_scr,
            seed=seeds[i],
            unit_probability=unit_probability,
            old_style_probability=old_style_probability,
            denom_repeat_probability=denom_repeat_probability,
            scalar_power_probability=scalar_power_probability,
            use_denominators=use_denominators,
            validate=validate,
            M=M,
            min_terms=min_terms,
            max_terms=max_terms,
            max_attempts_factor=max_attempts_factor,
            full_expand_scrambled=full_expand_scrambled,
        )
        for i, count in enumerate(batch_counts)
    ]

    if log_path:
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(
                f"# gen_data batched log N={N} target={num_samples} "
                f"batches={len(job_specs)} batch_size={batch_size} jobs={jobs} "
                f"terms=[{min_terms},{max_terms}] scr=[{min_scr},{max_scr}] "
                f"unit_probability={unit_probability} "
                f"old_style_probability={old_style_probability} "
                f"spurious_repeat_probability={denom_repeat_probability} "
                f"scalar_power_probability={scalar_power_probability} "
                f"full_expand_scrambled={full_expand_scrambled} seed={seed} base_seed={base_seed}\n"
            )

    pairs: list[tuple[str, str]] = []
    if jobs == 1:
        iterator = (_worker_build_dataset(job) for job in job_specs)
        for batch_pairs in _progress(iterator, total=len(job_specs), enabled=progress, desc="generating"):
            pairs.extend(batch_pairs)
    else:
        with mp.Pool(processes=jobs) as pool:
            iterator = pool.imap_unordered(_worker_build_dataset, job_specs)
            for batch_pairs in _progress(iterator, total=len(job_specs), enabled=progress, desc="generating"):
                pairs.extend(batch_pairs)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"# SUMMARY accepted={len(pairs)} requested={num_samples} "
                f"batches={len(job_specs)} jobs={jobs}\n"
            )
    return pairs


def _resolve_jobs(value: str) -> int:
    if str(value).lower() == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(value))


def dedupe_pairs(
    pairs: list[tuple[str, str]],
    *,
    keep: str = "first",
) -> tuple[list[tuple[str, str]], int]:
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")
    if keep == "first":
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for item in pairs:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out, len(pairs) - len(out)
    last_idx = {item: i for i, item in enumerate(pairs)}
    out = [item for i, item in enumerate(pairs) if last_idx[item] == i]
    return out, len(pairs) - len(out)


def write_csv(pairs: Iterable[tuple[str, str]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["simple", "scrambled"])
        for simple, scrambled in pairs:
            writer.writerow([simple, scrambled])


def tokenise_csv(inp: str, out: str, *, max_particles: int = DEFAULT_TOKENIZER_MAX_PARTICLES) -> None:
    from Tokenizer import ScatteringAmplitudeTokenizer

    tok = ScatteringAmplitudeTokenizer(max_particles=max_particles)
    with open(inp, newline="", encoding="utf-8") as fin, open(
        out, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["simple", "scrambled"])
        writer.writeheader()
        for row in reader:
            writer.writerow(
                {
                    "simple": json.dumps(tok.encode_infix(row["simple"])),
                    "scrambled": json.dumps(tok.encode_infix(row["scrambled"])),
                }
            )


def _safe_eval_float(expr_str: str) -> float:
    tree = _ast.parse(expr_str, mode="eval")

    class _Visitor(_ast.NodeVisitor):
        def visit_Expression(self, node):
            return self.visit(node.body)

        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            ops = {
                _ast.Add: float.__add__,
                _ast.Sub: float.__sub__,
                _ast.Mult: float.__mul__,
                _ast.Div: float.__truediv__,
                _ast.Pow: float.__pow__,
            }
            for cls, fn in ops.items():
                if isinstance(node.op, cls):
                    return fn(left, right)
            raise ValueError("disallowed op")

        def visit_UnaryOp(self, node):
            value = self.visit(node.operand)
            if isinstance(node.op, _ast.UAdd):
                return +value
            if isinstance(node.op, _ast.USub):
                return -value
            raise ValueError("disallowed unary")

        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("disallowed constant")

        visit_Num = visit_Constant

        def generic_visit(self, node):
            raise ValueError("disallowed node")

    return _Visitor().visit(tree)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scalar-QED-like amplitude data.")
    parser.add_argument("N", nargs="?", type=int, default=DEFAULT_N_PARTICLES, help="Number of external legs.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--max-scr", type=int, default=DEFAULT_MAX_SCR)
    parser.add_argument("--min-scr", type=int, default=DEFAULT_MIN_SCR)
    parser.add_argument("--min-terms", type=int, default=DEFAULT_MIN_TERMS)
    parser.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--unit-probability",
        type=float,
        default=UNIT_PROBABILITY,
        help=(
            "Probability that a generated expression uses only unit coefficients "
            "(+1 or -1) for all top-level terms. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--old-style-probability",
        type=float,
        default=OLD_STYLE_PROBABILITY,
        help=(
            "Probability of old-source-like samples: two terms, ±1 top-level "
            "coefficients, and 4pt block weights biased toward single-F chains. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--denom-repeat-probability",
        "--spurious-repeat-probability",
        type=float,
        default=DENOM_REPEAT_PROBABILITY,
        help=(
            "Probability of adding a repeated denominator pole when the extra copy "
            "is spurious because the pole appears in an adjacent p·F-chain "
            "expansion. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--scalar-power-probability",
        type=float,
        default=SCALAR_POWER_PROBABILITY,
        help=(
            "Probability, when adding optional scalar p·p numerator factors, "
            "of repeating an existing physical-pole scalar factor. This creates "
            "manifest numerator powers like (p_i · p_j)^2. Repeated denominators "
            "are generated separately from hidden p·F-chain factors, not from "
            "these trivial scalar cancellations. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of accepted pairs per generation batch. Default: %(default)s",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default=DEFAULT_JOBS,
        help="Number of worker processes, or 'auto'. Default: %(default)s",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS)
    parser.add_argument("--raw-out", type=str, default=None)
    parser.add_argument("--tok-out", type=str, default=None)
    parser.add_argument("--log-out", type=str, default=None)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-tokenise", action="store_true")
    parser.add_argument(
        "--grouped-scrambled",
        action="store_true",
        help="Keep the old grouped scrambled style instead of fully expanding scrambled expressions.",
    )
    args = parser.parse_args()

    nsamps = args.samples // 1000
    raw_out = args.raw_out or DEFAULT_RAW_OUT_TEMPLATE.format(N=args.N, NSAMPS=nsamps)
    tok_out = args.tok_out or DEFAULT_TOK_OUT_TEMPLATE.format(N=args.N, NSAMPS=nsamps)
    log_out = args.log_out or DEFAULT_LOG_OUT_TEMPLATE.format(N=args.N, NSAMPS=nsamps)

    t0 = time.perf_counter()
    oversample = int(round(args.samples * DEFAULT_OVERSAMPLE_FACTOR))
    pairs = build_dataset_batched(
        args.N,
        oversample,
        max_scr=args.max_scr,
        min_scr=args.min_scr,
        seed=args.seed,
        unit_probability=args.unit_probability,
        old_style_probability=args.old_style_probability,
        denom_repeat_probability=args.denom_repeat_probability,
        scalar_power_probability=args.scalar_power_probability,
        use_denominators=DEFAULT_USE_DENOMINATORS,
        validate=not args.no_validate,
        M=args.mass,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
        log_path=log_out,
        full_expand_scrambled=(not args.grouped_scrambled) if DEFAULT_FULL_EXPAND_SCRAMBLED else False,
        batch_size=args.batch_size,
        jobs=_resolve_jobs(args.jobs),
        progress=DEFAULT_PROGRESS and not args.no_progress,
    )
    t1 = time.perf_counter()

    before = len(pairs)
    pairs, removed = dedupe_pairs(pairs)
    pairs = pairs[: args.samples]
    write_csv(pairs, raw_out)
    if DEFAULT_TOKENISE and not args.no_tokenise:
        tokenise_csv(raw_out, tok_out)
    t2 = time.perf_counter()

    print(f"{len(pairs)} pairs -> {raw_out}")
    print(f"  generation : {t1 - t0:.2f}s")
    print(f"  dedupe     : removed {removed} ({before} -> {len(pairs)})")
    print(f"  write/tok  : {t2 - t1:.2f}s")
    print(f"  log        : {log_out}")
