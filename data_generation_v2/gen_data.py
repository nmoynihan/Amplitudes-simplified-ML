#!/usr/bin/env python3
"""
gen_data.py — Build (simple, scrambled) training pairs for N-point amplitudes.

    p_1, p_N          : massive scalars
    p_2 … p_{N-1}     : massless photons (each owns e_i, F_i)

"simple"    = gauge-invariant monomial built from Tr(F·…·F) and p·F·…·F·p blocks,
              each F_i appearing exactly once (weight 1), times optional p·p factors.
"scrambled" = that monomial expanded into e·p / p·p dot products, then hit by
              algebraic identities that preserve numerical equality.

Output: CSV with columns (simple, scrambled).
"""
from __future__ import annotations

import ast as _ast
import csv
import json
import math
import random
import re
import sys
import time
from itertools import product as iproduct
from typing import List, Tuple

from kinematics import generate_kinematics, mdot

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §1  String helpers                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

DOT = "·"


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


def _format_signed_sum(terms: list[tuple[int, str]]) -> str:
    """Format [(sign, expr), …] as '(t1 - t2 + t3 …)'."""
    pieces: list[str] = []
    for sgn, t in terms:
        if not pieces:
            pieces.append(t if sgn > 0 else f"-{t}")
        else:
            pieces.append(("+ " if sgn > 0 else "- ") + t)
    return "(" + " ".join(pieces) + ")"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §2  Rewrite rules:  GI blocks → e·p / p·p products               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _rw_pFchainp(*idxs: int) -> str:
    """Expand  p_i · F_{j1} · … · F_{jn} · p_k  into dot products.

    Args: (i, j1, j2, …, jn, k)  with n ≥ 1.
    """
    assert len(idxs) >= 3
    i, *Fs, k = idxs
    n = len(Fs)
    terms: list[tuple[int, str]] = []
    for mask in range(1 << n):
        sgn = 1 if bin(mask).count("1") % 2 == 0 else -1
        factors: list[str] = []
        prev = ("p", i)
        for bit, j in enumerate(Fs):
            swap = (mask >> bit) & 1
            L = ("e", j) if swap else ("p", j)
            R = ("p", j) if swap else ("e", j)
            factors.append(f"({dot(_vec(*prev), _vec(*L))})")
            prev = R
        factors.append(f"({dot(_vec(*prev), p(k))})")
        terms.append((sgn, "*".join(factors)))
    return _format_signed_sum(terms)


def _rw_TrN(*js: int) -> str:
    """Expand  Tr(F_{j1} · … · F_{jn})  into dot products.

    Uses  F_j^{μν} = p_j^μ e_j^ν − e_j^μ p_j^ν  and cyclic trace.
    """
    n = len(js)
    assert n >= 2
    terms: list[tuple[int, str]] = []
    for mask in range(1 << n):
        sgn = 1 if bin(mask).count("1") % 2 == 0 else -1
        # (first, second) for each F_j
        pairs = []
        for bit, j in enumerate(js):
            if (mask >> bit) & 1:
                pairs.append((e(j), p(j)))
            else:
                pairs.append((p(j), e(j)))
        # Cyclic contraction:  second_i · first_{(i+1) mod n}
        factors = [
            f"({dot(pairs[i][1], pairs[(i + 1) % n][0])})"
            for i in range(n)
        ]
        terms.append((sgn, "*".join(factors)))
    return _format_signed_sum(terms)


def _vec(tag: str, idx: int) -> str:
    return f"{tag}_{idx}"


# ── Regex patterns for matching GI blocks in strings ─────────────────

_RE_pp = re.compile(r"p_(\d+)\s*·\s*p_(\d+)")
# General patterns — we match the whole block and extract F indices inside
_RE_pFchainp = re.compile(
    r"p_(\d+)((?:\s*·\s*F_\d+)+)\s*·\s*p_(\d+)"
)
_RE_TrN = re.compile(
    r"Tr\((F_\d+(?:\s*·\s*F_\d+)*)\)"
)


def rewrite_gi(block: str) -> str:
    """Rewrite a single GI block into e·p / p·p dot products."""
    block = block.strip()

    # Tr(F_j · F_k · …)
    m = _RE_TrN.fullmatch(block)
    if m:
        js = [int(x) for x in re.findall(r"F_(\d+)", m.group(1))]
        return _rw_TrN(*js)

    # p_i · F_j · … · F_k · p_l
    m = _RE_pFchainp.fullmatch(block)
    if m:
        i = int(m.group(1))
        Fs = [int(x) for x in re.findall(r"F_(\d+)", m.group(2))]
        k = int(m.group(3))
        return _rw_pFchainp(i, *Fs, k)

    # p_i · p_j  (pass through)
    if _RE_pp.fullmatch(block):
        return block

    return block


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §3  Canonicalisation                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _canon_pp(term: str) -> str:
    m = _RE_pp.fullmatch(term)
    if not m:
        return term
    i, j = sorted(map(int, m.groups()))
    return dot(p(i), p(j))


def _canon_TrN(term: str) -> str:
    """Canonical form: smallest cyclic rotation of the F indices."""
    m = _RE_TrN.fullmatch(term)
    if not m:
        return term
    js = [int(x) for x in re.findall(r"F_(\d+)", m.group(1))]
    n = len(js)
    # Find lexicographically smallest rotation
    best = js
    for r in range(1, n):
        rotated = js[r:] + js[:r]
        if rotated < best:
            best = rotated
    return Tr(*(F(j) for j in best))


def _factor_sort_key(term: str) -> tuple:
    """Sort key for ordering factors in a GI product."""
    term = term.strip()
    if _RE_TrN.fullmatch(term):
        js = re.findall(r"F_(\d+)", term)
        return (0, len(js), _canon_TrN(term))
    if _RE_pFchainp.fullmatch(term):
        Fs = re.findall(r"F_(\d+)", term)
        return (1, len(Fs), term)
    if _RE_pp.fullmatch(term):
        return (2, 0, _canon_pp(term))
    return (3, 0, term)


def canonicalise_gi_product(prod: str) -> str:
    """Canonicalise a '*'-joined product of GI factors."""
    factors = [f.strip() for f in prod.split("*") if f.strip()]
    canon = []
    for f in factors:
        if _RE_TrN.fullmatch(f):
            f = _canon_TrN(f)
        elif _RE_pp.fullmatch(f):
            f = _canon_pp(f)
        canon.append(f)
    canon.sort(key=_factor_sort_key)
    return "*".join(canon)


def canonicalise_denominator(den: str) -> str:
    if not den:
        return den
    fs = [_canon_pp(f.strip()) for f in den.split("*")]
    fs.sort()
    return "*".join(fs)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §4  AST classes (module scope — not recreated per call)           ║
# ╚══════════════════════════════════════════════════════════════════════╝

class _Num:
    __slots__ = ("v",)
    def __init__(self, v: float):
        self.v = float(v)


class _Vec:
    __slots__ = ("tag", "idx")
    def __init__(self, tag: str, idx: int):
        self.tag = tag   # 'p', 'e', 'F'
        self.idx = idx


class _DotChain:
    """Sequence of Vec nodes (possibly with F), or ending with _TR sentinel."""
    __slots__ = ("parts",)
    _TR = object()  # sentinel

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


# Tuple nodes used in the expanded AST
_DOT_TAG = "DOT"
_VEC_TAG = "VEC"

_TOKEN_RE = re.compile(
    r"\s*(\d+\.\d+|\d+|p_\d+|e_\d+|F_\d+|Tr\b|\*\*|\^|\+|\-|\*|/|\(|\)|\.|·|,)"
)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §5  Parser (module scope)                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

class _Parser:
    """Recursive-descent parser for amplitude expressions.

    Grammar (precedence low→high):
        expr   = term (('+' | '-') term)*
        term   = factor (('*' | '/') factor)*
        factor = ['+' | '-'] power
        power  = primary ('**' primary)*
        primary = '(' expr ')' | Tr(...) | dot-chain | number
    """

    def __init__(self, toks: list[str]):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def pop(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        return self._expr()

    def _expr(self):
        node = self._term()
        while self.peek() in ("+", "-"):
            op = self.pop()
            node = _BinOp(op, node, self._term())
        return node

    def _term(self):
        node = self._factor()
        while self.peek() in ("*", "/"):
            op = self.pop()
            node = _BinOp(op, node, self._factor())
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
        t = self.peek()

        # Parenthesised sub-expression
        if t == "(":
            self.pop()
            node = self._expr()
            if self.peek() == ")":
                self.pop()
            return node

        # Trace: Tr(F_i · F_j · …)
        if t == "Tr":
            self.pop()
            if self.peek() == "(":
                self.pop()
                parts = self._parse_F_chain()
                if self.peek() == ")":
                    self.pop()
                vecs = [_Vec("F", int(re.search(r"\d+", tok).group())) for tok in parts]
                return _DotChain(vecs + [_DotChain._TR])
            return _Num(0.0)

        # Dot-chain: sequence of p_/e_/F_ separated by '·' or '.'
        parts: list[_Vec] = []
        while True:
            tok = self.peek()
            if tok and re.match(r"(p_|e_|F_)\d+", str(tok)):
                m = re.match(r"(p_|e_|F_)(\d+)", tok)
                parts.append(_Vec(m.group(1)[0], int(m.group(2))))
                self.pop()
                if self.peek() in ("·", "."):
                    self.pop()
                    continue
                break
            else:
                break

        if parts:
            return parts[0] if len(parts) == 1 else _DotChain(parts)

        # Number literal
        if t and re.match(r"\d+(?:\.\d+)?", str(t)):
            self.pop()
            return _Num(float(t))

        # Fallback
        if t is not None:
            self.pop()
        return _Num(0.0)

    def _parse_F_chain(self) -> list[str]:
        """Collect F_i tokens inside Tr(…)."""
        args: list[str] = []
        while True:
            tok = self.peek()
            if tok and re.match(r"F_\d+", str(tok)):
                args.append(self.pop())
                if self.peek() in ("·", ".", ","):
                    self.pop()
                    continue
                break
            else:
                break
        return args


def _tokenize(s: str) -> list[str]:
    """Tokenize an expression string into a flat list."""
    s = s.replace("^", "**")
    pos = 0
    toks: list[str] = []
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if m:
            toks.append(m.group(1))
            pos = m.end()
        else:
            m2 = re.match(r"\s*([A-Za-z_]\w*)", s[pos:])
            if m2:
                toks.append(m2.group(1))
                pos += m2.end()
            else:
                pos += 1
    return toks


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §6  AST expansion:  F / Tr / dot-chains → arithmetic of dots     ║
# ╚══════════════════════════════════════════════════════════════════════╝

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
    """Expand a DotChain node into arithmetic AST of Minkowski dot products."""
    parts = dc.parts

    # ── Trace ────────────────────────────────────────────────────────
    if parts and parts[-1] is _DotChain._TR:
        Fs = [v for v in parts[:-1] if isinstance(v, _Vec)]
        n = len(Fs)
        js = [v.idx for v in Fs]
        acc = None
        for mask in range(1 << n):
            sgn = 1 if bin(mask).count("1") % 2 == 0 else -1
            pairs = []
            for bit, j in enumerate(js):
                if (mask >> bit) & 1:
                    pairs.append(("e", j, "p", j))
                else:
                    pairs.append(("p", j, "e", j))
            # Cyclic contraction: second_i · first_{(i+1) mod n}
            factors = [
                _mk_dot(
                    _vn(pairs[i][2], pairs[i][3]),
                    _vn(pairs[(i + 1) % n][0], pairs[(i + 1) % n][1]),
                )
                for i in range(n)
            ]
            term = factors[0]
            for f in factors[1:]:
                term = _ast_mul(term, f)
            if acc is None:
                acc = term if sgn > 0 else _UnaryOp("-", term)
            else:
                acc = _ast_add(acc, term) if sgn > 0 else _ast_sub(acc, term)
        return acc

    # ── p · F · … · F · p chain ──────────────────────────────────────
    F_vecs = [v for v in parts if isinstance(v, _Vec) and v.tag == "F"]
    if F_vecs:
        if not (isinstance(parts[0], _Vec) and parts[0].tag == "p"
                and isinstance(parts[-1], _Vec) and parts[-1].tag == "p"):
            return _Num(0.0)
        i_idx = parts[0].idx
        k_idx = parts[-1].idx
        F_idxs = [v.idx for v in F_vecs]
        n = len(F_idxs)
        acc = None
        for mask in range(1 << n):
            sgn = 1 if bin(mask).count("1") % 2 == 0 else -1
            factors = []
            prev = ("p", i_idx)
            for bit, j in enumerate(F_idxs):
                swap = (mask >> bit) & 1
                L = ("e", j) if swap else ("p", j)
                R = ("p", j) if swap else ("e", j)
                factors.append(_mk_dot(_vn(*prev), _vn(*L)))
                prev = R
            factors.append(_mk_dot(_vn(*prev), _vn("p", k_idx)))
            term = factors[0]
            for f in factors[1:]:
                term = _ast_mul(term, f)
            if acc is None:
                acc = term if sgn > 0 else _UnaryOp("-", term)
            else:
                acc = _ast_add(acc, term) if sgn > 0 else _ast_sub(acc, term)
        return acc

    # ── Pure p/e dot chain (e.g. p_a · p_b) ─────────────────────────
    acc = None
    for a, b in zip(parts, parts[1:]):
        d = _mk_dot(_vn(a.tag, a.idx), _vn(b.tag, b.idx))
        acc = d if acc is None else _ast_mul(acc, d)
    return acc if acc is not None else _Num(0.0)


def _expand_ast(node):
    """Walk AST and replace DotChain nodes with expanded arithmetic."""
    if isinstance(node, _Num):
        return node
    if isinstance(node, _Vec):
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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §7  Numeric evaluator                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

def eval_infix_numeric(expr: str, momenta, pols) -> float:
    """Parse an infix expression and evaluate numerically on given kinematics.

    momenta: (N, 4) array;  pols: (N-2, 4) array  [ε_2 … ε_{N-1}].
    """
    N = len(momenta)
    P = {f"p_{i}": momenta[i - 1] for i in range(1, N + 1)}
    E = {f"e_{i}": pols[i - 2] for i in range(2, N)}

    def _eval(node) -> float:
        if isinstance(node, _Num):
            return node.v
        if isinstance(node, _BinOp):
            l, r = _eval(node.left), _eval(node.right)
            if node.op == "+":
                return l + r
            if node.op == "-":
                return l - r
            if node.op == "*":
                return l * r
            if node.op == "/":
                return l / r
            if node.op == "**":
                return l ** r
            raise ValueError(f"Unknown op: {node.op}")
        if isinstance(node, _UnaryOp):
            v = _eval(node.operand)
            return -v if node.op == "-" else v
        if isinstance(node, tuple):
            if node[0] == _DOT_TAG:
                lhs, rhs = node[1], node[2]
                if lhs[0] == _VEC_TAG and rhs[0] == _VEC_TAG:
                    va = P.get(f"p_{lhs[2]}") if lhs[1] == "p" else E.get(f"e_{lhs[2]}")
                    vb = P.get(f"p_{rhs[2]}") if rhs[1] == "p" else E.get(f"e_{rhs[2]}")
                    if va is not None and vb is not None:
                        return mdot(va, vb)
                return 0.0
            if node[0] == _VEC_TAG:
                return 0.0
        return 0.0

    toks = _tokenize(expr)
    tree = _Parser(toks).parse()
    expanded = _expand_ast(tree)
    return float(_eval(expanded))


# Also keep a lightweight string-based evaluator for quick checks
_RE_DOT = re.compile(r"(p_\d+|e_\d+)\s*·\s*(p_\d+|e_\d+)")


def _safe_eval_float(expr_str: str) -> float:
    """Evaluate a purely numeric arithmetic expression safely via ast."""
    tree = _ast.parse(expr_str, mode="eval")

    class _V(_ast.NodeVisitor):
        def visit_Expression(self, n):
            return self.visit(n.body)
        def visit_BinOp(self, n):
            l, r = self.visit(n.left), self.visit(n.right)
            ops = {_ast.Add: float.__add__, _ast.Sub: float.__sub__,
                   _ast.Mult: float.__mul__, _ast.Div: float.__truediv__,
                   _ast.Pow: float.__pow__}
            for cls, fn in ops.items():
                if isinstance(n.op, cls):
                    return fn(l, r)
            raise ValueError("disallowed op")
        def visit_UnaryOp(self, n):
            v = self.visit(n.operand)
            if isinstance(n.op, _ast.UAdd):
                return +v
            if isinstance(n.op, _ast.USub):
                return -v
            raise ValueError("disallowed unary")
        def visit_Constant(self, n):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("disallowed constant")
        visit_Num = visit_Constant
        def generic_visit(self, n):
            raise ValueError("disallowed node")

    return _V().visit(tree)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §8  Gauge-invariant monomial generator                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

EXTRA_SCALARS_MAX = 4  # upper bound on extra p·p factors


def _singleF(j: int, N: int) -> str:
    pool = [x for x in range(1, N + 1) if x != j]
    a, b = random.sample(pool, 2)
    return f"{p(a)} {DOT} {F(j)} {DOT} {p(b)}"


def _doubleF(j: int, k: int, N: int) -> str:
    pool = [x for x in range(1, N + 1) if x not in (j, k)]
    a, b = random.sample(pool, 2)
    return f"{p(a)} {DOT} {F(j)} {DOT} {F(k)} {DOT} {p(b)}"


def _tripleF(j: int, k: int, l: int, N: int) -> str:
    """p_a · F_j · F_k · F_l · p_b — three field strengths sandwiched."""
    pool = [x for x in range(1, N + 1) if x not in (j, k, l)]
    a, b = random.sample(pool, 2)
    return f"{p(a)} {DOT} {F(j)} {DOT} {F(k)} {DOT} {F(l)} {DOT} {p(b)}"


def _tr2(j: int, k: int) -> str:
    return Tr(F(j), F(k))


def _tr3(j: int, k: int, l: int) -> str:
    return Tr(F(j), F(k), F(l))


def _tr4(j: int, k: int, l: int, m: int) -> str:
    return Tr(F(j), F(k), F(l), F(m))


def _scalar(N: int) -> str:
    i, j = random.sample(range(1, N + 1), 2)
    return dot(p(i), p(j))


def strict_gi_monomial(N: int) -> str:
    """Generate a random weight-1 GI monomial using all photon F_i exactly once.

    Supports traces up to length 4 and F-chains up to length 3.
    """
    remaining = list(range(2, N))  # photon labels
    random.shuffle(remaining)
    factors: list[str] = []

    while remaining:
        r = len(remaining)
        ops: list[str] = []
        if r >= 4:
            ops.append("tr4")
        if r >= 3:
            ops.extend(["tr3", "tripleF"])
        if r >= 2:
            ops.extend(["tr2", "doubleF"])
        if r >= 1:
            ops.append("singleF")

        kind = random.choice(ops)

        if kind == "tr4":
            chosen = random.sample(remaining, 4)
            factors.append(_tr4(*chosen))
            remaining = [x for x in remaining if x not in chosen]
        elif kind == "tr3":
            chosen = random.sample(remaining, 3)
            factors.append(_tr3(*chosen))
            remaining = [x for x in remaining if x not in chosen]
        elif kind == "tripleF":
            chosen = random.sample(remaining, 3)
            factors.append(_tripleF(*chosen, N))
            remaining = [x for x in remaining if x not in chosen]
        elif kind == "tr2":
            j, k = random.sample(remaining, 2)
            factors.append(_tr2(j, k))
            remaining = [x for x in remaining if x not in (j, k)]
        elif kind == "doubleF":
            j, k = random.sample(remaining, 2)
            factors.append(_doubleF(j, k, N))
            remaining = [x for x in remaining if x not in (j, k)]
        else:
            j = remaining.pop()
            factors.append(_singleF(j, N))

    for _ in range(random.randint(0, EXTRA_SCALARS_MAX)):
        factors.append(_scalar(N))

    random.shuffle(factors)
    return "*".join(factors)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §9  Scramblers                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# Design: every scrambler returns a string that is algebraically equal
# to the input on the support of momentum conservation + Ward identity.
# We remove the "add a free-floating zero" scramblers (trivially detectable
# by a network) and replace them with substitutions that act *inside* the
# expression, restructuring its tree.


def scr_mul_by_one(expr: str, N: int) -> str:
    """Multiply by  (p_i · p_j) / (p_i · p_j) = 1."""
    i, j = random.sample(range(1, N + 1), 2)
    one = f"({dot(p(i), p(j))})/({dot(p(i), p(j))})"
    return f"({expr})*{one}"


def scr_ward_substitute(expr: str, Ngamma: int, N: int) -> str:
    """Replace one occurrence of  e_i · p_k  with  −Σ_{s≠k} e_i · p_s.

    Valid because  Σ_s e_i · p_s = 0  (Ward identity / transversality + mom. cons.).
    """
    i = random.randint(2, Ngamma + 1)
    k = random.randint(1, N)
    target = re.escape(dot(e(i), p(k)))
    replacement = "-(" + " + ".join(dot(e(i), p(s)) for s in range(1, N + 1) if s != k) + ")"
    return re.sub(target, replacement, expr, count=1)


def scr_momentum_substitute(expr: str, N: int) -> str:
    """Replace one occurrence of  p_a · p_k  with  −Σ_{s≠a} p_s · p_k.

    Valid because  Σ_s p_s = 0  implies  p_a · p_k = −Σ_{s≠a} p_s · p_k.
    """
    matches = list(_RE_pp.finditer(expr))
    if not matches:
        return expr
    m = random.choice(matches)
    a, k = int(m.group(1)), int(m.group(2))
    replacement = "-(" + " + ".join(dot(p(s), p(k)) for s in range(1, N + 1) if s != a) + ")"
    return expr[: m.start()] + replacement + expr[m.end() :]


def scr_commute_dot(expr: str, Ngamma: int, N: int) -> str:
    """Flip the order of one random dot product (symmetric under η)."""
    matches = list(_RE_DOT.finditer(expr))
    if not matches:
        return expr
    m = random.choice(matches)
    a, b = m.group(1), m.group(2)
    return expr[: m.start()] + dot(b, a) + expr[m.end() :]


def scr_mul_by_ratio(expr: str, N: int) -> str:
    """Multiply numerator and denominator by a different Mandelstam invariant.

    Unlike scr_mul_by_one which uses p_i·p_j / p_i·p_j, this uses:
      (p_i·p_j + p_k·p_l) / (p_i·p_j + p_k·p_l)
    creating a structurally richer denominator.
    """
    legs = random.sample(range(1, N + 1), min(4, N))
    if len(legs) < 4:
        return scr_mul_by_one(expr, N)
    i, j, k, l = legs
    numer = f"({dot(p(i), p(j))} + {dot(p(k), p(l))})"
    return f"({expr})*{numer}/{numer}"


def scr_double_ward(expr: str, Ngamma: int, N: int) -> str:
    """Apply Ward substitution twice on different photon indices."""
    out = scr_ward_substitute(expr, Ngamma, N)
    # Second substitution on a different photon if possible
    if Ngamma >= 2:
        out = scr_ward_substitute(out, Ngamma, N)
    return out


# ── Partial fraction helpers ─────────────────────────────────────────

def _split_top_level(s: str, sep: str) -> list[str]:
    """Split string s by `sep`, but only at parenthesis depth 0.

    For example, _split_top_level("(a+b)*c*(d+e)", "*")
    returns ["(a+b)", "c", "(d+e)"].
    """
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
    """Find all  /(…)  blocks in `expr`.

    Returns a list of (slash_pos, open_paren_pos, close_paren_pos).
    The denominator content is  expr[open+1 : close].
    """
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
    """If expr[end-1] == ')', walk backward to find the matching '('.

    Returns the index of the opening '(', or None if no match.
    """
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


def scr_partial_fraction(expr: str, N: int) -> str:
    """Apply partial fraction decomposition to a propagator product.

    For an expression containing  (NUM)/(…*D_a*D_b*…), rewrite using the
    algebraic identity

        1/(D_a D_b) = 1/(D_a − D_b) · (1/D_b − 1/D_a)

    so that

        (NUM) / (D_a * D_b * R)
            →  (NUM) / ((D_a−D_b) * D_b * R)  −  (NUM) / ((D_a−D_b) * D_a * R)

    This splits one fraction into a sum of two fractions whose denominators
    involve *differences* of Mandelstam invariants — the kind of structure
    that appears naturally in Feynman-diagram expansions and is genuinely
    hard to undo.
    """
    # ── Step 1: locate all  /(denom)  blocks ─────────────────────────
    denom_blocks = _find_denom_blocks(expr)
    if not denom_blocks:
        return expr

    # ── Step 2: find blocks with ≥2 splittable p·p factors AND an
    #            identifiable parenthesised numerator just before the /
    viable: list[tuple] = []
    for slash_pos, dopen, dclose in denom_blocks:
        denom_str = expr[dopen + 1 : dclose]
        factors = _split_top_level(denom_str, "*")
        pp_idx = [i for i, f in enumerate(factors) if _RE_pp.fullmatch(f.strip())]
        if len(pp_idx) < 2:
            continue
        # The numerator should be a parenthesised block ending at slash_pos
        num_start = _find_paren_block_ending_at(expr, slash_pos)
        if num_start is None:
            continue
        viable.append((slash_pos, dopen, dclose, factors, pp_idx, num_start))

    if not viable:
        return expr

    # ── Step 3: pick one block, pick two denominator factors ─────────
    slash_pos, dopen, dclose, factors, pp_idx, num_start = random.choice(viable)

    ia, ib = random.sample(pp_idx, 2)
    Da = factors[ia].strip()
    Db = factors[ib].strip()
    rest = [f.strip() for i, f in enumerate(factors) if i not in (ia, ib)]

    NUM = expr[num_start:slash_pos]          # includes outer ( … )
    diff = f"({Da} - {Db})"

    if rest:
        rest_str = "*".join(rest)
        den1 = f"({diff}*{Db}*{rest_str})"
        den2 = f"({diff}*{Da}*{rest_str})"
    else:
        den1 = f"({diff}*{Db})"
        den2 = f"({diff}*{Da})"

    # ── Step 4: reconstruct the expression ───────────────────────────
    prefix = expr[:num_start]
    suffix = expr[dclose + 1 :]
    new_frac = f"({NUM}/{den1} - {NUM}/{den2})"
    return f"{prefix}{new_frac}{suffix}"


_SCRAMBLERS = [
    lambda e, Ng, Nt: scr_mul_by_one(e, Nt),
    lambda e, Ng, Nt: scr_ward_substitute(e, Ng, Nt),
    lambda e, Ng, Nt: scr_momentum_substitute(e, Nt),
    lambda e, Ng, Nt: scr_commute_dot(e, Ng, Nt),
    lambda e, Ng, Nt: scr_mul_by_ratio(e, Nt),
    lambda e, Ng, Nt: scr_double_ward(e, Ng, Nt),
    lambda e, Ng, Nt: scr_partial_fraction(e, Nt),
]


def scramble(
    expr: str,
    Ngamma: int,
    N: int,
    min_scr: int = 0,
    max_scr: int = 3,
    max_len: int = 4000,
) -> str:
    """Apply min_scr…max_scr random scramblers, capping output length."""
    min_scr = max(0, int(min_scr))
    max_scr = max(min_scr, int(max_scr))
    n = random.randint(min_scr, max_scr) if max_scr > 0 else 0
    out = expr
    for _ in range(n):
        cand = random.choice(_SCRAMBLERS)(out, Ngamma, N)
        if len(cand) <= max_len:
            out = cand
    return out


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §10  Mass dimension & signature utilities                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _numerator_signature(simple_num: str) -> tuple:
    """Factor-type signature for matching polynomial term dimensions.

    Returns (#TrN_by_len, #pFchainp_by_len, #pp) so that two monomials with
    the same signature have the same mass dimension.
    """
    if not simple_num:
        return ()
    tr_lengths: list[int] = []
    chain_lengths: list[int] = []
    pp = 0
    for f in simple_num.split("*"):
        f = f.strip()
        m = _RE_TrN.fullmatch(f)
        if m:
            n_Fs = len(re.findall(r"F_\d+", m.group(1)))
            tr_lengths.append(n_Fs)
            continue
        m = _RE_pFchainp.fullmatch(f)
        if m:
            n_Fs = len(re.findall(r"F_\d+", m.group(2)))
            chain_lengths.append(n_Fs)
            continue
        if _RE_pp.fullmatch(f):
            pp += 1
            continue
    # Sort the lists so the signature is permutation-invariant
    return (tuple(sorted(tr_lengths)), tuple(sorted(chain_lengths)), pp)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §11  Dataset construction                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _random_denominator(N: int, base_leg: int = 1) -> str:
    k = random.randint(1, min(3, N - 1))
    js = random.sample([j for j in range(1, N + 1) if j != base_leg], k)
    return "*".join(dot(p(base_leg), p(j)) for j in js)


def _gauge_denominator(N: int) -> str:
    """Π_i (p_ref · p_i) over photon legs, using a scalar as reference."""
    photons = list(range(2, N))
    if not photons:
        return ""
    ref = random.choice([1, N])
    return "*".join(dot(p(ref), p(i)) for i in photons)


def _generate_monomial(N: int, use_denom: bool) -> tuple[str, str, int, bool, tuple]:
    """Return (simple_term, expanded_term, denom_len, has_denom, signature)."""
    gi = strict_gi_monomial(N)
    simple_num = canonicalise_gi_product(gi)
    den_parts: list[str] = []
    if use_denom and random.random() < 0.6:
        den_parts.append(_random_denominator(N, base_leg=1))
    den_parts.append(_gauge_denominator(N))
    den_parts = [d for d in den_parts if d]

    if den_parts:
        denom = canonicalise_denominator("*".join(den_parts))
        simple_term = f"({simple_num})/({denom})"
        expd_num = "*".join(rewrite_gi(b) for b in simple_num.split("*"))
        expd_term = f"({expd_num})/({denom})"
        return simple_term, expd_term, len(denom.split("*")), True, _numerator_signature(simple_num)
    else:
        expd = "*".join(rewrite_gi(b) for b in simple_num.split("*"))
        return simple_num, expd, 0, False, _numerator_signature(simple_num)


def _validate_pair(
    expr_a: str,
    expr_b: str,
    N: int,
    M: float,
    n_checks: int = 2,
    tol_rel: float = 1e-8,
    tol_abs: float = 1e-10,
) -> tuple[bool, str]:
    """Numerically verify that two expressions agree on random kinematics."""
    for _ in range(n_checks):
        mom, pol = generate_kinematics(N, M=M)
        try:
            va = eval_infix_numeric(expr_a, mom, pol)
            vb = eval_infix_numeric(expr_b, mom, pol)
        except Exception as ex:
            return False, f"exception:{ex}"
        if not (math.isfinite(va) and math.isfinite(vb)):
            return False, "non-finite"
        diff = abs(va - vb)
        if diff > max(tol_abs, tol_rel * max(1.0, abs(va))):
            return False, f"mismatch|Δ={diff:.3e}"
    return True, ""


def _format_poly(terms: list[str], coeffs: list[int]) -> str:
    """Format a polynomial from terms and integer coefficients."""
    parts: list[str] = []
    for i, (t, c) in enumerate(zip(terms, coeffs)):
        ac = abs(c)
        sign = "-" if c < 0 else "+"
        core = t if ac == 1 else f"{ac}*{t}"
        if i == 0:
            parts.append(f"-{core}" if sign == "-" else core)
        else:
            parts.append(f" {sign} {core}")
    return "".join(parts)


def build_dataset(
    N: int,
    num_samples: int,
    *,
    max_scr: int = 3,
    min_scr: int = 0,
    seed: int | None = None,
    use_denominators: bool = True,
    validate: bool = True,
    M: float = 2.0,
    min_terms: int = 1,
    max_terms: int = 1,
    log_path: str | None = None,
    max_attempts_factor: int = 10,
) -> list[tuple[str, str]]:
    """Build a dataset of (simple, scrambled) expression pairs.

    When max_terms > 1, each sample is a polynomial whose terms share the
    same mass-dimension signature (strictly enforced — mismatches are discarded,
    never silently relaxed).
    """
    min_terms = max(1, int(min_terms))
    max_terms = max(min_terms, int(max_terms))
    if seed is not None:
        random.seed(seed)

    Ngamma = N - 2
    data: list[tuple[str, str]] = []
    stats = {"attempts": 0, "parity_fail": 0, "scramble_fail": 0}
    max_attempts = num_samples * max_attempts_factor

    if log_path:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"# gen_data log  N={N}  target={num_samples}  "
                     f"terms=[{min_terms},{max_terms}]  scr=[{min_scr},{max_scr}]  seed={seed}\n")

    while len(data) < num_samples and stats["attempts"] < max_attempts:
        stats["attempts"] += 1
        T = random.randint(min_terms, max_terms)

        # Generate first term
        s0, e0, dlen0, hd0, sig0 = _generate_monomial(N, use_denominators)
        terms_s = [s0]
        terms_e = [e0]
        coeffs = [random.choice([c for c in range(-9, 10) if c != 0])]

        # Subsequent terms must match dimension signature exactly
        ok_build = True
        for _ in range(T - 1):
            matched = False
            for _attempt in range(50):
                sx, ex, dlen, hd, sig = _generate_monomial(N, use_denominators)
                if hd == hd0 and dlen == dlen0 and sig == sig0:
                    matched = True
                    break
            if not matched:
                ok_build = False
                break
            terms_s.append(sx)
            terms_e.append(ex)
            coeffs.append(random.choice([c for c in range(-100, 101) if c != 0]))

        if not ok_build:
            continue

        simple_poly = _format_poly(terms_s, coeffs)
        expanded_poly = _format_poly(terms_e, coeffs)

        # Validate parity: simple (with GI blocks) ≡ expanded (e·p only)
        if validate:
            ok, reason = _validate_pair(simple_poly, expanded_poly, N, M)
            if not ok:
                stats["parity_fail"] += 1
                continue

        # Scramble the expanded form
        scrambled = scramble(expanded_poly, Ngamma, N, min_scr, max_scr)

        # Validate scramble
        if validate:
            ok, reason = _validate_pair(expanded_poly, scrambled, N, M)
            if not ok:
                stats["scramble_fail"] += 1
                continue

        data.append((simple_poly, scrambled))

    if log_path:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"# SUMMARY  accepted={len(data)}  "
                     f"parity_fail={stats['parity_fail']}  "
                     f"scramble_fail={stats['scramble_fail']}  "
                     f"attempts={stats['attempts']}\n")

    return data


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §12  Deduplication                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def dedupe_pairs(
    pairs: list[tuple[str, str]], keep: str = "first"
) -> tuple[list[tuple[str, str]], int]:
    if not pairs:
        return pairs, 0
    if keep == "first":
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for item in pairs:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out, len(pairs) - len(out)
    elif keep == "last":
        last_idx = {item: i for i, item in enumerate(pairs)}
        out = [item for i, item in enumerate(pairs) if last_idx[item] == i]
        return out, len(pairs) - len(out)
    else:
        raise ValueError("keep must be 'first' or 'last'")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §13  I/O                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def write_csv(pairs: list[tuple[str, str]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["simple", "scrambled"])
        for s, t in pairs:
            w.writerow([s, t])


def tokenise_csv(inp: str, out: str, max_particles: int = 8) -> None:
    from Tokenizer import ScatteringAmplitudeTokenizer

    tok = ScatteringAmplitudeTokenizer(max_particles=max_particles)
    with open(inp, newline="", encoding="utf-8") as fi, \
         open(out, "w", newline="", encoding="utf-8") as fo:
        r = csv.reader(fi)
        w = csv.writer(fo)
        w.writerow(["simple", "scrambled"])
        first = True
        for row in r:
            if not row:
                continue
            if first and row[0].strip().lower() == "simple":
                first = False
                continue
            first = False
            w.writerow([
                json.dumps(tok.encode_infix(row[0])),
                json.dumps(tok.encode_infix(row[1]) if len(row) > 1 else []),
            ])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  §14  CLI driver                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate amplitude training data.")
    parser.add_argument("N", nargs="?", type=int, default=6,
                        help="Number of external legs (default: 6)")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--max-scr", type=int, default=5)
    parser.add_argument("--min-scr", type=int, default=0)
    parser.add_argument("--min-terms", type=int, default=1)
    parser.add_argument("--max-terms", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-tokenise", action="store_true")
    args = parser.parse_args()

    N = args.N
    RAW = f"gi_{N}pt.csv"
    TOK = f"gi_{N}pt_tok.csv"
    LOG = f"gen_data_{N}pt.log"

    t0 = time.perf_counter()
    oversample = int(round(args.samples * 1.2))
    pairs = build_dataset(
        N, oversample,
        max_scr=args.max_scr, min_scr=args.min_scr,
        seed=args.seed, use_denominators=True,
        validate=not args.no_validate,
        min_terms=args.min_terms, max_terms=args.max_terms,
        log_path=LOG,
    )
    t1 = time.perf_counter()

    before = len(pairs)
    pairs, removed = dedupe_pairs(pairs)
    if len(pairs) > args.samples:
        pairs = pairs[: args.samples]
    write_csv(pairs, RAW)
    if not args.no_tokenise:
        tokenise_csv(RAW, TOK)
    t2 = time.perf_counter()

    print(f"{len(pairs)} pairs → {RAW}")
    print(f"  generation : {t1 - t0:.2f}s")
    print(f"  dedupe     : removed {removed} ({before} → {len(pairs)})")
    print(f"  write+tok  : {t2 - t1:.2f}s")
    print(f"  log        : {LOG}")
