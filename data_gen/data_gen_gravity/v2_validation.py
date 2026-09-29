"""Strict exact algebra and independent numerical/family release gates for v2.

Compact atoms are scalar contractions, not commuting matrices: the matrix
order is part of each atom.  Sparse Laurent dictionaries use exact Fractions.
No unknown syntax, missing parenthesis, unsupported node, or trailing input is
interpreted as zero.  Numerical samples are complementary checks, not proofs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import math
import re
from typing import Mapping, Sequence

import numpy as np
import sympy as sp

from .core import PROCESS_SPECS
from .kinematics import SpinorKinematics, generate_kinematics, mdot, with_references

Atom = tuple
Monomial = tuple[tuple[Atom, int], ...]
Laurent = dict[Monomial, Fraction]


class ParseError(ValueError):
    """An expression is malformed or outside the supported exact grammar."""


_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([peF]_[1-5])|(Tr)|(\*\*|[+*/^().·,-]))")


def _tokens(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty expression")
    pos, out = 0, []
    while pos < len(text):
        if not text[pos:].strip():
            break
        match = _TOKEN.match(text, pos)
        if match is None:
            raise ParseError(f"unsupported syntax at character {pos}: {text[pos:pos+24]!r}")
        out.append(next(x for x in match.groups() if x is not None))
        pos = match.end()
    return tuple(out)


def _canonical_atom(atom: Atom) -> tuple[int, Atom]:
    kind, *legs = atom
    if kind in ("d", "ee"):
        return 1, (kind, *sorted(legs))
    if kind == "ep":
        return 1, atom
    if kind == "X":
        i, a, b = legs
        if a == b:
            return 0, atom
        return (1, atom) if a < b else (-1, (kind, i, b, a))
    if kind == "Q":
        i, j, a, b = legs
        other = (kind, j, i, b, a)
        return 1, min(atom, other)
    if kind == "T":
        n = len(legs)
        forward = [tuple(legs[k:] + legs[:k]) for k in range(n)]
        backward = list(reversed(legs))
        reverse = [tuple(backward[k:] + backward[:k]) for k in range(n)]
        best = min(forward + reverse)
        if n % 2 and best in forward and best in reverse:
            return 0, (kind, *best)
        return (1 if best in forward else (-1)**n), (kind, *best)
    if kind == "C":
        # (C, endpoint-kind, endpoint-leg, F-leg,..., endpoint-kind, endpoint-leg)
        first, a, *middle, last, b = legs
        other = (kind, last, b, *reversed(middle), first, a)
        if atom == other and len(middle) % 2:
            return 0, atom
        return (1, atom) if atom < other else ((-1)**len(middle), other)
    raise ParseError(f"unsupported atom: {atom!r}")


def _dot_atom(a: tuple[str, int], b: tuple[str, int]) -> Atom:
    if a[0] == b[0] == "p":
        return ("d", *sorted((a[1], b[1])))
    if a[0] == b[0] == "e":
        return ("ee", *sorted((a[1], b[1])))
    if a[0] == "p":
        a, b = b, a
    return ("ep", a[1], b[1])


def _chain_atom(parts: Sequence[tuple[str, int]], trace: bool = False) -> tuple[int, Atom]:
    if trace:
        if len(parts) < 2 or any(k != "F" for k, _ in parts):
            raise ParseError("trace requires at least two field strengths")
        return _canonical_atom(("T", *(i for _, i in parts)))
    if len(parts) == 2 and all(k in ("p", "e") for k, _ in parts):
        return 1, _dot_atom(parts[0], parts[1])
    if (len(parts) < 3 or parts[0][0] not in ("p", "e")
            or parts[-1][0] not in ("p", "e")
            or any(k != "F" for k, _ in parts[1:-1])):
        raise ParseError("expected two vector endpoints enclosing only ordered F factors")
    first, last = parts[0], parts[-1]
    fs = [i for _, i in parts[1:-1]]
    if first[0] == last[0] == "p" and len(fs) <= 2:
        atom = ("X" if len(fs) == 1 else "Q", *fs, first[1], last[1])
    else:
        atom = ("C", *first, *fs, *last)
    return _canonical_atom(atom)


class _StrictParser:
    def __init__(self, text: str):
        self.tokens, self.i = _tokens(text), 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ParseError(f"expected {expected or 'expression'}, got {token!r}")
        self.i += 1
        return token

    def parse(self):
        node = self.expr()
        if self.i != len(self.tokens):
            raise ParseError(f"unconsumed input starting at {self.peek()!r}")
        return node

    def expr(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() in ("*", "/"):
            op = self.take()
            node = (op, node, self.unary())
        return node

    def unary(self):
        if self.peek() in ("+", "-"):
            op = self.take()
            child = self.unary()
            return child if op == "+" else ("neg", child)
        return self.power()

    def power(self):
        node = self.primary()
        if self.peek() in ("^", "**"):
            self.take()
            exponent = self.unary()
            value = _constant_value(exponent)
            if value.denominator != 1 or abs(value) > 32:
                raise ParseError("power must be an integer in [-32, 32]")
            node = ("pow", node, int(value))
        return node

    def primary(self):
        token = self.peek()
        if token == "(":
            self.take()
            node = self.expr()
            self.take(")")
            return node
        trace = token == "Tr"
        if trace:
            self.take()
            self.take("(")
            token = self.peek()
        if token is not None and re.fullmatch(r"[peF]_[1-5]", token):
            parts = []
            while True:
                vec = self.take()
                if not re.fullmatch(r"[peF]_[1-5]", vec):
                    raise ParseError("missing vector in contraction")
                parts.append((vec[0], int(vec[2:])))
                if self.peek() not in ("·", "."):
                    break
                self.take()
            if trace:
                self.take(")")
            sign, atom = _chain_atom(parts, trace)
            return ("number", Fraction(0)) if sign == 0 else (("atom", atom) if sign == 1 else ("neg", ("atom", atom)))
        if trace:
            raise ParseError("empty or malformed trace")
        if token is not None and re.fullmatch(r"\d+(?:\.\d+)?", token):
            self.take()
            return ("number", Fraction(token))
        raise ParseError(f"expected scalar contraction or number, got {token!r}")


def _constant_value(node) -> Fraction:
    if node[0] == "number":
        return node[1]
    if node[0] == "neg":
        return -_constant_value(node[1])
    if node[0] == "pow":
        return _constant_value(node[1]) ** node[2]
    if node[0] in ("+", "-", "*", "/"):
        a, b = _constant_value(node[1]), _constant_value(node[2])
        if node[0] == "+": return a + b
        if node[0] == "-": return a - b
        if node[0] == "*": return a * b
        if not b: raise ParseError("division by zero")
        return a / b
    raise ParseError("nonconstant exponent")


def _clean(poly: Mapping) -> Laurent:
    return {m: Fraction(c) for m, c in poly.items() if c}


def add_laurent(a: Mapping, b: Mapping, scale=Fraction(1)) -> Laurent:
    out = dict(a)
    for monomial, coefficient in b.items():
        out[monomial] = out.get(monomial, Fraction(0)) + coefficient * scale
    return _clean(out)


def multiply_laurent(a: Mapping, b: Mapping) -> Laurent:
    out = defaultdict(Fraction)
    for ma, ca in a.items():
        for mb, cb in b.items():
            powers = dict(ma)
            for atom, power in mb:
                powers[atom] = powers.get(atom, 0) + power
            monomial = tuple(sorted((atom, n) for atom, n in powers.items() if n))
            out[monomial] += ca * cb
    return _clean(out)


def power_laurent(poly: Mapping, n: int) -> Laurent:
    if n < 0:
        if len(poly) != 1:
            raise ParseError("Laurent denominator must be a nonzero monomial")
        (monomial, coefficient), = poly.items()
        return {tuple((a, p*n) for a, p in monomial): coefficient**n}
    result = {(): Fraction(1)}
    while n:
        if n & 1:
            result = multiply_laurent(result, poly)
        n //= 2
        if n:
            poly = multiply_laurent(poly, poly)
    return result


def _node_laurent(node) -> Laurent:
    op = node[0]
    if op == "number": return {(): node[1]} if node[1] else {}
    if op == "atom": return {((node[1], 1),): Fraction(1)}
    if op == "neg": return {m: -c for m, c in _node_laurent(node[1]).items()}
    if op == "pow": return power_laurent(_node_laurent(node[1]), node[2])
    a, b = _node_laurent(node[1]), _node_laurent(node[2])
    if op == "+": return add_laurent(a, b)
    if op == "-": return add_laurent(a, b, Fraction(-1))
    if op == "*": return multiply_laurent(a, b)
    if op == "/": return multiply_laurent(a, power_laurent(b, -1))
    raise ParseError(f"unsupported operation {op}")


def parse_laurent(text: str | Mapping, *, expand_f: bool = False) -> Laurent:
    """Fully parse exact rational expressions with monomial denominators."""
    result = _clean(text) if isinstance(text, Mapping) else _node_laurent(_StrictParser(text).parse())
    return expand_laurent(result) if expand_f else result


@lru_cache(maxsize=4096)
def _expanded_atom(atom: Atom) -> Laurent:
    kind, *legs = atom
    if kind in ("d", "ep", "ee"):
        return {((atom, 1),): Fraction(1)}
    trace = kind == "T"
    if kind == "X":
        i, a, b = legs
        fs, left, right = [i], ("p", a), ("p", b)
    elif kind == "Q":
        i, j, a, b = legs
        fs, left, right = [i, j], ("p", a), ("p", b)
    elif kind == "T":
        fs, left, right = legs, None, None
    elif kind == "C":
        tag, a, *fs, tag2, b = legs
        left, right = (tag, a), (tag2, b)
    else:
        raise ParseError(f"unsupported atom {atom}")
    if len(fs) > 8:
        raise ParseError("field-strength chain exceeds supported length eight")
    out = {}
    for choices in itertools.product((0, 1), repeat=len(fs)):
        us = [("e" if flip else "p", i) for i, flip in zip(fs, choices)]
        vs = [("p" if flip else "e", i) for i, flip in zip(fs, choices)]
        dots = [_dot_atom(vs[k], us[k+1]) for k in range(len(fs)-1)]
        dots += [_dot_atom(vs[-1], us[0])] if trace else [_dot_atom(left, us[0]), _dot_atom(vs[-1], right)]
        counts = defaultdict(int)
        for dot in dots: counts[dot] += 1
        term = {tuple(sorted(counts.items())): Fraction((-1)**sum(choices))}
        out = add_laurent(out, term)
    return out


def expand_laurent(poly: Mapping) -> Laurent:
    """Expand ordered field strengths exactly over independent dot variables."""
    out = {}
    for monomial, coefficient in poly.items():
        term = {(): Fraction(coefficient)}
        for atom, exponent in monomial:
            term = multiply_laurent(term, power_laurent(_expanded_atom(atom), exponent))
        out = add_laurent(out, term)
    return out


def _sympy_atom(atom: Atom):
    return sp.Symbol("_".join(map(str, atom)))


def laurent_to_sympy(poly: Mapping):
    return sp.Add(*(sp.Rational(c.numerator, c.denominator) * sp.Mul(*(_sympy_atom(a)**n for a, n in m)) for m, c in poly.items()))


def parse_exact(text: str | Mapping, *, expand_f: bool = False):
    """Strict full-consumption parser, with exact rational coefficients."""
    if isinstance(text, Mapping):
        return laurent_to_sympy(expand_laurent(text) if expand_f else text)
    def visit(node):
        op = node[0]
        if op == "number": return sp.Rational(node[1].numerator, node[1].denominator)
        if op == "atom": return laurent_to_sympy(_expanded_atom(node[1])) if expand_f else _sympy_atom(node[1])
        if op == "neg": return -visit(node[1])
        if op == "pow": return visit(node[1])**node[2]
        a, b = visit(node[1]), visit(node[2])
        if op == "+": return a+b
        if op == "-": return a-b
        if op == "*": return a*b
        if op == "/":
            if b == 0: raise ParseError("division by zero")
            return a/b
        raise ParseError(op)
    result = visit(_StrictParser(text).parse())
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ParseError("nonfinite expression")
    return result


def exact_equivalent(left, right) -> bool:
    try:
        return expand_laurent(parse_laurent(left)) == expand_laurent(parse_laurent(right))
    except ParseError:
        return sp.cancel(parse_exact(left, expand_f=True)-parse_exact(right, expand_f=True)) == 0


def expand_field_strengths(expression):
    """Expand our compact SymPy symbols, or parse and expand a string."""
    if isinstance(expression, (str, Mapping)):
        return parse_exact(expression, expand_f=True)
    substitutions = {}
    for symbol in expression.free_symbols:
        pieces = str(symbol).split("_")
        try:
            atom = (pieces[0], *(int(x) if x.isdigit() else x for x in pieces[1:]))
            substitutions[symbol] = laurent_to_sympy(_expanded_atom(atom))
        except (ValueError, TypeError) as exc:
            raise ParseError(f"unsupported symbol {symbol}") from exc
    return expression.xreplace(substitutions)


def _atom_degree(atom: Atom) -> tuple[int, dict[int, int]]:
    kind, *legs = atom
    if kind == "d": return 2, {}
    if kind == "ep": return 1, {legs[0]: 1}
    if kind == "ee":
        d = defaultdict(int)
        for leg in legs: d[leg] += 1
        return 0, dict(d)
    if kind == "X": fs, dimension = legs[:1], 3
    elif kind == "Q": fs, dimension = legs[:2], 4
    elif kind == "T": fs, dimension = legs, len(legs)
    elif kind == "C":
        first, a, *fs, last, b = legs
        d = defaultdict(int)
        for leg in fs: d[leg] += 1
        if first == "e": d[a] += 1
        if last == "e": d[b] += 1
        return len(fs) + (first == "p") + (last == "p"), dict(d)
    else: raise ParseError(f"unsupported atom {atom}")
    d = defaultdict(int)
    for leg in fs: d[leg] += 1
    return dimension, dict(d)


def check_process(expression, process: str) -> dict:
    poly = parse_laurent(expression)
    if not poly:
        raise ValueError("identically zero target")
    spec = PROCESS_SPECS[process]
    expected = {leg: 2 for leg in spec.graviton_legs}
    shapes = []
    for monomial in poly:
        dimension, degrees = 0, defaultdict(int)
        for atom, exponent in monomial:
            dim, deg = _atom_degree(atom)
            dimension += dim * exponent
            for leg, degree in deg.items(): degrees[leg] += degree * exponent
        shape = (dimension, {k: v for k, v in degrees.items() if v})
        if shape != (spec.target_dimension, expected):
            raise ValueError(f"wrong process degree/dimension: {shape}, expected {(spec.target_dimension, expected)}")
        shapes.append(shape)
    return {"mass_dimension": spec.target_dimension, "polarization_degrees": expected, "monomials": len(poly)}


def _relabel_atom(atom: Atom, mapping: Mapping[int, int]):
    return _canonical_atom((atom[0], *(mapping[x] if isinstance(x, int) else x for x in atom[1:])))


def relabel_laurent(poly: Mapping, mapping: Mapping[int, int]) -> Laurent:
    out = {}
    for monomial, coefficient in poly.items():
        powers = defaultdict(int)
        for atom, power in monomial:
            sign, new = _relabel_atom(atom, mapping)
            coefficient *= sign**power
            powers[new] += power
        new_monomial = tuple(sorted((a, n) for a, n in powers.items() if n))
        out = add_laurent(out, {new_monomial: coefficient})
    return out


@lru_cache(maxsize=2)
def species_mappings(process: str):
    spec = PROCESS_SPECS[process]
    return tuple(dict(zip(spec.scalar_legs + spec.graviton_legs, s + h)) for s in itertools.permutations(spec.scalar_legs) for h in itertools.permutations(spec.graviton_legs))


def canonical_family(expression, process: str, *, ignore_coefficients: bool = True) -> str:
    """Species-canonical structural family; coefficient descendants stay together.

    This is deliberately conservative when ignoring coefficients. General
    algebraic equivalence is additionally checked by independent fingerprints.
    """
    poly = parse_laurent(expression)
    if not poly:
        raise ValueError("zero has no nonzero projective family")
    candidates = []
    for mapping in species_mappings(process):
        relabeled = relabel_laurent(poly, mapping)
        terms = sorted(relabeled.items())
        if ignore_coefficients:
            candidates.append(repr(tuple(m for m, _ in terms)))
        else:
            pivot = terms[0][1]
            candidates.append(repr(tuple((m, str(c/pivot)) for m, c in terms)))
    return hashlib.sha256((process + ":" + min(candidates)).encode()).hexdigest()


def _general_polarizations(base, legs, seed):
    rng = np.random.default_rng(seed)
    pols = {}
    for leg in legs:
        p = base.momenta[leg-1]
        q = max((base.momenta[j-1] for j in range(1, 6) if j != leg), key=lambda q: abs(mdot(p, q)))
        v = rng.normal(size=4) + 1j*rng.normal(size=4)
        pols[leg] = v - mdot(v, p)/mdot(q, p)*q
    return SpinorKinematics(base.lambdas, base.tildes, base.momenta, pols, {})


class NumericalContext:
    """Vectorized values over independent momenta, references and gauge shifts."""
    def __init__(self, process: str, *, seed=730003, momentum_samples=3, include_general=True):
        self.process, self.seed = process, seed
        self.points, self.labels = [], []
        legs = PROCESS_SPECS[process].graviton_legs
        for j in range(momentum_samples):
            point_seed = seed + 1009*j
            base = generate_kinematics(seed=point_seed, graviton_legs=legs, min_invariant=0.05)
            for mode in ("first", "last", "random", "gauge_shift"):
                shifted = mode == "gauge_shift"
                self.points.append(with_references(base, legs, reference_mode="cyclic" if shifted else mode, seed=point_seed+71, gauge_shifts={i: complex(.19*(i+1), -.07*i) for i in legs} if shifted else None))
                self.labels.append(f"{point_seed}:{mode}")
            if include_general:
                general = _general_polarizations(base, legs, point_seed + 809)
                self.points.append(general)
                self.labels.append(f"{point_seed}:general_transverse")
        self._atoms = {}

    def atom(self, atom: Atom):
        if atom in self._atoms:
            return self._atoms[atom]
        kind, *legs = atom
        if kind not in ("d", "ep", "ee"):
            value = self.evaluate(_expanded_atom(atom))
        else:
            value = []
            for kin in self.points:
                def p(i): return kin.momenta[i-1]
                def e(i): return kin.polarisations[i]
                a, b = legs
                value.append(mdot(p(a), p(b)) if kind == "d" else mdot(e(a), p(b)) if kind == "ep" else mdot(e(a), e(b)))
            value = np.asarray(value, dtype=np.complex128)
        self._atoms[atom] = value
        return value

    def evaluate(self, expression, *, absolute_bound=False):
        poly = parse_laurent(expression) if not isinstance(expression, Mapping) else expression
        value = np.zeros(len(self.points), dtype=float if absolute_bound else complex)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for monomial, coefficient in poly.items():
                term = np.full(len(self.points), abs(float(coefficient)) if absolute_bound else float(coefficient), dtype=float if absolute_bound else complex)
                for atom, exponent in monomial:
                    factor = self.atom(atom)
                    if absolute_bound: factor = np.abs(factor)
                    term *= factor ** exponent
                value += term
        return value


def _comparison(a, b, atol, rtol):
    # Match the repository's absolute-or-relative gate.
    delta = np.abs(a-b)
    scale = np.maximum(np.abs(a), np.abs(b))
    relative = np.divide(delta, scale, out=delta.copy(), where=scale != 0)
    return bool(np.all((delta <= atol) | (relative <= rtol))), float(np.max(relative))


class PairValidator:
    def __init__(self, *, seed=730003, momentum_samples=3, rtol=2e-8, atol=2e-9, max_resamples=3):
        self.seed, self.momentum_samples = seed, momentum_samples
        self.rtol, self.atol, self.max_resamples = rtol, atol, max_resamples
        self.contexts = {}

    def context(self, process, attempt=0):
        key = process, attempt
        if key not in self.contexts:
            self.contexts[key] = NumericalContext(process, seed=self.seed + 100003*attempt, momentum_samples=self.momentum_samples)
        return self.contexts[key]

    def validate(self, simple, scrambled, process: str, *, check_degrees=True) -> dict:
        try:
            a, b = parse_laurent(simple), parse_laurent(scrambled)
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            return {"ok": False, "reason": "parse_error", "detail": str(exc)}
        if check_degrees:
            try:
                check_process(a, process)
                check_process(b, process)
            except (ValueError, KeyError) as exc:
                return {"ok": False, "reason": "process_degree", "detail": str(exc)}
        for attempt in range(self.max_resamples + 1):
            try:
                context = self.context(process, attempt)
                av, bv = context.evaluate(a), context.evaluate(b)
                finite = np.isfinite(av) & np.isfinite(bv)
                ab, bb = context.evaluate(a, absolute_bound=True), context.evaluate(b, absolute_bound=True)
                # A cancellation condition number above 1e10 is resampled;
                # tolerances are never relaxed to make a point pass.
                cond = np.maximum(ab/np.maximum(np.abs(av), 1e-300), bb/np.maximum(np.abs(bv), 1e-300))
                if not np.all(finite) or np.any(cond > 1e10):
                    continue
                if np.all(np.abs(av) <= 1e-12):
                    return {"ok": False, "reason": "zero_target", "samples": len(av)}
                ok, worst = _comparison(av, bv, self.atol, self.rtol)
                return {"ok": ok, "reason": "ok" if ok else "numerical_mismatch", "worst_relative_error": worst, "samples": len(av), "resamples": attempt, "general_polarization_samples": self.momentum_samples, "seed": self.seed + 100003*attempt}
            except (ValueError, ZeroDivisionError, FloatingPointError, KeyError) as exc:
                return {"ok": False, "reason": "numerical_error", "detail": str(exc)}
        return {"ok": False, "reason": "singular_or_ill_conditioned", "resamples": self.max_resamples}


def _projective_match(a, b, rtol=2e-8, atol=2e-9):
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return False
    pivot = int(np.argmax(np.abs(b)))
    if abs(b[pivot]) < 1e-14 or abs(a[pivot]) < 1e-14:
        return False
    ratio = a[pivot]/b[pivot]
    return _comparison(a, ratio*b, atol, rtol)[0]


class FrozenFamilyGuard:
    """Reserve reference descendants using structural and projective checks.

    The structural guard conservatively excludes independent coefficient
    variations of each reference skeleton. The numerical guard compares every
    species relabeling against all independent samples, allowing nonzero scale.
    It also conservatively reserves positive-helicity-only matches. This protects
    same-helicity benchmark descendants without claiming their identity holds
    for general polarizations or that finite fingerprints prove inequivalence.
    """
    def __init__(self, references: Mapping[str, Sequence[str]], *, seed=910009, momentum_samples=3):
        self.seed, self.momentum_samples = seed, momentum_samples
        self.contexts = {}
        self.signatures = defaultdict(set)
        self.skeleton_signatures = defaultdict(set)
        self.fingerprints = defaultdict(list)
        self._matrices = {}
        self.add_references(references)

    def add_references(self, references: Mapping[str, Sequence]):
        """Add held-out origins before training variants are constructed."""
        for process, expressions in references.items():
            if process not in self.contexts:
                self.contexts[process] = NumericalContext(process, seed=self.seed, momentum_samples=self.momentum_samples)
            for expression in expressions:
                poly = parse_laurent(expression)
                self.signatures[process].add(canonical_family(poly, process, ignore_coefficients=False))
                self.skeleton_signatures[process].add(canonical_family(poly, process))
                for mapping in species_mappings(process):
                    relabeled = relabel_laurent(poly, mapping)
                    values = self.contexts[process].evaluate(relabeled)
                    if not np.all(np.isfinite(values)):
                        raise ValueError("nonfinite frozen-family fingerprint")
                    self.fingerprints[process].append(values)
            matrix = np.asarray(self.fingerprints[process], dtype=np.complex128)
            self._matrices[process] = matrix.reshape((-1, len(self.contexts[process].points)))

    def is_reserved(self, expression, process: str) -> bool:
        poly = parse_laurent(expression)
        if canonical_family(poly, process) in self.skeleton_signatures[process]:
            return True
        return self.matches_fingerprint(self.contexts[process].evaluate(poly), process)

    def matches_fingerprint(self, values: np.ndarray, process: str) -> bool:
        """Vectorized scale comparison; unrounded full vectors are decisive."""
        matrix = self._matrices[process]
        values = np.asarray(values, dtype=np.complex128)
        positive = [i for i, label in enumerate(self.contexts[process].labels)
                    if "general_transverse" not in label]
        return (self._matches_grid(values, matrix)
                or self._matches_grid(values[positive], matrix[:, positive]))

    @staticmethod
    def _matches_grid(values, matrix):
        if matrix.shape[0] == 0 or not np.all(np.isfinite(values)):
            return False
        pivot = int(np.argmax(np.abs(values)))
        if abs(values[pivot]) < 1e-14:
            return False
        mask = np.abs(matrix[:, pivot]) >= 1e-14
        if not np.any(mask):
            return False
        active = np.flatnonzero(mask)
        ratios = values[pivot] / matrix[active, pivot]
        # Use points spread over independent momenta/polarizations as cheap
        # preliminary filters; every surviving candidate is checked in full.
        indices = tuple(dict.fromkeys((0, matrix.shape[1]//2, matrix.shape[1]-1)))
        for index in indices:
            candidate = ratios * matrix[active, index]
            difference = np.abs(values[index] - candidate)
            scale = np.maximum(abs(values[index]), np.abs(candidate))
            keep = (difference <= 2e-9) | (difference <= 2e-8*scale)
            active, ratios = active[keep], ratios[keep]
            if not len(active):
                return False
        candidates = ratios[:, None] * matrix[active]
        difference = np.abs(values[None, :] - candidates)
        scale = np.maximum(np.abs(values)[None, :], np.abs(candidates))
        return bool(np.any(np.all((difference <= 2e-9) | (difference <= 2e-8*scale), axis=1)))

    def fingerprint(self, expression, process: str) -> np.ndarray:
        return self.contexts[process].evaluate(expression)


def validate_token_roundtrip(original: str, decoded: str) -> dict:
    try:
        equivalent = parse_laurent(original) == parse_laurent(decoded)
        return {"ok": equivalent, "reason": "ok" if equivalent else "token_semantics_mismatch"}
    except (ValueError, ZeroDivisionError) as exc:
        return {"ok": False, "reason": "token_parse_error", "detail": str(exc)}
