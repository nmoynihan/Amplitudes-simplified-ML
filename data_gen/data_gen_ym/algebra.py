"""algebra — extracted from gen_data.py (scaffold, verbatim)."""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from notation import *

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


def simplify_to_lowest_terms(expr: str) -> str:
    """Cancel common numerator/denominator factors in each top-level term."""
    trace_placeholders: dict[str, str] = {}
    dot_placeholders: dict[str, str] = {}
    symbol_names: set[str] = set()

    def _replace_trace(match: re.Match) -> str:
        name = f"TR_{len(trace_placeholders)}"
        trace_placeholders[name] = match.group(0)
        symbol_names.add(name)
        return name

    def _dot_symbol_name(left: str, right: str) -> str:
        ltag, lidx = left.split("_", 1)
        rtag, ridx = right.split("_", 1)
        if ltag == rtag == "p":
            a, b = sorted((int(lidx), int(ridx)))
            return f"pp{a}_{b}"
        if ltag == rtag == "e":
            a, b = sorted((int(lidx), int(ridx)))
            return f"ee{a}_{b}"
        if ltag == "e" and rtag == "p":
            return f"ep{int(lidx)}_{int(ridx)}"
        return f"ep{int(ridx)}_{int(lidx)}"

    def _dot_text(left: str, right: str) -> str:
        ltag, lidx = left.split("_", 1)
        rtag, ridx = right.split("_", 1)
        if ltag == rtag == "p":
            a, b = sorted((int(lidx), int(ridx)))
            return f"p_{a} {DOT} p_{b}"
        if ltag == rtag == "e":
            a, b = sorted((int(lidx), int(ridx)))
            return f"e_{a} {DOT} e_{b}"
        if ltag == "e" and rtag == "p":
            return f"e_{int(lidx)} {DOT} p_{int(ridx)}"
        return f"e_{int(ridx)} {DOT} p_{int(lidx)}"

    def _replace_dot(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        name = _dot_symbol_name(left, right)
        dot_placeholders[name] = _dot_text(left, right)
        symbol_names.add(name)
        return name

    encoded = _RE_TrN.sub(_replace_trace, expr)
    encoded = _RE_DOT.sub(_replace_dot, encoded)
    encoded = encoded.replace("^", "**")

    locals_map = {name: sp.Symbol(name) for name in symbol_names}
    sym_expr = sp.sympify(encoded, locals=locals_map)
    expanded = sp.expand(sym_expr)
    pieces = [sp.cancel(term) for term in sp.Add.make_args(expanded)]
    simplified = sp.Add(*pieces, evaluate=False) if pieces else sp.Integer(0)
    rendered = sp.sstr(simplified)

    substitutions = {
        **{name: text for name, text in trace_placeholders.items()},
        **{name: f"({text})" for name, text in dot_placeholders.items()},
    }
    for name in sorted(substitutions, key=len, reverse=True):
        rendered = re.sub(rf"\b{re.escape(name)}\b", substitutions[name], rendered)
    return rendered.replace("**", "^")


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

__all__ = [
    '_canon_pp',
    '_canon_TrN',
    '_factor_sort_key',
    'canonicalise_gi_product',
    'canonicalise_denominator',
    '_RatTerm',
    '_is_zero_coeff',
    '_format_number',
    '_format_factor',
    '_node_to_factor',
    '_negate_terms',
    '_mul_terms',
    '_divide_terms',
    '_rational_terms',
    '_format_rational_term',
    'full_expand_expression',
    'simplify_to_lowest_terms',
    '_Num',
    '_Vec',
    '_DotChain',
    '_BinOp',
    '_UnaryOp',
    '_DOT_TAG',
    '_VEC_TAG',
    '_Parser',
    '_tokenize',
    '_mk_dot',
    '_vn',
    '_ast_add',
    '_ast_sub',
    '_ast_mul',
    '_expand_dotchain',
    '_expand_ast',
    '_prec',
    '_vec_name',
    '_ast_to_infix',
]
