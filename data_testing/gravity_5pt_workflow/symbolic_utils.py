"""Exact algebra and token-compatible printing for the five-point workflow.

``symbolic`` expands field strengths into independent scalar dots, whereas
``parse`` preserves each field-strength contraction as one commuting atom.
Neither operation imposes helicity, on-shell, or momentum-conservation rules.
The repository gravity parser supplies strict syntax and exact rational leaves.
"""
from __future__ import annotations

import re

import sympy as sp
from sympy.printing.str import StrPrinter

from data_gen import gen_data as algebra
from data_gen.data_gen_gravity.core import parse_expression
from data_gen.ordered_gravity_gen import parenthesize_for_semantic_tokenization as safe

__all__ = ["symbolic", "render", "safe", "parse", "serialize"]


def _arithmetic(node, convert):
    if isinstance(node, algebra._Num):
        return sp.Rational(str(node.value))
    if isinstance(node, algebra._UnaryOp):
        value = convert(node.operand)
        return -value if node.op == "-" else value
    if isinstance(node, algebra._BinOp):
        left, right = convert(node.left), convert(node.right)
        operations = {
            "+": lambda: left + right,
            "-": lambda: left - right,
            "*": lambda: left * right,
            "/": lambda: left / right,
            "**": lambda: left ** right,
        }
        return operations[node.op]()
    raise TypeError(f"Unsupported algebra node: {type(node).__name__}")


def symbolic(expression: str) -> sp.Expr:
    """Expand contractions into exact independent symmetric dot products."""
    def convert(node):
        if isinstance(node, tuple) and node[0] == algebra._DOT_TAG:
            parts = sorted((part[1], part[2]) for part in node[1:])
            return sp.Symbol("".join(f"{tag}{index}" for tag, index in parts))
        return _arithmetic(node, convert)

    return convert(algebra._expand_ast(parse_expression(expression)))


def _integer(value: int, digits: bool) -> str:
    # The digit-free form was used in exploratory preprocessing. It is an
    # identity of rational functions; d12 != 0 is a serialization chart.
    dot = "(p_1 · p_2)"
    if value < 0:
        return "-(" + _integer(-value, digits) + ")"
    if digits:
        return str(value)
    if value == 0:
        return f"({dot}-{dot})"
    return "(" + " + ".join([dot] * value) + f")/{dot}"


def render(expression: sp.Expr, digits: bool = True) -> str:
    """Print scalar-dot algebra with repeated products, as in preparation.

    This deliberately retains the original workflow's ordering and spelling.
    Call ``safe`` before feeding the result to the legacy semantic tokenizer.
    """
    expression = sp.sympify(expression)
    if isinstance(expression, sp.Symbol):
        match = re.fullmatch(r"([ep])(\d+)([ep])(\d+)", str(expression))
        if match is None:
            raise ValueError(f"Expected a scalar-dot symbol, got {expression!s}")
        a, i, b, j = match.groups()
        return f"({a}_{i} · {b}_{j})"
    if isinstance(expression, sp.Rational):
        if expression.q == 1:
            return "(" + _integer(int(expression.p), digits) + ")"
        return ("(" + _integer(int(expression.p), digits) + ")/("
                + _integer(int(expression.q), digits) + ")")
    if isinstance(expression, sp.Add):
        parts = []
        for term in expression.as_ordered_terms():
            negative = term.could_extract_minus_sign()
            parts.append(("-" if negative else "+") + "("
                         + render(-term if negative else term, digits) + ")")
        return "(" + "".join(parts).lstrip("+") + ")"
    if isinstance(expression, sp.Mul):
        if expression.could_extract_minus_sign():
            return "-(" + render(-expression, digits) + ")"
        numerator, denominator = sp.fraction(expression)
        if denominator != 1:
            return "(" + render(numerator, digits) + ")/(" + render(denominator, digits) + ")"
        return "(" + "*".join(render(term, digits) for term in expression.args) + ")"
    if isinstance(expression, sp.Pow):
        base, power = expression.args
        if not power.is_Integer:
            raise ValueError("Scalar rendering requires an integer power")
        if power == 0:
            return "(" + _integer(1, digits) + ")"
        value = "(" + "*".join("(" + render(base, digits) + ")"
                                for _ in range(abs(int(power)))) + ")"
        return value if power > 0 else "(" + _integer(1, digits) + ")/(" + value + ")"
    raise TypeError(f"Cannot render scalar expression {expression!r}")


def parse(expression: str) -> sp.Expr:
    """Keep ordered F chains/trace factors atomic while simplifying arithmetic.

    Scalar dots have canonical symmetric names (``e4p1``, ``p1p4``). Ordered
    chains use names such as ``p1F4p5`` and traces use ``TrF4F5``. These atom
    names can be round-tripped without global spelling state.
    """
    def atomize(node):
        if isinstance(node, algebra._DotChain):
            trace = node.parts[-1] is algebra._DotChain._TR
            vectors = node.parts[:-1] if trace else node.parts
            parts = [(part.tag, part.idx) for part in vectors]
            if not trace and len(parts) == 2:
                parts.sort()
            name = ("Tr" if trace else "") + "".join(f"{tag}{index}" for tag, index in parts)
            return sp.Symbol(name)
        return _arithmetic(node, atomize)

    return atomize(parse_expression(expression))


def _symbol_spelling(symbol: sp.Symbol) -> str:
    name = str(symbol)
    trace = name.startswith("Tr")
    chain = name[2:] if trace else name
    if not re.fullmatch(r"(?:[epF]\d+){2,}", chain):
        raise ValueError(f"Unknown contraction atom {name!r}")
    parts = re.findall(r"([epF])(\d+)", chain)
    if trace and any(tag != "F" for tag, _ in parts):
        raise ValueError(f"Trace atom must contain only F factors: {name!r}")
    content = "·".join(f"{tag}_{index}" for tag, index in parts)
    return f"Tr({content})" if trace else f"({content})"


class _ContractionPrinter(StrPrinter):
    def _print_Symbol(self, expression):
        return _symbol_spelling(expression)


class _NoPowersPrinter(_ContractionPrinter):
    def _print_Pow(self, expression, rational=False):
        if expression.exp.is_Integer and expression.exp > 1:
            return "(" + "*".join("(" + self._print(expression.base) + ")"
                                    for _ in range(int(expression.exp))) + ")"
        return super()._print_Pow(expression, rational=rational)


def serialize(expression: sp.Expr, powers: bool = True) -> str:
    """Print contraction atoms with ``^`` powers or repeated multiplication.

    As with the original cleanup script, callers verify the tokenizer's exact
    semantic round trip before accepting a serialized final expression.
    """
    printer = _ContractionPrinter() if powers else _NoPowersPrinter()
    return printer.doprint(sp.sympify(expression)).replace("**", "^")
