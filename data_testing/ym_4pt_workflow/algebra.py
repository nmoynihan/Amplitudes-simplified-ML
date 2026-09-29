"""Independent exact four-gluon algebra used by preparation and verification.

Conventions: F_i = p_i tensor e_i - e_i tensor p_i, s = 2 p_1.p_2,
t = 2 p_1.p_4. The quotient imposes masslessness, momentum conservation,
and e_i.p_i = 0, without four-dimensional Gram identities. Rational
equalities are understood away from denominator poles. No model is imported.
"""
from __future__ import annotations

import ast
import csv
import gzip
import itertools
import re
import sys
from pathlib import Path

import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = REPO_ROOT / "data/data_ym/gluon4feyn1234.csv.gz"
DEFAULT_REFERENCE = REPO_ROOT / "data/data_ym/gluon4feyn.csv.gz"
DATA_GEN = REPO_ROOT / "data_gen"
LABEL = re.compile(r"\b([peF])_([1-4])\b")
ATOM = re.compile(r"([pe])([1-4])([pe])([1-4])")
VEC = re.compile(r"([peF])_([1-4])\b")
TRACE = re.compile(r"Tr\s*\(\s*F_[1-4](?:\s*[·.]\s*F_[1-4])+\s*\)")
CHAIN = re.compile(r"[pe]_[1-4](?:\s*[·.]\s*[peF]_[1-4]\b)+")
s, t = sp.symbols("s t", nonzero=True)
PP = {(1, 2): s / 2, (1, 3): -(s + t) / 2, (1, 4): t / 2,
      (2, 3): t / 2, (2, 4): -(s + t) / 2, (3, 4): s / 2}


def use_data_gen() -> None:
    """Enable the repository's historical data_gen_ym package layout."""
    if str(DATA_GEN) not in sys.path:
        sys.path.insert(0, str(DATA_GEN))


def tokenizer():
    use_data_gen()
    from Tokenizer import ScatteringAmplitudeTokenizer
    return ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)


def read_expression(path: Path) -> str:
    """Read exactly one [id, expression] CSV row, optionally with a header."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if rows and rows[0][0].strip().lower() == "id":
        rows = rows[1:]
    if len(rows) != 1 or len(rows[0]) != 2 or not rows[0][1].strip():
        raise ValueError(f"Expected one [id, expression] CSV row in {path}")
    return rows[0][1].strip()


def rotate(expression: str, shift: int = 1) -> str:
    return LABEL.sub(lambda m: f"{m[1]}_{(int(m[2]) - 1 + shift) % 4 + 1}", expression)


def atom(a: str, i: int, b: str, j: int) -> sp.Symbol:
    if a not in ("p", "e") or b not in ("p", "e"):
        raise ValueError("Scalar products require vector endpoints")
    if int(i) not in range(1, 5) or int(j) not in range(1, 5):
        raise ValueError("This workflow supports four external legs")
    left, right = sorted(((a, int(i)), (b, int(j))))
    return sp.Symbol(f"{left[0]}{left[1]}{right[0]}{right[1]}")


def ep(i: int, j: int):
    if i == j:
        return sp.Integer(0)
    if j == 4:
        return -sum(ep(i, k) for k in (1, 2, 3))
    if i == 4 and j == 3:
        return -ep(4, 1) - ep(4, 2)
    return atom("e", i, "p", j)


def dot(a: str, i: int, b: str, j: int):
    """An on-shell dot product, useful when constructing a tensor basis."""
    if a == b == "p":
        return sp.Integer(0) if i == j else PP[tuple(sorted((i, j)))]
    if a == b == "e":
        return atom(a, i, b, j)
    return ep(i, j) if a == "e" else ep(j, i)


def on_shell(expression):
    substitutions = {}
    for symbol in expression.free_symbols:
        match = ATOM.fullmatch(str(symbol))
        if match:
            a, i, b, j = match.groups()
            substitutions[symbol] = dot(a, int(i), b, int(j))
    return expression.xreplace(substitutions)


def expand_chain(vectors, trace: bool = False):
    """Expand tensor contractions independently of the training generator."""
    if len(vectors) < 2:
        raise ValueError("A contraction needs at least two entries")
    if not trace and any(v[0] not in ("p", "e") for v in (vectors[0], vectors[-1])):
        raise ValueError("An open chain needs vector endpoints")
    fields = vectors if trace else vectors[1:-1]
    if not trace and len(vectors) == 2:
        return atom(*vectors[0], *vectors[1])
    if any(tag != "F" for tag, _ in fields):
        raise ValueError(f"Invalid field-strength chain: {vectors}")
    terms = []
    for bits in itertools.product((0, 1), repeat=len(fields)):
        pairs = [(("p", i), ("e", i)) if bit == 0 else (("e", i), ("p", i))
                 for (_, i), bit in zip(fields, bits)]
        factors = [atom(*pairs[k][1], *pairs[k + 1][0]) for k in range(len(pairs) - 1)]
        factors += ([atom(*pairs[-1][1], *pairs[0][0])] if trace else
                    [atom(*vectors[0], *pairs[0][0]), atom(*pairs[-1][1], *vectors[-1])])
        terms.append((-1) ** sum(bits) * sp.prod(factors))
    return sp.Add(*terms)


def parse(expression: str):
    """Strict arithmetic AST parser with independent F-chain expansion.

    Only integers, contractions, +, -, *, / and integer powers are accepted;
    input is never evaluated as Python or unrestricted SymPy code.
    """
    # User identifiers must not alias the placeholders introduced below.
    remainder = CHAIN.sub("0", TRACE.sub("0", expression))
    if re.search(r"[A-Za-z_]", remainder):
        raise ValueError("Unsupported identifier outside a tensor contraction")
    contractions = {}

    def substitute(match):
        key = f"BLOCK{len(contractions)}"
        vectors = [(tag, int(i)) for tag, i in VEC.findall(match.group())]
        contractions[key] = expand_chain(vectors, match.group().startswith("Tr"))
        return key

    text = CHAIN.sub(substitute, TRACE.sub(substitute, expression)).strip().replace("^", "**")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in contractions:
            return contractions[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Integer:
                return left ** right
        raise ValueError(f"Unsupported expression syntax: {ast.dump(node)}")

    result = visit(ast.parse(text, mode="eval"))
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Nonfinite scalar expression")
    return result


def scalar(expression: str):
    if "F_" in expression or "Tr" in expression:
        raise ValueError("Expected an expression containing p/e dots only")
    return parse(expression)


def symbolic(expression: str):
    return on_shell(parse(expression))


def require_zero(expression, description: str) -> None:
    difference = sp.cancel(expression)
    if difference != 0:
        raise ValueError(f"{description}: exact difference is {difference}")


def ward(expression, leg: int):
    substitutions = {}
    for symbol in expression.free_symbols:
        match = ATOM.fullmatch(str(symbol))
        if match:
            a, i, b, j = match.groups()
            i, j = int(i), int(j)
            substitutions[symbol] = atom("p" if a == "e" and i == leg else a, i,
                                         "p" if b == "e" and j == leg else b, j)
    return on_shell(expression.xreplace(substitutions))


def render(expression) -> str:
    """Render scalar SymPy algebra using the historical dot notation."""
    if isinstance(expression, sp.Symbol):
        match = ATOM.fullmatch(str(expression))
        if not match:
            raise ValueError(f"Cannot render non-contraction symbol {expression}")
        a, i, b, j = match.groups()
        return f"({a}_{i} · {b}_{j})"
    if isinstance(expression, sp.Integer):
        return str(expression)
    if isinstance(expression, sp.Rational):
        return f"({expression.p}/{expression.q})"
    if isinstance(expression, sp.Add):
        pieces = []
        for term in expression.as_ordered_terms():
            sign = "-" if term.could_extract_minus_sign() else "+"
            body = render(-term if sign == "-" else term)
            pieces.append((sign if pieces or sign == "-" else "") + body)
        return "(" + " ".join(pieces) + ")"
    if isinstance(expression, sp.Mul):
        numerator, denominator = expression.as_numer_denom()
        if denominator != 1:
            return "(" + render(numerator) + "/" + render(denominator) + ")"
        return "(" + "*".join(render(factor) for factor in expression.args) + ")"
    if isinstance(expression, sp.Pow):
        if expression.exp < 0:
            return "(1/" + render(expression.base ** (-expression.exp)) + ")"
        return "(" + render(expression.base) + "^" + render(expression.exp) + ")"
    raise ValueError(f"Unsupported expression: {expression}")
