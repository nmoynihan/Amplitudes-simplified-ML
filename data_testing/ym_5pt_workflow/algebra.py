"""Independent exact five-point algebra, with F_i = p_i tensor e_i - e_i tensor p_i.

No training-time expansion routines or four-dimensional Gram identities are
used. The quotient imposes masslessness, momentum conservation and e_i.p_i=0;
rational identities hold away from denominator poles.
"""
from __future__ import annotations

import ast
import itertools
import re

import sympy as sp

LABELS = {}

def atom(t, i, u, j):
    pair = tuple(sorted(((t, int(i)), (u, int(j)))))
    sym = sp.Symbol(''.join(f'{a}{b}' for a, b in pair))
    LABELS[sym] = pair
    return sym


a, b, c, d, e = sp.symbols('a b c d e')
PP = {(1,2):a, (2,3):b, (3,4):c, (4,5):d, (1,5):e,
      (1,3):d-a-b, (2,4):e-b-c, (3,5):a-c-d,
      (1,4):b-d-e, (2,5):c-e-a}


def on_shell(expr):
    rules = {}
    for s in expr.free_symbols:
        if s not in LABELS: continue
        (t,i),(u,j) = LABELS[s]
        if t == u == 'p':
            rules[s] = sp.Integer(0) if i == j else PP[tuple(sorted((i,j)))]
        elif t == 'e' and u == 'p':
            others = [k for k in range(1,6) if k != i]
            if i == j: rules[s] = sp.Integer(0)
            elif j == others[-1]:
                rules[s] = -sum(atom('e',i,'p',k) for k in others[:-1])
    return sp.cancel(expr.xreplace(rules))


def cyclic(expr, shift):
    return expr.xreplace({s:atom(t,(i-1+shift)%5+1,u,(j-1+shift)%5+1)
                         for s,((t,i),(u,j)) in list(LABELS.items()) if s in expr.free_symbols})


def ward(expr, leg):
    return on_shell(expr.xreplace({s:atom('p' if t=='e' and i==leg else t,i,
                                         'p' if u=='e' and j==leg else u,j)
                                  for s,((t,i),(u,j)) in list(LABELS.items()) if s in expr.free_symbols}))



VEC = re.compile(r'([peF])_([1-5])')
TRACE = re.compile(r'Tr\s*\(\s*F_[1-5](?:\s*[·.]\s*F_[1-5])+\s*\)')
CHAIN = re.compile(r'[pe]_[1-5](?:\s*[·.]\s*[peF]_[1-5])+')

def dot(x, y):
    if x[0] not in ('p', 'e') or y[0] not in ('p', 'e'):
        raise ValueError('A scalar dot product needs two vectors')
    return atom(x[0], x[1], y[0], y[1])

def expand_chain(vectors, trace=False):
    if len(vectors) < 2:
        raise ValueError('A contraction needs at least two entries')
    if not trace and (vectors[0][0] not in ('p', 'e') or vectors[-1][0] not in ('p', 'e')):
        raise ValueError('Open field-strength chains need vector endpoints')
    fs = vectors if trace else vectors[1:-1]
    if not trace and len(vectors) == 2:
        return dot(*vectors)
    if any(t != 'F' for t, _ in fs):
        raise ValueError(f'Invalid chain: {vectors}')
    terms = []
    for bits in itertools.product((0, 1), repeat=len(fs)):
        pairs = [(('p', i), ('e', i)) if bit == 0 else (('e', i), ('p', i))
                 for (_, i), bit in zip(fs, bits)]
        factors = [dot(pairs[i][1], pairs[i+1][0]) for i in range(len(pairs)-1)]
        factors += ([dot(pairs[-1][1], pairs[0][0])] if trace else
                    [dot(vectors[0], pairs[0][0]), dot(pairs[-1][1], vectors[-1])])
        terms.append((-1)**sum(bits)*sp.prod(factors))
    return sp.Add(*terms)

def parse(expression):
    """Strict scalar arithmetic parser, with independent F tensor expansion."""
    atoms = {}
    def contraction(m):
        key = f'BLOCK{len(atoms)}'
        vectors = [(t, int(i)) for t, i in VEC.findall(m.group())]
        atoms[key] = expand_chain(vectors, m.group().startswith('Tr'))
        return key
    text = TRACE.sub(contraction, expression)
    text = CHAIN.sub(contraction, text).strip().replace('^', '**')
    def visit(n):
        if isinstance(n, ast.Expression): return visit(n.body)
        if isinstance(n, ast.Name) and n.id in atoms: return atoms[n.id]
        if isinstance(n, ast.Constant) and type(n.value) is int: return sp.Integer(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -visit(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd): return visit(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return a+b
            if isinstance(n.op, ast.Sub): return a-b
            if isinstance(n.op, ast.Mult): return a*b
            if isinstance(n.op, ast.Div): return a/b
            if isinstance(n.op, ast.Pow) and b.is_Integer: return a**b
        raise ValueError(ast.dump(n))
    expression = visit(ast.parse(text, mode='eval'))
    if expression.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError('Nonfinite scalar expression')
    return expression
