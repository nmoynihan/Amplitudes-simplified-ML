"""Independent exact 4D Yang-Mills vertex audit, metric (+---), all outgoing.

This uses explicit four-vectors, its own contraction parser, and independently
implemented vertices. It does not import the scalar symbolic audit or the model.
"""

import ast
import hashlib
import itertools
from pathlib import Path
import re

import sympy as S

from ._io import command_line, read_source, require, write_result


class CSVAmplitude:
    """Parse one amplitude without evaluating arbitrary Python input."""

    def __init__(self, expression):
        self.labels = {}
        pattern = r'\(([ep])_([1-4])\s*·\s*([ep])_([1-4])\)'
        if re.search(r'[A-Za-z_]', re.sub(pattern, '0', expression)):
            raise ValueError('Unsupported identifier outside a scalar contraction')

        def name(match):
            key = tuple(match.groups())
            label = ''.join(key)
            self.labels[label] = key
            return label

        normalized = re.sub(pattern, name, expression).strip().replace('^', '**')
        tree = ast.parse(normalized, mode='eval')

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Name) and node.id in self.labels:
                return S.Symbol(node.id)
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return S.Integer(node.value)
            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    return -evaluate(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return evaluate(node.operand)
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
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
            raise ValueError(f'Unsupported amplitude syntax: {ast.dump(node)}')

        self.expr = evaluate(tree)
        self.symbols = sorted(self.expr.free_symbols, key=str)

    def __call__(self, p, e):
        substitutions = {}
        for symbol in self.symbols:
            a, i, b, j = self.labels[str(symbol)]
            substitutions[symbol] = dot({'e': e, 'p': p}[a][int(i)-1],
                                        {'e': e, 'p': p}[b][int(j)-1])
        return S.simplify(self.expr.xreplace(substitutions))


def dot(a, b):
    return a[0]*b[0] - sum(a[i]*b[i] for i in (1, 2, 3))


def points(c, s):
    return list(map(S.Matrix, [[-1, 0, 0, -1], [-1, 0, 0, 1],
                               [1, s, 0, c], [1, -s, 0, -c]]))


def basis(c, s):
    return [(S.Matrix([0, 1, 0, 0]), S.Matrix([0, 0, 1, 0])),
            (S.Matrix([0, -1, 0, 0]), S.Matrix([0, 0, 1, 0])),
            (S.Matrix([0, c, 0, -s]), S.Matrix([0, 0, 1, 0])),
            (S.Matrix([0, -c, 0, s]), S.Matrix([0, 0, 1, 0]))]


def current(p, e, a, b):
    # V^{mu nu rho}(pa,pb,-pa-pb) ea_mu eb_nu,
    # valid for transverse external states; no couplings or i factors.
    return (dot(e[a], e[b])*(p[a]-p[b]) + 2*dot(e[a], p[b])*e[b]
            - 2*dot(e[b], p[a])*e[a])


def ym(p, e):
    ss = dot(p[0]+p[1], p[0]+p[1])
    tt = dot(p[1]+p[2], p[1]+p[2])
    ns = dot(current(p, e, 0, 1), current(p, e, 2, 3))
    nt = dot(current(p, e, 1, 2), current(p, e, 3, 0))
    contact = (2*dot(e[0], e[2])*dot(e[1], e[3])
               - dot(e[0], e[1])*dot(e[2], e[3])
               - dot(e[0], e[3])*dot(e[1], e[2]))
    return S.simplify(ns/ss + nt/tt + contact)


def run(source: Path, output_dir: Path) -> dict:
    """Evaluate all linear/circular polarizations at three exact kinematic points."""
    source, data, rows = read_source(source)
    csv_amplitude = CSVAmplitude(rows[0][1])
    angles = [(S.Rational(3, 5), S.Rational(4, 5)),
              (S.Rational(5, 13), S.Rational(12, 13)),
              (S.Rational(-7, 25), S.Rational(24, 25))]
    result = {
        'source': str(source), 'sha256': hashlib.sha256(data).hexdigest(),
        'row_count': len(rows), 'terms': len(S.Add.make_args(csv_amplitude.expr)),
        'conventions': 'metric (+---), all outgoing, g and global i omitted; unnormalized ordered vertices',
        'points': [],
    }
    ratios = set()
    for c, s in angles:
        p = points(c, s)
        bas = basis(c, s)
        require(sum(p, S.zeros(4, 1)) == S.zeros(4, 1), 'Momentum conservation failed')
        require(all(dot(x, x) == 0 for x in p), 'Massless momenta check failed')
        require(all(dot(p[i], b) == 0 for i, pair in enumerate(bas) for b in pair),
                'Transverse basis check failed')
        item = {'cos_theta': str(c), 'sin_theta': str(s),
                'momenta': [[str(x) for x in z] for z in p],
                'linear': [], 'helicity': [], 'ward': []}
        for bits in itertools.product((0, 1), repeat=4):
            e = [bas[i][bits[i]] for i in range(4)]
            a, b = csv_amplitude(p, e), ym(p, e)
            require(S.simplify(a-b/2) == 0, ('linear', c, bits, a, b))
            ratio = S.simplify(a/b) if b else None
            if ratio is not None:
                ratios.add(str(ratio))
            item['linear'].append({'basis': ''.join(map(str, bits)),
                                   'csv': str(a), 'ym': str(b), 'ratio': str(ratio)})
        for helicities in itertools.product((1, -1), repeat=4):
            e = [(bas[i][0]+S.I*helicities[i]*bas[i][1])/S.sqrt(2) for i in range(4)]
            a, b = csv_amplitude(p, e), ym(p, e)
            require(S.simplify(a-b/2) == 0, ('helicity', c, helicities, a, b))
            ratio = S.simplify(a/b) if b else None
            if ratio is not None:
                ratios.add(str(ratio))
            item['helicity'].append({'helicities': ''.join('+' if h == 1 else '-' for h in helicities),
                                     'csv': str(a), 'ym': str(b), 'ratio': str(ratio)})
        generic = [(i+1)*pair[0] + (i+2)*pair[1] + S.Rational(i+1, 7)*p[i]
                   for i, pair in enumerate(bas)]
        a, b = csv_amplitude(p, generic), ym(p, generic)
        require(S.simplify(a-b/2) == 0, ('generic', c, a, b))
        item['generic'] = {'csv': str(a), 'ym': str(b),
                           'ratio': str(S.simplify(a/b)) if b else None}
        for i in range(4):
            e = generic.copy()
            e[i] = p[i]
            a, b = csv_amplitude(p, e), ym(p, e)
            require(a == b == 0, ('ward', c, i, a, b))
            item['ward'].append({'leg': i+1, 'csv': str(a), 'ym': str(b)})
        result['points'].append(item)
    result['all_nonzero_ym_ratios'] = sorted(ratios)
    write_result(output_dir, 'vertices_results.json', result)
    return result


if __name__ == '__main__':
    command_line(run, 'Independent exact four-vector Yang-Mills audit')
