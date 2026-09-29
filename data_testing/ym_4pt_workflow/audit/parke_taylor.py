"""Independent symbolic angle and Parke-Taylor comparison (global i stripped)."""

import hashlib
import itertools
from pathlib import Path

import sympy as S

from . import vertices as audit
from ._io import command_line, read_source, require, write_result


def run(source: Path, output_dir: Path) -> dict:
    """Compare all sixteen helicities with spinor formulas at symbolic angle."""
    source, data, rows = read_source(source)
    csv_amplitude = audit.CSVAmplitude(rows[0][1])
    x = S.Symbol('x', positive=True)
    c = (1-x*x)/(1+x*x)
    sint = 2*x/(1+x*x)
    p = audit.points(c, sint)
    bas = audit.basis(c, sint)
    # The bispinor map is p_{a adot} = [[p0+p3,p1-i p2],[p1+i p2,p0-p3]].
    # All outgoing momenta factor p = lambda * lambda_tilde^T; planar point permits
    # lambda_tilde=lambda (including imaginary lambdas for the negative energies).
    lambdas = [S.Matrix([S.I*S.sqrt(2), 0]), S.Matrix([0, S.I*S.sqrt(2)]),
               S.sqrt(2)/S.sqrt(1+x*x)*S.Matrix([1, x]),
               S.sqrt(2)/S.sqrt(1+x*x)*S.Matrix([x, -1])]
    for momentum, lam in zip(p, lambdas):
        bispinor = S.Matrix([[momentum[0]+momentum[3], momentum[1]-S.I*momentum[2]],
                            [momentum[1]+S.I*momentum[2], momentum[0]-momentum[3]]])
        require(all(S.simplify(v) == 0 for v in bispinor-lam*lam.T),
                'Momentum bispinor factorization failed')

    def angle(i, j):
        return S.simplify(S.det(S.Matrix.hstack(lambdas[i], lambdas[j])))

    denominator = S.prod(angle(i, (i+1) % 4) for i in range(4))
    results = []
    for helicities in itertools.product((1, -1), repeat=4):
        eps = [(bas[i][0]+S.I*helicities[i]*bas[i][1])/S.sqrt(2) for i in range(4)]
        require(all(S.simplify(audit.dot(e, e)) == 0 for e in eps),
                'Null circular polarization check failed')
        require(all(S.simplify(audit.dot(e, pp)) == 0 for e, pp in zip(eps, p)),
                'Transverse circular polarization check failed')
        measured = csv_amplitude(p, eps)
        negative = [i for i, h in enumerate(helicities) if h == -1]
        if len(negative) == 2:
            pt = S.simplify(angle(*negative)**4/denominator)
        else:
            pt = S.Integer(0)
        residual = S.factor(measured-pt)
        require(residual == 0, ('Parke-Taylor', helicities, residual))
        results.append({'helicities': ''.join('+' if h == 1 else '-' for h in helicities),
                        'csv': str(S.factor(measured)), 'parke_taylor_i_stripped': str(S.factor(pt)),
                        'residual': str(residual)})
    result = {
        'source': str(source), 'sha256': hashlib.sha256(data).hexdigest(),
        'parameter': 'x = tan(theta/2) > 0',
        'metric': '(+---)', 'global_factor': 'i and coupling g^2 stripped',
        'momenta': [[str(S.factor(v)) for v in z] for z in p],
        'polarizations': 'epsilon_i(h) = (b_i + i*h*y)/sqrt(2), b1=(0,1,0,0), b2=(0,-1,0,0), b3=(0,c,0,-sin(theta)), b4=-b3, y=(0,0,1,0); c=(1-x^2)/(1+x^2)',
        'spinors': [[str(v) for v in z] for z in lambdas],
        'parke_taylor_formula': '<ij>^4 / (<12><23><34><41>) for negative helicity legs i,j; <ij>=det(lambda_i,lambda_j)',
        'checks': results,
    }
    write_result(output_dir, 'parke_taylor_results.json', result)
    return result


if __name__ == '__main__':
    command_line(run, 'Exact symbolic Parke-Taylor four-gluon helicity audit')
