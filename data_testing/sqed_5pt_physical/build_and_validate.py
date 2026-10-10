#!/usr/bin/env python3
"""Build one physical 2-scalar/3-photon SQED amplitude and verify it.

All momenta outgoing, metric (+---), scalar legs 1,5, photon legs 2,3,4.
The overall i*q**3 is stripped. Along the scalar line use the cubic rule
i*q*(2*P+k).e and the seagull rule 2*i*q**2*e_i.e_j. Thus the stripped
three-cubic diagrams have positive vertex products, and diagrams containing
one seagull have negative vertex products. Scalar propagator: i/(P**2-m**2).
"""
from __future__ import annotations

import csv
import gzip
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_gen import gen_data as gd
from data_gen.kinematics import generate_kinematics, mdot
from data_gen.Tokenizer import ScatteringAmplitudeTokenizer

OUT = Path(__file__).resolve().parent
PHOTONS = (2, 3, 4)
MODEL = Path('/Users/kymani/Documents/SQED_KYMANI/SQED/5PT_Model/best_model.pt')
TRAIN = Path('/Users/kymani/Documents/SQED_KYMANI/sqed_5pt_500k/sqed_5pt_oneshot_train_499800.csv.gz')


def dot(a, b):
    return f'({a} · {b})'


def expressions():
    ep = lambda i, j: dot(f'e_{i}', f'p_{j}')
    pp = lambda i, j: dot(f'p_{i}', f'p_{j}')
    terms = []
    # Six orderings along the scalar line, each expanded into two monomials.
    # D_i=2*p1.pi, D_ij=2*p5.pk, and (p1+pi+pj).ek=-p5.ek.
    for i, j, k in itertools.permutations(PHOTONS):
        for middle in (1, i):
            terms.append(f'-2*{ep(i,1)}*{ep(j,middle)}*{ep(k,5)}/({pp(1,i)}*{pp(k,5)})')
    # Three photon pairs, with the seagull before or after the cubic vertex.
    for k in PHOTONS:
        i, j = [x for x in PHOTONS if x != k]
        ee = dot(f'e_{i}', f'e_{j}')
        terms.append(f'2*{ee}*{ep(k,5)}/{pp(k,5)}')
        terms.append(f'-2*{ee}*{ep(k,1)}/{pp(1,k)}')
    scrambled = ' + '.join(terms).replace('+ -', '- ')
    # Gauge-project every polarisation to E_i=e_i-p_i*(p1.e_i)/(p1.p_i).
    # Only seagull-first diagrams survive; E_i.E_j=-p1.F_i.F_j.p1/(a_i*a_j).
    reference_terms = []
    for k in PHOTONS:
        i, j = [x for x in PHOTONS if x != k]
        reference_terms.append(
            f'-2*(p_1 · F_{i} · F_{j} · p_1)*(p_1 · F_{k} · p_5)'
            f'/({pp(1,2)}*{pp(1,3)}*{pp(1,4)}*{pp(k,5)})'
        )
    return {'simple': ' + '.join(reference_terms).replace('+ -', '- '),
            'scrambled': scrambled}


def symbolic(expression):
    """Exact rational expression after expanding F tensors, without kinematics."""
    def visit(node):
        if isinstance(node, gd._Num):
            return sp.Rational(str(node.value))
        if isinstance(node, gd._UnaryOp):
            value = visit(node.operand)
            return -value if node.op == '-' else value
        if isinstance(node, gd._BinOp):
            a, b = visit(node.left), visit(node.right)
            return {'+': lambda: a+b, '-': lambda: a-b, '*': lambda: a*b,
                    '/': lambda: a/b, '**': lambda: a**b}[node.op]()
        if isinstance(node, tuple) and node[0] == gd._DOT_TAG:
            a, b = sorted([f'{v[1]}{v[2]}' for v in node[1:]])
            return sp.Symbol(f'{a}_{b}')
        raise TypeError(type(node))
    return visit(gd._expand_ast(gd._Parser(gd._tokenize(expression)).parse()))


def on_shell(expr):
    """Eliminate p5 and impose transverse photons and equal scalar masses."""
    s = lambda a, b: sp.Symbol('_'.join(sorted((a,b))))
    rules = {s(f'e{i}', f'p{i}'): 0 for i in PHOTONS}
    for i in PHOTONS:
        rules[s(f'e{i}', 'p5')] = -sum(s(f'e{i}',f'p{j}') for j in (1,2,3,4) if j != i)
        rules[s(f'p{i}', 'p5')] = -sum(s(f'p{i}',f'p{j}') for j in (1,2,3,4) if j != i)
    result = expr.subs(rules, simultaneous=True)
    # p5^2=p1^2 and p2^2=p3^2=p4^2=0.
    result = result.subs(s('p1','p4'), -sum(s(f'p{i}',f'p{j}') for i,j in
                                         ((1,2),(1,3),(2,3),(2,4),(3,4))))
    # Combine the rational terms before expanding only their common numerator.
    # cancel() alone can expand enormous products of duplicated denominators.
    return sp.expand(sp.together(result).as_numer_denom()[0])


def direct_diagrams(p, e, mass):
    """Independent four-vector Feynman calculation, before endpoint reduction."""
    terms = []
    for i,j,k in itertools.permutations(range(3)):
        pi, pj, pk = p[i+1], p[j+1], p[k+1]
        q1, q2 = p[0]+pi, p[0]+pi+pj
        terms.append(mdot(2*p[0]+pi,e[i])*mdot(2*q1+pj,e[j])*mdot(2*q2+pk,e[k]) /
                     ((mdot(q1,q1)-mass**2)*(mdot(q2,q2)-mass**2)))
    for k in range(3):
        i,j = [x for x in range(3) if x != k]
        q_pair = p[0]+p[i+1]+p[j+1]
        q_single = p[0]+p[k+1]
        terms.append(-2*mdot(e[i],e[j])*mdot(2*q_pair+p[k+1],e[k]) /
                     (mdot(q_pair,q_pair)-mass**2))
        terms.append(-2*mdot(e[i],e[j])*mdot(2*p[0]+p[k+1],e[k]) /
                     (mdot(q_single,q_single)-mass**2))
    return float(sum(terms))


def main():
    row = expressions()
    sym = {key: symbolic(value) for key,value in row.items()}
    assert on_shell(sym['simple']-sym['scrambled']) == 0
    print('Exact reference equivalence passed.', flush=True)
    for leg in PHOTONS:
        replacements = {}
        for symbol in sym['scrambled'].free_symbols:
            parts = symbol.name.split('_')
            if f'e{leg}' in parts:
                replacements[symbol] = sp.Symbol('_'.join(sorted(
                    f'p{leg}' if part == f'e{leg}' else part for part in parts)))
        ward = sym['scrambled'].subs(replacements, simultaneous=True)
        ward = ward.subs({sp.Symbol(f'e{leg}_p{leg}'): 0 for leg in PHOTONS})
        # No squared momenta occur after the endpoint reduction.
        assert on_shell(ward) == 0, leg
    print('All three exact Ward identities passed.', flush=True)

    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=5000)
    tokens = {key: tokenizer.encode_infix(value) for key,value in row.items()}
    for key, ids in tokens.items():
        assert 1 not in ids and max(ids) < 58 and len(ids)+2 <= 5000
        assert on_shell(symbolic(tokenizer.decode_infix(ids))-sym[key]) == 0

    max_error = {'reference_vs_input': 0., 'independent_diagrams': 0., 'ward': 0.,
                 'gauge_shift': 0., 'photon_permutations': 0.}
    count = 0
    def check(label, a, b):
        error = abs(a-b)/max(1.,abs(a),abs(b))
        max_error[label] = max(max_error[label], error)
        assert np.isfinite(a) and np.isfinite(b) and error < 1e-9, (label,a,b,error)

    for mass in (0.7, 2., 5.):
        for mode in ('coulomb','covariant'):
            for seed in range(20):
                p,e = generate_kinematics(5, M=mass, pol_mode=mode, seed=7000+seed)
                assert np.max(abs(p.sum(axis=0))) < 1e-10
                values = {key: gd.eval_infix_numeric(value,p,e,strict=True) for key,value in row.items()}
                check('reference_vs_input', values['simple'], values['scrambled'])
                check('independent_diagrams', values['scrambled'], direct_diagrams(p,e,mass))
                for k in range(3):
                    ward_e = e.copy(); ward_e[k] = p[k+1]
                    check('ward',gd.eval_infix_numeric(row['scrambled'],p,ward_e,strict=True),0.)
                shifted = e + np.array([0.13,-0.27,0.41])[:,None]*p[1:4]
                check('gauge_shift',gd.eval_infix_numeric(row['scrambled'],p,shifted,strict=True),values['scrambled'])
                for perm in itertools.permutations(range(3)):
                    pp = p[[0,*[x+1 for x in perm],4]]
                    ee = e[list(perm)]
                    check('photon_permutations',gd.eval_infix_numeric(row['scrambled'],pp,ee,strict=True),values['scrambled'])
                count += 1

    # Confirm this tokenizer reproduces a supplied 5PT training token row.
    with gzip.open(TRAIN,'rt') as f:
        train_row = next(csv.DictReader(f))
    train_tok_path = TRAIN.with_name(TRAIN.name.replace('.csv.gz','_tok.csv.gz'))
    with gzip.open(train_tok_path,'rt') as f:
        train_tokens = next(csv.DictReader(f))
    for key in row:
        assert tokenizer.encode_infix(train_row[key]) == json.loads(train_tokens[key])

    for filename, values in [('sqed_5pt_physical.csv',row),
                             ('sqed_5pt_physical_tok.csv',{k:json.dumps(v) for k,v in tokens.items()})]:
        with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['simple','scrambled'])
            writer.writeheader(); writer.writerow(values)
    report = {
        'process': 'phi(p1), gamma(p2), gamma(p3), gamma(p4), phi_conjugate(p5)',
        'conventions': 'All outgoing, metric (+---), p1^2=p5^2=m^2, photons massless/transverse; i*q^3 stripped.',
        'construction': 'Six scalar-line orderings plus six seagull diagrams; 18 expanded input terms; 3 F-tensor reference terms.',
        'reference_derivation': 'Gauge project all e_i to e_i-p_i*(p1.e_i)/(p1.p_i); only seagull-first diagrams survive.',
        'physical_poles': 'p1.pi and p5.pi for i=2,3,4; D_i=2*p1.pi and D_ij=2*p5.pk.',
        'symbolic_reference_equivalence': True, 'symbolic_ward_identities': [2,3,4],
        'numerical_phase_space_points': count, 'max_normalized_errors': max_error,
        'token_counts_without_bos_eos': {k:len(v) for k,v in tokens.items()},
        'tokenizer_max_particles': 8, 'training_token_row_exactly_reproduced': True,
        'model_path': str(MODEL), 'model_vocabulary_size': 58, 'model_sequence_capacity': 5000,
        'five_point_training_data': str(TRAIN),
        'user_supplied_training_path_note': 'The supplied sqed_4pt_500k path contains four-point data. The model log names sqed_5pt_oneshot_train_499800_tok.csv.',
        'physics_background': 'https://doi.org/10.1007/JHEP08(2022)131 (section 2.2.1); expressions derived here from scalar QED vertices, not copied from the paper.',
    }
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
