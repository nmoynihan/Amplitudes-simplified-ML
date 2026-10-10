#!/usr/bin/env python3
"""Derive and validate the full coupling-stripped tree-level SQED amplitude.

All momenta outgoing, metric (+---), scalar legs 1,4 and photon legs 2,3.
The CSV contains A with i*M = i*q^2*A. Use cubic i*q*(2*P+k).e,
contact 2*i*q^2*e2.e3 and scalar propagator i/(P^2-m^2).
Run this script, then build_csv.mjs, then this script with --check-csv.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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
TRAIN = Path('/Users/kymani/Documents/SQED_KYMANI/sqed_4pt_500k/sqed_4pt_oneshot_train_499800.csv.gz')
MODEL = Path('/Users/kymani/Documents/SQED_KYMANI/SQED/4PT_Model/best_model.pt')


def expressions():
    return {
        'simple': '-2*(p_1 · F_2 · F_3 · p_1)/((p_1 · p_2)*(p_1 · p_3))',
        'scrambled': ('2*(e_2 · e_3)'
                      ' + 2*(e_2 · p_1)*(e_3 · p_4)/(p_1 · p_2)'
                      ' + 2*(e_3 · p_1)*(e_2 · p_4)/(p_1 · p_3)'),
    }


def symbolic(expression):
    """Expand the repository's F chains into exact independent dot symbols."""
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
            return sp.Symbol('_'.join(sorted(f'{v[1]}{v[2]}' for v in node[1:])))
        raise TypeError(type(node))
    return visit(gd._expand_ast(gd._Parser(gd._tokenize(expression)).parse()))


def invariant(a, b):
    return sp.Symbol('_'.join(sorted((a, b))))


def on_shell(expr):
    # p4=-p1-p2-p3; p4^2=p1^2=m^2 implies p2.p3=-p1.p2-p1.p3.
    rules = {invariant('e2','p2'): 0, invariant('e3','p3'): 0,
             invariant('p2','p2'): 0, invariant('p3','p3'): 0,
             invariant('e2','p4'): -invariant('e2','p1')-invariant('e2','p3'),
             invariant('e3','p4'): -invariant('e3','p1')-invariant('e3','p2'),
             invariant('p2','p4'): -invariant('p1','p2')-invariant('p2','p3'),
             invariant('p3','p4'): -invariant('p1','p3')-invariant('p2','p3')}
    result = expr.subs(rules, simultaneous=True)
    return sp.cancel(result.subs(invariant('p2','p3'),
                                -invariant('p1','p2')-invariant('p1','p3')))


def ward(expr, leg):
    rules = {s: sp.Symbol('_'.join(sorted(
        f'p{leg}' if part == f'e{leg}' else part for part in s.name.split('_'))))
        for s in expr.free_symbols if f'e{leg}' in s.name.split('_')}
    return on_shell(expr.subs(rules, simultaneous=True))


def direct_diagrams(p, e, mass):
    """Three independent Feynman diagrams using unreduced propagators."""
    value = 2*mdot(e[0], e[1])
    for i, j in ((0, 1), (1, 0)):
        internal = p[0]+p[i+1]
        value -= (mdot(2*p[0]+p[i+1], e[i])*
                  mdot(2*internal+p[j+1], e[j]) /
                  (mdot(internal, internal)-mass**2))
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-csv', action='store_true')
    args = parser.parse_args()
    row = expressions()
    sym = {key: symbolic(value) for key, value in row.items()}
    assert on_shell(sym['simple']-sym['scrambled']) == 0
    assert all(ward(expr, leg) == 0 for expr in sym.values() for leg in (2, 3))
    swap = {s: sp.Symbol('_'.join(sorted(
        {'e2':'e3','e3':'e2','p2':'p3','p3':'p2'}.get(part, part)
        for part in s.name.split('_')))) for s in sym['scrambled'].free_symbols}
    assert on_shell(sym['scrambled'].xreplace(swap)-sym['scrambled']) == 0

    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=5000)
    tokens = {key: tokenizer.encode_infix(value) for key, value in row.items()}
    for key, ids in tokens.items():
        assert 1 not in ids and max(ids) < 58 and len(ids)+2 <= 5000
        assert on_shell(symbolic(tokenizer.decode_infix(ids))-sym[key]) == 0
    train_tok = TRAIN.with_name(TRAIN.name.replace('.csv.gz','_tok.csv.gz'))
    with gzip.open(TRAIN, 'rt') as handle:
        raw_training_row = next(csv.DictReader(handle))
    with gzip.open(train_tok, 'rt') as handle:
        training_tokens = next(csv.DictReader(handle))
    assert all(tokenizer.encode_infix(raw_training_row[k]) == json.loads(training_tokens[k])
               for k in row)

    errors = dict(reference_vs_input=0., independent_diagrams=0., ward=0.,
                  gauge_shift=0., photon_exchange=0.)
    def check(label, a, b):
        error = abs(a-b)/max(1., abs(a), abs(b))
        errors[label] = max(errors[label], error)
        assert np.isfinite(a) and np.isfinite(b) and error < 1e-9, (label, a, b)

    count = 0
    for mass in (0.7, 2., 5.):
        for mode in ('coulomb', 'covariant'):
            for seed in range(20):
                p, e = generate_kinematics(4, M=mass, pol_mode='coulomb', seed=8100+seed)
                # Random transverse directions exercise both photon polarisations.
                rng = np.random.default_rng(9100+seed)
                for k in range(2):
                    other = np.cross(p[k+1,1:]/np.linalg.norm(p[k+1,1:]), e[k,1:])
                    angle = rng.uniform(0., 2*np.pi)
                    e[k] = np.cos(angle)*e[k]+np.sin(angle)*np.r_[0., other]
                if mode == 'covariant':
                    e += rng.uniform(-0.5, 0.5, (2,1))*p[1:3]
                assert np.max(abs(p.sum(axis=0))) < 1e-10
                assert all(abs(mdot(e[k], p[k+1])) < 1e-10 for k in range(2))
                values = {key: gd.eval_infix_numeric(value, p, e, strict=True)
                          for key, value in row.items()}
                check('reference_vs_input', values['simple'], values['scrambled'])
                check('independent_diagrams', values['scrambled'], direct_diagrams(p,e,mass))
                for key in row:
                    for k in range(2):
                        changed = e.copy(); changed[k] = p[k+1]
                        check('ward', gd.eval_infix_numeric(row[key],p,changed,strict=True),0.)
                    shifted = e+np.array([0.23,-0.37])[:,None]*p[1:3]
                    check('gauge_shift',gd.eval_infix_numeric(row[key],p,shifted,strict=True),values[key])
                    check('photon_exchange',gd.eval_infix_numeric(row[key],p[[0,2,1,3]],e[[1,0]],strict=True),values[key])
                count += 1

    if args.check_csv:
        with (OUT/'sqed_4pt_physical.csv').open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == ['simple', 'scrambled']
            assert list(reader) == [row]
    else:
        (OUT/'amplitude.json').write_text(json.dumps(row, indent=2, ensure_ascii=False)+'\n')
    report = {
        'process': 'phi(p1), gamma(p2), gamma(p3), phi_conjugate(p4)',
        'conventions': 'All outgoing; metric (+---); p1^2=p4^2=m^2; photons massless/transverse; i*q^2 stripped.',
        'construction': 'Two scalar exchange diagrams and one scalar-scalar-photon-photon contact diagram.',
        'reference_derivation': 'Set E_i=e_i-p_i*(p1.e_i)/(p1.p_i). Exchange diagrams vanish; 2*E2.E3=-2*p1.F2.F3.p1/(p1.p2*p1.p3).',
        'exact_reference_equivalence': True, 'exact_ward_legs': [2,3],
        'exact_bose_symmetry': True, 'numerical_phase_space_points': count,
        'masses': [0.7, 2., 5.], 'polarisation_modes': ['coulomb','covariant'],
        'max_normalized_errors': errors,
        'token_counts_without_bos_eos': {k:len(v) for k,v in tokens.items()},
        'tokenizer_max_particles': 8, 'training_token_row_exactly_reproduced': True,
        'training_data': str(TRAIN), 'model_path': str(MODEL),
        'model_vocabulary_size': 58, 'model_sequence_capacity': 5000,
        'saved_csv_exactly_verified': args.check_csv,
        'normalization_note': 'Older repository Compton seed chain/(a*b) equals this physical amplitude divided by -2.',
        'physics_background': 'https://arxiv.org/abs/1309.0804, equations 2.11, 2.18, 2.21; formula independently derived from the stated scalar QED vertices.',
    }
    (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
