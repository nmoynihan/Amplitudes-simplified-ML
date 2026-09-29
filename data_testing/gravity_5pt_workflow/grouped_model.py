"""Original common-reference scalar grouping, without model inference.

Adapted from the 25 September 2026 workflow. The coefficient decomposition,
projection and normalization are analytic operations in the ++ sector.
"""
from pathlib import Path
import csv
import hashlib
import json

import sympy as sp
from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_gen.data_gen_gravity.core import expand_expression
from .symbolic_utils import symbolic, render, safe
from .common import read_source, require


def main(source: Path, output_dir: Path):
    original = read_source(source)
    original_sym = symbolic(original)
    x, y, z, w = sp.symbols("e4p2 e4p5 e5p3 e5p4")
    h, k, u, v, t = sp.symbols("p1p4 p1p5 p2p4 p3p5 p4p5")
    zeros = {sp.Symbol(name): 0 for name in ("e4p1", "e5p1", "e4e5")}
    fixed = sp.expand(original_sym.subs(zeros))
    rows, columns = [x*x, x*y, y*y], [z*z, z*w, w*w]
    poly = sp.Poly(fixed, x, y, z, w)
    matrix = sp.Matrix([[poly.coeff_monomial(a*b) for b in columns] for a in rows])
    pivot = matrix[0, 0]
    first = sp.factor(sum(matrix[i, 0]*rows[i] for i in range(3)) *
                      sum(matrix[0, j]*columns[j] for j in range(3)) / pivot)
    second = sp.factor(fixed-first)
    require(sp.cancel(first+second-fixed) == 0, 'Verification failed: sp.cancel(first+second-fixed) == 0')
    residual_matrix = matrix-matrix[:, 0]*matrix[0, :]/pivot
    require(residual_matrix.rank() == 1, 'Verification failed: residual_matrix.rank() == 1')

    # Reference projection in scalar notation only. It agrees in common q=p1.
    projection = {x:x-u*sp.Symbol("e4p1")/h,
                  y:y-t*sp.Symbol("e4p1")/h,
                  z:z-v*sp.Symbol("e5p1")/k,
                  w:w-t*sp.Symbol("e5p1")/k}
    groups = [sp.factor(part.subs(projection, simultaneous=True)) for part in (first, second)]
    require(sp.cancel((sum(groups)-original_sym).subs(zeros)) == 0, 'Verification failed: sp.cancel((sum(groups)-original_sym).subs(zeros)) == 0')
    D = sp.prod(sp.Symbol(f"p{a}p{b}") for a,b in [(1,2),(1,3),(1,4),(1,5),(2,4),(3,5)])
    tok = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
    prepared = []
    for index, group in enumerate(groups, 1):
        scalar, polarization = sp.Integer(1), sp.Integer(1)
        for factor in sp.Mul.make_args(group):
            if any(str(atom).startswith("e") for atom in factor.free_symbols):
                polarization *= factor
            else:
                scalar *= factor
        require(sp.cancel(group-scalar*polarization) == 0, 'Verification failed: sp.cancel(group-scalar*polarization) == 0')
        core_scalar = polarization/D
        factored_scalar_text = safe(render(core_scalar))
        require("F_" not in factored_scalar_text, 'Verification failed: "F_" not in factored_scalar_text')
        model_input = safe(expand_expression(factored_scalar_text, full=True))
        ids = tok.encode_infix(model_input)
        require(1 not in ids, 'Verification failed: 1 not in ids')
        require(sp.cancel(symbolic(model_input)-core_scalar) == 0, 'Verification failed: sp.cancel(symbolic(model_input)-core_scalar) == 0')
        require(sp.cancel(symbolic(tok.decode_infix(ids))-core_scalar) == 0, 'Verification failed: sp.cancel(symbolic(tok.decode_infix(ids))-core_scalar) == 0')
        prepared.append({"name":f"group_{index}", "input":model_input,
                         "factored_scalar_input":factored_scalar_text,
                         "input_tokens":len(ids), "ids":ids,
                         "weight":safe(render(sp.cancel(scalar*D))),
                         "scalar_group":str(group)})
    (output_dir/"grouped_inputs.json").write_text(json.dumps(prepared,indent=2)+"\n")
    with (output_dir/"grouped_inputs.csv").open("w") as handle:
        csv.writer(handle).writerows((r["name"],r["input"]) for r in prepared)
    derivation = {"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                  "coefficient_matrix":str(matrix),"coefficient_matrix_rank":matrix.rank(),
                  "rank_one_group":str(first),"residual_group":str(second),
                  "residual_matrix":str(residual_matrix),
                  "exact_common_reference_reconstruction":"0",
                  "compact_target_read_or_constructed":False,
                  "analytic_steps":["common-reference helicity reduction", "coefficient matrix rank decomposition",
                                    "scalar polynomial factoring", "scalar gauge-invariant projection", "external scalar normalization"],
                  "scope":"four-dimensional positive-positive helicity, generic nonzero reference and normalization denominators"}
    (output_dir/"grouped_derivation.json").write_text(json.dumps(derivation,indent=2)+"\n")
    print(json.dumps({"prepared":[{k:r[k] for k in ("name","input_tokens","weight")} for r in prepared]}),flush=True)
    return prepared
