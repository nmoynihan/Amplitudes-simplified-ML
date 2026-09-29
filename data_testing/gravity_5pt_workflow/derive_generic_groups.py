"""Derive exact scalar-only groups directly from the original amplitude CSV.

This script performs no model inference and loads no previous compact answer,
model target, or earlier decomposition. Field-strength identities are not used
in this scalar-only derivation. It emits two scalar polynomial groups suitable
for separate model inference, together with their exact external weights and a
remaining scalar completed square.
"""
import hashlib
import json
from pathlib import Path
import re

import sympy as s


from .symbolic_utils import symbolic
from .common import read_source


def parse_scalar_expression(text: str) -> s.Expr:
    """Map scalar dots to independent atoms using the repository parser."""
    if re.search(r"\b(?:F_\d+|Tr)\b", text):
        raise ValueError("Generic grouping requires a scalar-only source expression")
    return symbolic(text)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(source: Path, output_dir: Path) -> dict:
    """Derive both groups from ``source`` and save their exact certificates."""
    source, output_dir = Path(source), Path(output_dir)
    source_bytes = source.read_bytes()
    raw = read_source(source)
    original = parse_scalar_expression(raw)
    a, x, y, b, z, w, g = s.symbols('e4p1 e4p2 e4p5 e5p1 e5p3 e5p4 e4e5')
    h, k, q, u, v, t = s.symbols('p1p4 p1p5 p2p3 p2p4 p3p5 p4p5')

    # Complete the square in e4.e5 using coefficients from the input itself.
    g_polynomial = s.Poly(original, g)
    require(g_polynomial.degree() == 2, "Expected a quadratic in e4.e5")
    quadratic = s.factor(g_polynomial.coeff_monomial(g**2))
    linear = s.factor(g_polynomial.coeff_monomial(g))
    shift = s.factor(linear / (2 * quadratic))
    scalar_block = quadratic * (g + shift)**2
    remainder = s.expand(original - scalar_block)
    require(g not in remainder.free_symbols, "Completing the square left e4.e5 dependence")

    # Each remaining term has degree two in each of the two polarizations.
    left_basis = [a*a, a*x, a*y, x*x, x*y, y*y]
    right_basis = [b*b, b*z, b*w, z*z, z*w, w*w]
    polynomial = s.Poly(remainder, a, x, y, b, z, w)
    matrix = s.Matrix(6, 6, lambda i, j: s.factor(
        polynomial.coeff_monomial(left_basis[i] * right_basis[j])))
    require(s.cancel(remainder - sum(matrix[i, j] * left_basis[i] * right_basis[j]
                                    for i in range(6) for j in range(6))) == 0,
            "Remainder is outside the expected polarization biquadratic basis")

    # Choose successive nonzero diagonal pivots; no field-strength target is used.
    residual_matrix = matrix
    blocks, pivot_indices = [], []
    while residual_matrix != s.zeros(6):
        pivot = next((i for i in range(6) if residual_matrix[i, i] != 0), None)
        require(pivot is not None, "Nonzero residual matrix has no supported diagonal pivot")
        rank_one = (residual_matrix[:, pivot] * residual_matrix[pivot, :] /
                    residual_matrix[pivot, pivot])
        block = s.factor(sum(rank_one[i, j] * left_basis[i] * right_basis[j]
                             for i in range(6) for j in range(6)))
        blocks.append(block)
        pivot_indices.append(pivot)
        residual_matrix = (residual_matrix - rank_one).applyfunc(s.factor)
    require(len(blocks) == 2, "Expected exactly two rank-one polarization blocks")
    exact_residual = s.cancel(original - scalar_block - sum(blocks))
    require(exact_residual == 0, "Generic scalar reconstruction failed")

    # Keep all momentum-only factors outside the groups. The model receives just
    # a scalar polynomial numerator divided by six distinct momentum dots.
    d12, d13 = s.symbols('p1p2 p1p3')
    training_denominator = d12*d13*h*k*u*v
    groups = []
    for index, block in enumerate(blocks, 1):
        outside, core = block.as_independent(a, x, y, b, z, w, as_Add=False)
        model_scalar_group = core / training_denominator
        external_weight = s.factor(outside * training_denominator)
        require(s.cancel(block - external_weight*model_scalar_group) == 0,
                "External weight does not reconstruct its scalar block")
        groups.append({
            'name': f'generic_group_{index}',
            'pivot_index': pivot_indices[index-1],
            'block': str(block),
            'scalar_numerator_factored': str(core),
            'scalar_group_for_inference': str(model_scalar_group),
            'external_weight': str(external_weight),
            'weight_times_group_residual': '0',
        })

    # Independent check of the common-reference reduction used previously.
    common_gauge = {a: 0, b: 0, g: 0}
    same_helicity_form = (-(t*x-u*y)**2*(t*z-v*w)**2/(u*v*t**4) +
                          (t*t-3*h*k)*y*y*w*w/(6*t**4))
    same_helicity_residual = s.cancel(original.subs(common_gauge)-same_helicity_form)
    require(same_helicity_residual == 0, "Common-reference reduction failed")

    report = {
        'source': str(source.resolve()),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'method': 'Complete square in e4.e5; two rank-one pivots of remaining scalar biquadratic coefficient matrix.',
        'input_dependencies': ['Original CSV only. No compact answer, F target, or model output is loaded.'],
        'neural_inference_performed_by_this_script': False,
        'symbol_convention': 'e4p2 means e_4 dot p_2; p2p4 means p_2 dot p_4; e4e5 means e_4 dot e_5.',
        'quadratic_coefficient': str(quadratic),
        'linear_coefficient': str(linear),
        'completed_square_shift': str(shift),
        'remaining_scalar_block': str(scalar_block),
        'left_basis': list(map(str, left_basis)),
        'right_basis': list(map(str, right_basis)),
        'coefficient_matrix': [[str(matrix[i, j]) for j in range(6)] for i in range(6)],
        'pivot_indices': pivot_indices,
        'rank': len(blocks),
        'final_residual_matrix': [[str(residual_matrix[i, j]) for j in range(6)] for i in range(6)],
        'training_denominator': str(training_denominator),
        'groups': groups,
        'generic_identity_residual': str(exact_residual),
        'common_reference_identity_residual': str(same_helicity_residual),
        'generic_validity': 'Exact rational identity treating all scalar dots as independent. No helicity, masslessness, or momentum conservation assumptions.',
        'limitations': [
            'This script demonstrates algebraic preprocessing, not neural compression.',
            'A model prediction must be checked for equality with its scalar group before substitution.',
            'Momentum-only weights and the remaining scalar block are necessary in the full reconstruction.',
            'Optional rewriting of the remaining scalar square as a field-strength trace is symbolic postprocessing, not a model prediction.',
            'Introduced coordinate denominators are understood on their common domain; the final equality is an identity of rational functions.',
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'generic_derivation.json').write_text(json.dumps(report, indent=2)+'\n')
    (output_dir/'generic_scalar_identity.txt').write_text(str(scalar_block+sum(blocks))+'\n')
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = main(args.source, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "rank", "pivot_indices", "generic_identity_residual",
        "common_reference_identity_residual",
    )}, indent=2))
