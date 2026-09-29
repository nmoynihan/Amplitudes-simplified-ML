"""Exact prediction verification and the original 40 fresh numerical checks."""
import math

import sympy as sp

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_gen.data_gen_ym.generate_clean_4pt import parenthesize_for_semantic_tokenization
from data_gen.data_gen_ym.kinematics import generate_kinematics
from data_gen.data_gen_ym.numerics import eval_infix_numeric

from .algebra import symbolic


def require_equal(left, right, description):
    if sp.cancel(left - right) != 0:
        raise ValueError(f"{description} failed exact symbolic verification")


def verify_prediction(source: str, prediction: str, reference: str) -> dict:
    """Verify the actual returned expression, never replace it with a target."""
    reference_value = symbolic(reference)
    if reference_value == 0:
        raise ValueError("The reference amplitude must be nonzero")
    require_equal(symbolic(source), reference_value, "Model input")
    require_equal(symbolic(prediction), reference_value, "Model prediction")
    rendered = parenthesize_for_semantic_tokenization(prediction)
    require_equal(symbolic(rendered), reference_value, "Exported prediction")
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
    for description, expression in (("Model input", source), ("Exported prediction", rendered)):
        ids = tokenizer.encode_infix(expression)
        decoded = tokenizer.decode_infix(ids)
        require_equal(symbolic(decoded), reference_value, f"{description} token roundtrip")

    maximum = 0.0
    points = []
    for seed in range(700, 720):
        for mode in ("coulomb", "covariant"):
            momenta, polarizations = generate_kinematics(4, seed=seed, pol_mode=mode)
            left = eval_infix_numeric(reference, momenta, polarizations, strict=True)
            right = eval_infix_numeric(prediction, momenta, polarizations, strict=True)
            if not (math.isfinite(left) and math.isfinite(right)):
                raise ValueError(f"Nonfinite numerical value at seed {seed}, {mode}")
            residual = abs(left - right)
            if residual > 1e-10 + 1e-8 * max(abs(left), abs(right)):
                raise ValueError(f"Numerical verification failed at seed {seed}, {mode}")
            maximum = max(maximum, residual)
            points.append({"seed": seed, "polarization_mode": mode, "absolute_residual": residual})
    return {
        "rendered_prediction": rendered,
        "exact_difference_from_completed_amplitude": "0",
        "export_and_token_roundtrip_residuals": "0",
        "independent_numeric_points": len(points),
        "numerical_atol": 1e-10, "numerical_rtol": 1e-8,
        "max_absolute_numeric_residual": maximum,
        "numerical_checks": points,
    }
