"""Independent exact-tensor oracle for antisymmetry zero classifications.

The expected tensor values in this module are computed directly with
``fractions.Fraction`` and small 4-by-4 matrices.  In particular, this module
does not import the production verifier or any production field-strength
expansion helper.
"""
from __future__ import annotations

import itertools
import unittest
from fractions import Fraction
from typing import Iterable, Sequence

from data_gen.filter_antisymmetry_zeros import (
    OnShellAssumptions,
    zero_factor_reasons,
)


Scalar = Fraction
Vector = tuple[Scalar, Scalar, Scalar, Scalar]
Matrix = tuple[Vector, Vector, Vector, Vector]

ZERO = Fraction(0)
ONE = Fraction(1)
ETA: Matrix = (
    (ONE, ZERO, ZERO, ZERO),
    (ZERO, -ONE, ZERO, ZERO),
    (ZERO, ZERO, -ONE, ZERO),
    (ZERO, ZERO, ZERO, -ONE),
)
IDENTITY: Matrix = (
    (ONE, ZERO, ZERO, ZERO),
    (ZERO, ONE, ZERO, ZERO),
    (ZERO, ZERO, ONE, ZERO),
    (ZERO, ZERO, ZERO, ONE),
)
ZERO_MATRIX: Matrix = (
    (ZERO, ZERO, ZERO, ZERO),
    (ZERO, ZERO, ZERO, ZERO),
    (ZERO, ZERO, ZERO, ZERO),
    (ZERO, ZERO, ZERO, ZERO),
)


def _vector(values: Sequence[int]) -> Vector:
    if len(values) != 4:
        raise ValueError("a Lorentz vector must have four components")
    return tuple(Fraction(value) for value in values)  # type: ignore[return-value]


def _matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(
            sum((left[row][inner] * right[inner][column] for inner in range(4)), ZERO)
            for column in range(4)
        )
        for row in range(4)
    )  # type: ignore[return-value]


def _matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        sum((matrix[row][column] * vector[column] for column in range(4)), ZERO)
        for row in range(4)
    )  # type: ignore[return-value]


def _row_matrix(vector: Vector, matrix: Matrix) -> Vector:
    return tuple(
        sum((vector[row] * matrix[row][column] for row in range(4)), ZERO)
        for column in range(4)
    )  # type: ignore[return-value]


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(
        tuple(matrix[column][row] for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def _negate_matrix(matrix: Matrix) -> Matrix:
    return tuple(
        tuple(-matrix[row][column] for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def _minkowski_dot(left: Vector, right: Vector) -> Scalar:
    lowered_right = _matrix_vector(ETA, right)
    return sum(
        (left[index] * lowered_right[index] for index in range(4)),
        ZERO,
    )


def _minkowski_chain(
    left: Vector,
    operators: Iterable[Matrix],
    right: Vector,
) -> Scalar:
    transformed = right
    for operator in reversed(tuple(operators)):
        transformed = _matrix_vector(operator, transformed)
    return _minkowski_dot(left, transformed)


def _trace_word(operators: Iterable[Matrix]) -> Scalar:
    product = IDENTITY
    for operator in operators:
        product = _matrix_multiply(product, operator)
    return sum((product[index][index] for index in range(4)), ZERO)


def _lower_antisymmetric(
    independent_entries: Sequence[int],
) -> Matrix:
    """Build A_{mu nu} from entries 01, 02, 03, 12, 13, and 23."""

    if len(independent_entries) != 6:
        raise ValueError("a 4-by-4 antisymmetric matrix has six free entries")
    result = [[ZERO for _ in range(4)] for _ in range(4)]
    index_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for raw_value, (row, column) in zip(independent_entries, index_pairs):
        value = Fraction(raw_value)
        result[row][column] = value
        result[column][row] = -value
    return tuple(tuple(row) for row in result)  # type: ignore[return-value]


def _mixed_operator(lower_field_strength: Matrix) -> Matrix:
    """Raise the first index: F^mu_nu = eta^{mu rho} A_{rho nu}."""

    return _matrix_multiply(ETA, lower_field_strength)


def _exact_field_strength(momentum: Vector, polarisation: Vector) -> Matrix:
    """Construct F^mu_nu directly from exact test-side vectors."""

    lower_momentum = _matrix_vector(ETA, momentum)
    lower_polarisation = _matrix_vector(ETA, polarisation)
    return tuple(
        tuple(
            momentum[row] * lower_polarisation[column]
            - polarisation[row] * lower_momentum[column]
            for column in range(4)
        )
        for row in range(4)
    )  # type: ignore[return-value]


def _reason_codes(
    factor: str,
    *,
    assumptions: OnShellAssumptions | None = None,
) -> set[str]:
    return {
        reason.code
        for reason in zero_factor_reasons(
            factor,
            assumptions=assumptions,
        )
    }


LOWER_FIELDS: dict[int, Matrix] = {
    1: _lower_antisymmetric((1, 2, 3, 4, 5, 6)),
    2: _lower_antisymmetric((2, -1, 1, 3, -2, 4)),
    3: _lower_antisymmetric((-1, 3, 2, -4, 1, 5)),
}
MIXED_FIELDS: dict[int, Matrix] = {
    label: _mixed_operator(lower)
    for label, lower in LOWER_FIELDS.items()
}
GENERIC_MOMENTUM = _vector((2, -1, 3, 1))


class ExactAntisymmetryOracleTests(unittest.TestCase):
    def test_test_tensors_have_the_lorentzian_skew_adjoint_property(self) -> None:
        for label, lower in LOWER_FIELDS.items():
            with self.subTest(label=label):
                self.assertEqual(_transpose(lower), _negate_matrix(lower))
                mixed = MIXED_FIELDS[label]
                self.assertEqual(
                    _matrix_multiply(_transpose(mixed), ETA),
                    _negate_matrix(_matrix_multiply(ETA, mixed)),
                )

    def test_every_short_classifier_positive_word_is_exactly_zero(self) -> None:
        open_chain_claims = 0
        trace_claims = 0
        for length in (1, 3, 5):
            for word in itertools.product((1, 2, 3), repeat=length):
                field_text = " · ".join(f"F_{label}" for label in word)
                operators = tuple(MIXED_FIELDS[label] for label in word)

                open_factor = f"p_7 · {field_text} · p_7"
                if (
                    "antisymmetric_palindrome_chain"
                    in _reason_codes(open_factor)
                ):
                    open_chain_claims += 1
                    with self.subTest(kind="open", word=word):
                        self.assertEqual(
                            _minkowski_chain(
                                GENERIC_MOMENTUM,
                                operators,
                                GENERIC_MOMENTUM,
                            ),
                            ZERO,
                        )

                trace_factor = f"Tr({field_text})"
                if "antisymmetric_cyclic_trace" in _reason_codes(trace_factor):
                    trace_claims += 1
                    with self.subTest(kind="trace", word=word):
                        self.assertEqual(_trace_word(operators), ZERO)

        self.assertGreater(open_chain_claims, 0)
        self.assertGreater(trace_claims, 0)

        cyclic_but_not_palindromic = (2, 3, 2, 1, 1)
        self.assertNotEqual(
            cyclic_but_not_palindromic,
            tuple(reversed(cyclic_but_not_palindromic)),
        )
        cyclic_text = " · ".join(
            f"F_{label}" for label in cyclic_but_not_palindromic
        )
        self.assertIn(
            "antisymmetric_cyclic_trace",
            _reason_codes(f"Tr({cyclic_text})"),
        )
        self.assertEqual(
            _trace_word(
                MIXED_FIELDS[label]
                for label in cyclic_but_not_palindromic
            ),
            ZERO,
        )

    def test_odd_non_palindrome_is_generically_nonzero(self) -> None:
        word = (1, 2, 3)
        operators = tuple(MIXED_FIELDS[label] for label in word)
        field_text = " · ".join(f"F_{label}" for label in word)

        self.assertNotIn(
            "antisymmetric_palindrome_chain",
            _reason_codes(f"p_7 · {field_text} · p_7"),
        )
        self.assertNotIn(
            "antisymmetric_cyclic_trace",
            _reason_codes(f"Tr({field_text})"),
        )
        self.assertEqual(
            _minkowski_chain(
                GENERIC_MOMENTUM,
                operators,
                GENERIC_MOMENTUM,
            ),
            Fraction(-315),
        )
        self.assertEqual(_trace_word(operators), Fraction(311))

    def test_scoped_on_shell_rules_and_massive_counterexample(self) -> None:
        null_momentum = _vector((1, 1, 0, 0))
        transverse_polarisation = _vector((0, 0, 1, 0))
        massive_momentum = _vector((2, 0, 0, 0))
        field_two = _exact_field_strength(
            null_momentum,
            transverse_polarisation,
        )

        self.assertEqual(_minkowski_dot(null_momentum, null_momentum), ZERO)
        self.assertEqual(
            _minkowski_dot(null_momentum, transverse_polarisation),
            ZERO,
        )
        self.assertEqual(
            _row_matrix(_matrix_vector(ETA, null_momentum), field_two),
            (ZERO, ZERO, ZERO, ZERO),
        )
        self.assertEqual(
            _matrix_vector(field_two, null_momentum),
            (ZERO, ZERO, ZERO, ZERO),
        )
        field_two_squared = _matrix_multiply(field_two, field_two)
        self.assertNotEqual(field_two_squared, ZERO_MATRIX)
        self.assertEqual(
            _matrix_multiply(field_two_squared, field_two),
            ZERO_MATRIX,
        )
        self.assertEqual(
            _minkowski_chain(
                null_momentum,
                (field_two,),
                massive_momentum,
            ),
            ZERO,
        )
        self.assertEqual(
            _minkowski_chain(
                massive_momentum,
                (field_two,),
                null_momentum,
            ),
            ZERO,
        )

        sqed_like = OnShellAssumptions(
            massless_momenta=frozenset({2, 3}),
            transverse_field_strengths=frozenset({2, 3}),
        )

        self.assertNotIn(
            "ym_massless_momentum_square",
            _reason_codes("p_1 · p_1", assumptions=sqed_like),
        )
        self.assertNotIn(
            "ym_massless_momentum_square",
            _reason_codes("p_4 · p_4", assumptions=sqed_like),
        )
        self.assertEqual(
            _minkowski_dot(massive_momentum, massive_momentum),
            Fraction(4),
        )
        self.assertIn(
            "ym_massless_momentum_square",
            _reason_codes("p_2 · p_2", assumptions=sqed_like),
        )
        self.assertIn(
            "ym_left_self_contraction",
            _reason_codes("p_2 · F_2 · p_1", assumptions=sqed_like),
        )
        self.assertIn(
            "ym_right_self_contraction",
            _reason_codes("p_1 · F_2 · p_2", assumptions=sqed_like),
        )

        unscoped_same_endpoint_codes = _reason_codes(
            "p_1 · F_2 · p_1",
            assumptions=sqed_like,
        )
        self.assertIn(
            "antisymmetric_palindrome_chain",
            unscoped_same_endpoint_codes,
        )
        self.assertNotIn(
            "ym_left_self_contraction",
            unscoped_same_endpoint_codes,
        )
        self.assertNotIn(
            "ym_right_self_contraction",
            unscoped_same_endpoint_codes,
        )

        self.assertIn(
            "ym_nilpotent_field_cube",
            _reason_codes(
                "p_1 · F_2 · F_2 · F_2 · p_4",
                assumptions=sqed_like,
            ),
        )
        self.assertNotIn(
            "ym_nilpotent_field_cube",
            _reason_codes(
                "p_2 · F_1 · F_1 · F_1 · p_3",
                assumptions=sqed_like,
            ),
        )

        for factor in (
            "p_2 · p_2",
            "p_2 · F_2 · p_1",
            "p_1 · F_2 · p_2",
            "p_1 · F_2 · F_2 · F_2 · p_4",
        ):
            with self.subTest(no_assumptions=factor):
                self.assertFalse(
                    any(code.startswith("ym_") for code in _reason_codes(factor))
                )


if __name__ == "__main__":
    unittest.main()
