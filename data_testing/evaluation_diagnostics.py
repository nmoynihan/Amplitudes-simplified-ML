"""Structured, backend-independent numerical evidence for scalar expressions.

The caller owns strict parsing, kinematics, domain assumptions and randomness.
This module never draws new samples, relaxes tolerances or repairs expressions.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Callable, Iterable

from data_gen.numeric_utils import numeric_values_close


class ExpressionValidationError(ValueError):
    """Give a strict parser's failure an explicit machine-readable status."""

    def __init__(self, status: str, message: str) -> None:
        if status not in {"invalid_syntax", "unsupported_expression", "invalid_expression"}:
            raise ValueError(f"Unknown expression-validation status: {status!r}")
        self.status = status
        super().__init__(message)


def _json_number(value: float) -> float | str:
    """Keep nonfinite evidence without writing non-standard JSON NaN values."""
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _value_record(value: complex) -> dict[str, float | str]:
    return {"real": _json_number(value.real), "imag": _json_number(value.imag)}


def _exception_status(exc: Exception, *, validation: bool) -> str:
    if isinstance(exc, ExpressionValidationError):
        return exc.status
    if isinstance(exc, SyntaxError):
        return "invalid_syntax"
    if isinstance(exc, KeyError):
        return "unsupported_expression"
    if validation:
        return "invalid_expression"
    return "numerical_evaluation_error"


def compare_expressions(
    source: str,
    candidate: str,
    samples: Iterable[Any],
    *,
    evaluator: Callable[[str, Any], float | complex],
    validator: Callable[[str], None] | None = None,
    atol: float,
    rtol: float,
    required_valid_samples: int | None = None,
) -> dict[str, Any]:
    """Compare both expressions at every supplied sample and return JSON evidence.

    ``evaluator`` must evaluate a scalar using strict, fully consumed notation;
    ``validator``, when supplied, must raise on invalid expressions. The helper
    does not assume that syntactically identical expressions are finite or valid.

    By default all supplied samples must be valid, and an empty collection is
    inconclusive. For deterministic bounded replacement, callers may supply an
    ordered pool of samples and a positive ``required_valid_samples``. Only
    source arithmetic exceptions/nonfinite source values can be excluded from
    that pool. Unsupported notation and unexpected callback errors cannot be
    replaced. A candidate-only failure or *any* valid mismatch prevents equivalence,
    including failures after the required count has already been reached.

    Every finite pair uses exactly the shared ``numeric_values_close`` rule:
    ``abs(source-candidate) <= max(atol, rtol*max(abs(source),abs(candidate)))``.
    Relative error has no unit-scale floor; scaled error divides absolute error
    by that comparison threshold. A zero threshold yields scaled error zero
    only for an exact match, infinity otherwise. Nonfinite evidence is encoded
    as strings so ``json.dumps(result, allow_nan=False)`` always succeeds.

    A numerical mismatch takes verdict priority over evaluation errors because
    it supplies counterexample evidence. Other unsuccessful outcomes are
    inconclusive; ``equivalent`` is false for all of them. Sample indices refer
    to the supplied order, leaving seeds/reference/gauge metadata with callers.
    """
    # Validate with the exact comparator used below, even when samples are empty.
    numeric_values_close(0.0, 0.0, tol_abs=atol, tol_rel=rtol)
    points = list(samples)
    if required_valid_samples is None:
        required = max(1, len(points))
    elif (
        isinstance(required_valid_samples, bool)
        or not isinstance(required_valid_samples, int)
        or required_valid_samples <= 0
    ):
        raise ValueError("required_valid_samples must be a positive integer")
    else:
        required = required_valid_samples

    counts = {
        "available_samples": len(points),
        "requested_valid_samples": required,
        "attempted_samples": 0,
        "valid_source_samples": 0,
        "valid_samples": 0,
        "equivalent_samples": 0,
        "mismatched_samples": 0,
        "source_error_samples": 0,
        "candidate_error_samples": 0,
    }
    result: dict[str, Any] = {
        "status": "insufficient_valid_samples",
        "equivalent": False,
        "conclusive": False,
        "atol": float(atol),
        "rtol": float(rtol),
        "comparison_rule": "abs(source-candidate) <= max(atol, rtol*max(abs(source),abs(candidate)))",
        "relative_error_definition": "absolute_error/max(abs(source),abs(candidate)); zero for two zeros",
        "scaled_error_definition": "absolute_error/allowed_error; zero for an exact match with zero allowed_error, inf otherwise",
        "sample_policy": "all_supplied; source arithmetic exceptions/nonfinite values only may be replaced within the supplied bounded pool",
        "counts": counts,
        "first_failure": None,
        "first_mismatch": None,
        "failures": [],
        "checks": [],
    }

    def record_failure(failure: dict[str, Any]) -> None:
        result["failures"].append(failure)
        if result["first_failure"] is None:
            result["first_failure"] = failure

    if validator is not None:
        for role, expression in (("source", source), ("candidate", candidate)):
            try:
                if validator(expression) is False:
                    raise ValueError("Expression validator returned false")
            except Exception as exc:
                record_failure({
                    "status": _exception_status(exc, validation=True),
                    "phase": "validation",
                    "role": role,
                    "sample_index": None,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                })
        if result["failures"]:
            result["status"] = result["first_failure"]["status"]
            return result

    for sample_index, sample in enumerate(points):
        counts["attempted_samples"] += 1
        check: dict[str, Any] = {"sample_index": sample_index}
        values: dict[str, complex] = {}
        for role, expression in (("source", source), ("candidate", candidate)):
            try:
                value = evaluator(expression, sample)
                if not isinstance(value, Number):
                    raise TypeError("Numerical evaluator must return a real or complex scalar")
                value = complex(value)
                check[role] = _value_record(value)
                if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                    failure = {
                        "status": "nonfinite_value",
                        "phase": "evaluation",
                        "role": role,
                        "sample_index": sample_index,
                        "value": check[role],
                        "reason": f"Nonfinite {role} value",
                        "replaceable_source_sample": role == "source",
                    }
                elif not math.isfinite(abs(value)):
                    failure = {
                        "status": "nonfinite_value",
                        "phase": "evaluation",
                        "role": role,
                        "sample_index": sample_index,
                        "value": check[role],
                        "reason": f"Nonfinite {role} magnitude",
                        "replaceable_source_sample": role == "source",
                    }
                else:
                    values[role] = value
                    if role == "source":
                        counts["valid_source_samples"] += 1
                    continue
            except Exception as exc:
                failure = {
                    "status": _exception_status(exc, validation=False),
                    "phase": "evaluation",
                    "role": role,
                    "sample_index": sample_index,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "replaceable_source_sample": role == "source" and isinstance(exc, ArithmeticError),
                }
            # Do not compare on a source-invalid sample. Candidate failures are
            # recorded whenever the source is finite and can never be replaced.
            if "source" in check:
                failure["source"] = check["source"]
            if "candidate" in check:
                failure["candidate"] = check["candidate"]
            counts[f"{role}_error_samples"] += 1
            check["status"] = failure["status"]
            check["failure_role"] = role
            record_failure(failure)
            break
        else:
            left, right = values["source"], values["candidate"]
            difference = abs(left - right)
            scale = max(abs(left), abs(right))
            allowed = max(atol, rtol * scale)
            matches = numeric_values_close(left, right, tol_abs=atol, tol_rel=rtol)
            check.update({
                "status": "equivalent" if matches else "numerical_mismatch",
                "absolute_error": _json_number(difference),
                "relative_error": _json_number(difference / scale if scale else 0.0),
                "allowed_error": _json_number(allowed),
                "scaled_error": _json_number(difference / allowed if allowed else (0.0 if difference == 0 else math.inf)),
                "atol": float(atol),
                "rtol": float(rtol),
            })
            counts["valid_samples"] += 1
            if matches:
                counts["equivalent_samples"] += 1
            else:
                counts["mismatched_samples"] += 1
                failure = dict(check, phase="comparison", role="pair")
                record_failure(failure)
                if result["first_mismatch"] is None:
                    result["first_mismatch"] = failure
        result["checks"].append(check)

    if counts["mismatched_samples"]:
        result.update(status="numerical_mismatch", conclusive=True)
    elif any(
        failure["role"] == "source" and not failure.get("replaceable_source_sample", False)
        for failure in result["failures"]
    ):
        result["status"] = next(
            failure["status"] for failure in result["failures"]
            if failure["role"] == "source" and not failure.get("replaceable_source_sample", False)
        )
    elif counts["candidate_error_samples"]:
        result["status"] = next(
            failure["status"] for failure in result["failures"]
            if failure["role"] == "candidate"
        )
    elif counts["valid_samples"] >= required:
        result.update(status="equivalent", equivalent=True, conclusive=True)
    return result


__all__ = ["ExpressionValidationError", "compare_expressions"]
