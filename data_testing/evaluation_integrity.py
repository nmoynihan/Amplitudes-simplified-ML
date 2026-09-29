"""Check actual evaluator token rows against their unmodified expressions."""

from __future__ import annotations

from typing import Any, Callable, Sequence


class InputIntegrityError(ValueError):
    """A failed preflight, retaining the attempted check as report evidence."""

    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


def _exact_roundtrip(original: str, decoded: str, process: str | None) -> dict[str, Any]:
    if process not in {"3s2h", "4s1h"}:
        return {"available": False, "ok": None, "reason": "no_exact_backend_for_process"}
    from data_gen.data_gen_gravity.v2_validation import parse_exact, validate_token_roundtrip
    import sympy as sp

    check = validate_token_roundtrip(original, decoded)
    if check["reason"] != "token_parse_error":
        return dict(check, available=True, method="exact_laurent_normalization")
    # Laurent normalization excludes sums in denominators. The exact rational
    # parser still supports them; compare its symbolic scalar form instead.
    try:
        left, right = parse_exact(original), parse_exact(decoded)
        ok = left == right or sp.cancel(left - right) == 0
        return {
            "available": True, "ok": bool(ok),
            "reason": "ok" if ok else "token_semantics_mismatch",
            "method": "exact_rational_normalization",
        }
    except (ValueError, ZeroDivisionError, TypeError, NotImplementedError) as exc:
        # Strict numerical comparison below remains mandatory and catches
        # unsupported syntax. Unavailable exact support is never an acceptance.
        return {
            "available": False, "ok": None, "reason": "exact_parser_unavailable",
            "detail": f"{type(exc).__name__}: {exc}",
        }


def validate_input_integrity(
    raw_rows: Sequence[dict[str, Any]],
    token_rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    processes: Sequence[str | None] | None,
    compare: Callable[[str, str, str | None], dict[str, Any]],
    *,
    reference_provided: bool | Sequence[bool] = True,
) -> list[dict[str, Any]]:
    """Validate source and each explicitly independent reference before inference.

    ``token_rows`` contain prefix *content* IDs, without BOS/EOS/padding. The
    function decodes those actual IDs, rather than re-encoding the raw text or
    trusting the tokenizer's permissive removal of special tokens. Model input
    length is content length plus its one BOS and one EOS token; checkpoint
    capacity remains the caller's explicit check.

    ``compare(original, decoded, process)`` must use a strict numerical backend
    at independent cached kinematics and return a structured result. Exact
    normalization, when supported, and numerical evidence must both agree.
    For single-amplitude inputs lacking a separate target pass
    ``reference_provided=False``; copied internal target fields are not treated
    as independent references. No expression or token sequence is modified.
    """
    if len(raw_rows) != len(token_rows):
        raise ValueError("Input integrity: raw and token row counts differ")
    if not raw_rows:
        raise ValueError("Input integrity: no evaluation input rows")
    assigned_processes = list(processes) if processes is not None else [None] * len(raw_rows)
    if len(assigned_processes) != len(raw_rows):
        raise ValueError("Input integrity: process metadata is not aligned with rows")
    if isinstance(reference_provided, bool):
        references = [reference_provided] * len(raw_rows)
    else:
        references = list(reference_provided)
        if len(references) != len(raw_rows) or any(type(flag) is not bool for flag in references):
            raise ValueError("Input integrity: reference provenance must be aligned booleans")

    known_ids = set(tokenizer.id_to_token)
    special_ids = {
        tokenizer.vocab[name] for name in ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
    }
    records: list[dict[str, Any]] = []
    for row_index, (raw, token_row, process, has_reference) in enumerate(
        zip(raw_rows, token_rows, assigned_processes, references, strict=True)
    ):
        row_record: dict[str, Any] = {
            "row_index": row_index, "process": process,
            "reference_provided": has_reference, "source": None, "reference": None,
            "token_count_convention": "prefix content tokens; model input adds one BOS and one EOS",
        }
        fields = (("scrambled", "source"), ("simple", "reference")) if has_reference else (("scrambled", "source"),)
        for field, role in fields:
            original, tokens = raw.get(field), token_row.get(field)
            check: dict[str, Any] = {
                "status": "unchecked", "field": field,
                "original_expression": original,
                "original_tokens": list(tokens) if isinstance(tokens, list) else tokens,
                "decoded_expression": None,
                "content_token_count": len(tokens) if isinstance(tokens, list) else None,
                "bos_eos_token_count": len(tokens) + 2 if isinstance(tokens, list) else None,
                "exact_check": None, "numerical_check": None,
            }
            row_record[role] = check

            def fail(reason: str, *, status: str = "representation_error") -> None:
                check["status"] = status
                check["reason"] = reason
                raise InputIntegrityError(
                    f"Input integrity row {row_index + 1} {field}: {status}: {reason}",
                    row_record,
                )

            if not isinstance(original, str) or not original.strip():
                fail("original expression must be a nonempty string", status="invalid_expression")
            if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
                fail("content tokens must be a nonempty list of integers")
            unknown = [token for token in tokens if token not in known_ids]
            if unknown:
                fail(f"out-of-vocabulary token IDs: {unknown}")
            if any(token in special_ids for token in tokens):
                fail("content tokens contain a special token; internal tokens cannot be removed")
            try:
                decoded = tokenizer.decode_infix(tokens)
            except (ValueError, KeyError, IndexError, RecursionError) as exc:
                fail(f"prefix decoding failed: {type(exc).__name__}: {exc}")
            check["decoded_expression"] = decoded
            exact = _exact_roundtrip(original, decoded, process)
            check["exact_check"] = exact
            if exact["available"] and not exact["ok"]:
                fail("token decoding changes exact expression semantics (possibly ambiguous adjacent numeric leaves)")
            try:
                numerical = compare(original, decoded, process)
            except Exception as exc:
                fail(f"numerical check failed: {type(exc).__name__}: {exc}", status="input_integrity_error")
            check["numerical_check"] = numerical
            if not isinstance(numerical, dict) or not numerical.get("equivalent", False):
                status = numerical.get("status", "invalid_comparison_result") if isinstance(numerical, dict) else "invalid_comparison_result"
                detail = numerical.get("first_failure") if isinstance(numerical, dict) else None
                fail(f"token round trip failed numerical verification: {status}; {detail}", status="input_integrity_error")
            check["status"] = "verified"
        records.append(row_record)
    return records


__all__ = ["InputIntegrityError", "validate_input_integrity"]
