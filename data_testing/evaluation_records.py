"""Strict raw-hypothesis checks and truthful simplification metrics.

No token repair occurs here. Raw decoder evidence is retained even when an
expression cannot be used for numerical comparison.
"""
from __future__ import annotations

from typing import Any, Callable


def inspect_sequence(raw_ids, tokenizer) -> dict[str, Any]:
    raw = list(raw_ids)
    errors = []
    known = set(tokenizer.id_to_token)
    if any(type(token) is not int for token in raw):
        errors.append("non_integer_token")
    unknown = [i for i, token in enumerate(raw) if type(token) is int and token not in known]
    if unknown:
        errors.append("unmapped_token")
    if not raw or raw[0] != 2:
        errors.append("missing_bos")
    eos = next((i for i, token in enumerate(raw) if token == 3), None)
    content = raw[1:eos] if raw and raw[0] == 2 else raw[:eos]
    if any(token in (0, 1, 2) for token in content):
        errors.append("illegal_internal_special_token")
    if eos is not None and any(token != 0 for token in raw[eos + 1:]):
        errors.append("unexpected_content_after_eos")
    if not content:
        errors.append("empty_prediction")
    expression, decode_error = "", None
    if not errors:
        try:
            expression = tokenizer.decode_infix(content)
        except (ValueError, KeyError, IndexError, RecursionError) as exc:
            decode_error = f"{type(exc).__name__}: {exc}"
            errors.append("prefix_decode_error")
    return {
        "raw_token_ids": raw, "tokens": content, "content_token_count": len(content),
        "eos_emitted": eos is not None, "completed": eos is not None,
        "completion_status": "eos" if eos is not None else "missing_eos",
        "token_valid": not errors, "decode_ok": not errors,
        "expr": expression, "sequence_errors": errors,
        "unmapped_positions": unknown,
        "decode_error": decode_error or (", ".join(errors) if errors else ""),
    }


def check_candidate(raw_ids, tokenizer, *, source_expression: str,
                    source_tokens: list[int], compare: Callable,
                    reference_expression: str | None = None,
                    reference_tokens: list[int] | None = None,
                    index: int = 0) -> dict[str, Any]:
    record = inspect_sequence(raw_ids, tokenizer)
    record["index"] = index
    if record["decode_ok"]:
        numerical = compare(source_expression, record["expr"])
        reference_check = (compare(reference_expression, record["expr"])
                           if reference_expression is not None else None)
    else:
        numerical = {"status": "token_decode_error", "equivalent": False,
                     "error": record["decode_error"]}
        reference_check = None
    # A complete parse can still contain an unsupported scalar/vector symbol.
    invalid_expression = numerical["status"] in {
        "invalid_syntax", "invalid_expression", "unsupported_expression"}
    record["syntax_valid"] = record["decode_ok"] and not invalid_expression
    equivalent = bool(numerical["equivalent"])
    record.update({
        "numerical_check": numerical, "reference_check": reference_check,
        "num_eq_scrambled": equivalent,
        "num_eq_simple": (bool(reference_check and reference_check["equivalent"])
                          if reference_expression is not None else None),
        "reference_provided": reference_expression is not None,
        "exact_token": (record["decode_ok"] and record["completed"] and record["tokens"] == reference_tokens
                        if reference_tokens is not None else None),
        "exact_string": None,
        "source_token_count": len(source_tokens),
        "token_reduction": len(source_tokens) - len(record["tokens"]),
        "copies_input": record["decode_ok"] and record["tokens"] == source_tokens,
        "shorter_equivalent": (equivalent and record["syntax_valid"]
                               and record["completed"]
                               and len(record["tokens"]) < len(source_tokens)),
    })
    if reference_tokens is not None:
        try:
            record["exact_string"] = (record["decode_ok"] and record["completed"] and record["expr"]
                                      == tokenizer.decode_infix(reference_tokens))
        except (ValueError, KeyError, IndexError, RecursionError):
            record["exact_string"] = False
    return record
