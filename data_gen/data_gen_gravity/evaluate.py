#!/usr/bin/env python3
"""Evaluate gravity predictions with complex kinematics and grouped metrics."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..Tokenizer import ScatteringAmplitudeTokenizer
from .core import numerically_equivalent


def _open(path: str | Path, mode: str = "r"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", newline="", encoding="utf-8")
    return open(path, mode, newline="", encoding="utf-8")


def _read(path: str | Path) -> list[dict[str, str]]:
    with _open(path) as handle:
        return list(csv.DictReader(handle))


def _normalise(expr: str) -> str:
    return re.sub(r"\s+", "", expr).replace("**", "^")


def _strip_special(sequence: Iterable[int]) -> list[int]:
    output: list[int] = []
    for raw in sequence:
        token = int(raw)
        if token in (0, 2):
            continue
        if token == 3:
            break
        output.append(token)
    return output


def _decode_model(
    *,
    model_path: str,
    token_rows: list[dict[str, str]],
    batch_size: int,
    max_output_tokens: int,
    decoding_method: str,
    beam_size: int,
    device: str,
) -> tuple[list[str], list[list[int]]]:
    import torch

    root = Path(__file__).resolve().parents[2]
    transformer_dir = root / "transformer"
    sys.path.insert(0, str(transformer_dir))
    from transformer_functions import (  # type: ignore
        TransformerRegressor,
        decode_with_model,
        load_transformer_model,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_transformer_model(
        TransformerRegressor, model_path, device=device
    )
    model = loaded["model"]
    model.eval()
    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=8, max_sequence_length=4096
    )
    predictions: list[str] = []
    prediction_tokens: list[list[int]] = []
    for start in range(0, len(token_rows), batch_size):
        rows = token_rows[start : start + batch_size]
        inputs = [
            [2] + json.loads(row["scrambled"]) + [3]
            for row in rows
        ]
        length = max(map(len, inputs))
        padded = [sequence + [0] * (length - len(sequence)) for sequence in inputs]
        source = torch.tensor(padded, dtype=torch.long)
        decoded, _ = decode_with_model(
            model,
            source,
            max_length=max_output_tokens,
            decoding_method=decoding_method,
            beam_size=beam_size,
            bos_token=2,
            eos_token=3,
            pad_token=0,
        )
        for sequence in decoded.tolist():
            clean = _strip_special(sequence)
            prediction_tokens.append(clean)
            try:
                predictions.append(tokenizer.decode_infix(clean))
            except Exception:
                predictions.append("")
    return predictions, prediction_tokens


def _summarise(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    if not count:
        return {
            "count": 0,
            "exact_match": 0.0,
            "numerical_equivalence": 0.0,
            "mean_token_reduction": 0.0,
        }
    return {
        "count": count,
        "exact_match": sum(row["exact_match"] for row in rows) / count,
        "numerical_equivalence": sum(
            row["numerical_equivalence"] for row in rows
        )
        / count,
        "mean_token_reduction": sum(row["token_reduction"] for row in rows)
        / count,
    }


def evaluate(
    raw_rows: list[dict[str, str]],
    metadata_rows: list[dict[str, str]],
    predictions: list[str],
    prediction_tokens: list[list[int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    if not (len(raw_rows) == len(metadata_rows) == len(predictions)):
        raise ValueError("Raw, metadata, and prediction row counts do not agree")
    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=8, max_sequence_length=4096
    )
    details: list[dict[str, Any]] = []
    for index, (raw, meta, prediction) in enumerate(
        zip(raw_rows, metadata_rows, predictions, strict=True)
    ):
        process = meta["process"]
        target = raw["simple"]
        source = raw["scrambled"]
        exact = bool(prediction) and _normalise(prediction) == _normalise(target)
        if prediction:
            try:
                equivalent, error = numerically_equivalent(
                    prediction,
                    target,
                    process,
                    seeds=(7001 + index, 9001 + index),
                )
            except Exception:
                equivalent, error = False, float("inf")
        else:
            equivalent, error = False, float("inf")
        source_tokens = len(tokenizer.encode_infix(source))
        if prediction_tokens is not None:
            predicted_tokens = len(prediction_tokens[index])
        elif prediction:
            try:
                predicted_tokens = len(tokenizer.encode_infix(prediction))
            except ValueError:
                predicted_tokens = 0
        else:
            predicted_tokens = 0
        reduction = (
            1.0 - predicted_tokens / source_tokens
            if predicted_tokens and source_tokens
            else 0.0
        )
        details.append(
            {
                "row_id": index,
                "process": process,
                "scramble_depth": int(meta["scramble_depth"]),
                "target": target,
                "scrambled": source,
                "prediction": prediction,
                "exact_match": int(exact),
                "numerical_equivalence": int(equivalent),
                "relative_error": error,
                "source_tokens": source_tokens,
                "prediction_tokens": predicted_tokens,
                "token_reduction": reduction,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = details
    for row in details:
        groups[f"process={row['process']}"].append(row)
        groups[
            f"process={row['process']};depth={row['scramble_depth']}"
        ].append(row)
    return details, {name: _summarise(group) for name, group in groups.items()}


def _write_details(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", default="data/gravity/benchmarks_raw.csv.gz"
    )
    parser.add_argument(
        "--tokenized", default="data/gravity/benchmarks_tok.csv.gz"
    )
    parser.add_argument(
        "--metadata", default="data/gravity/benchmarks_metadata.csv.gz"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path")
    source.add_argument(
        "--predictions",
        help="CSV containing prediction or top1_prediction_expr",
    )
    parser.add_argument(
        "--output", default="data/gravity/benchmark_evaluation.csv.gz"
    )
    parser.add_argument(
        "--summary", default="data/gravity/benchmark_evaluation_summary.json"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--decoding-method", choices=("greedy", "beam"), default="beam"
    )
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_rows = _read(args.raw)
    metadata_rows = _read(args.metadata)
    prediction_tokens = None
    if args.model_path:
        predictions, prediction_tokens = _decode_model(
            model_path=args.model_path,
            token_rows=_read(args.tokenized),
            batch_size=args.batch_size,
            max_output_tokens=args.max_output_tokens,
            decoding_method=args.decoding_method,
            beam_size=args.beam_size,
            device=args.device,
        )
    else:
        rows = _read(args.predictions)
        if not rows:
            raise ValueError("Prediction CSV is empty")
        column = (
            "prediction"
            if "prediction" in rows[0]
            else "top1_prediction_expr"
        )
        predictions = [row.get(column, "") for row in rows]
    details, summary = evaluate(
        raw_rows, metadata_rows, predictions, prediction_tokens
    )
    _write_details(args.output, details)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
