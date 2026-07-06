#!/usr/bin/env python3
"""
Convert single-amplitude token CSVs into transformer simple/scrambled CSVs.

Input format:
    id,tokens
    1,"[4, 5, 6, ...]"

Output format:
    simple,scrambled
    "[4, 5, 6, ...]","[4, 5, 6, ...]"

The output is suitable for transformer.data_import.TransformerDataset, which
expects token lists in columns named simple and scrambled.
"""
import argparse
import csv
import gzip
import json
import pathlib
from typing import TextIO


def _open_text(path: pathlib.Path, mode: str) -> TextIO:
    text_mode = mode if "t" in mode else f"{mode}t"
    if path.suffix == ".gz":
        return gzip.open(path, text_mode, newline="", encoding="utf-8")
    return path.open(text_mode, newline="", encoding="utf-8")


def _normalise_token_list(value: str, *, row_number: int, column_name: str) -> str:
    try:
        tokens = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Row {row_number} column '{column_name}' is not valid JSON: {exc}"
        ) from exc

    if not isinstance(tokens, list) or not all(isinstance(tok, int) for tok in tokens):
        raise ValueError(
            f"Row {row_number} column '{column_name}' must be a JSON list of ints"
        )

    return json.dumps(tokens)


def convert_file(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    tokens_column: str = "tokens",
) -> int:
    with _open_text(src, "r") as fin, _open_text(dst, "w") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty or has no header: {src}")
        if tokens_column not in reader.fieldnames:
            raise ValueError(
                f"Input CSV must contain a '{tokens_column}' column; "
                f"got {reader.fieldnames}"
            )

        writer = csv.DictWriter(fout, fieldnames=["simple", "scrambled"])
        writer.writeheader()

        rows_written = 0
        for row_number, row in enumerate(reader, start=2):
            token_text = _normalise_token_list(
                row[tokens_column],
                row_number=row_number,
                column_name=tokens_column,
            )
            writer.writerow({"simple": token_text, "scrambled": token_text})
            rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an id,tokens single-amplitude CSV into a simple,scrambled "
            "token CSV for transformer evaluation."
        )
    )
    parser.add_argument("input_csv", type=pathlib.Path)
    parser.add_argument("output_csv", type=pathlib.Path)
    parser.add_argument(
        "--tokens-column",
        default="tokens",
        help="Name of the input column containing the token list.",
    )
    args = parser.parse_args()

    rows_written = convert_file(
        args.input_csv,
        args.output_csv,
        tokens_column=args.tokens_column,
    )
    print(f"Wrote {rows_written} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
