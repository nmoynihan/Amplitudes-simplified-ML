#!/usr/bin/env python3
"""
tokenize_data.py - Tokenise CSV files of amplitude expressions.

For the usual training-data format, reads a CSV with columns (simple,
scrambled) and writes a new CSV where each cell contains the JSON-encoded list
of integer token IDs.

For Feynman-amplitude CSVs such as gluon4feyn.csv, pass --input-format feyn.
Those files are headerless rows like "1,<amplitude>"; the output columns are
(id, tokens).

Usage:
    python tokenize_data.py input.csv output.csv [--max-particles 8]
    python tokenize_data.py tokenised.csv decoded.csv --decode-format infix
    python tokenize_data.py gluon4feyn.csv gluon4feyn_tok.csv --input-format feyn
"""
import argparse
import csv
import gzip
import itertools
import json
import pathlib
from typing import Optional, TextIO

from Tokenizer import ScatteringAmplitudeTokenizer


PAIR_COLUMNS = {"simple", "scrambled"}


def _resolve_input_path(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        return path

    gz_path = pathlib.Path(f"{path}.gz")
    if gz_path.exists():
        return gz_path

    return path


def _open_text(path: pathlib.Path, mode: str) -> TextIO:
    text_mode = mode if "t" in mode else f"{mode}t"
    if path.suffix == ".gz":
        return gzip.open(path, text_mode, newline="", encoding="utf-8")
    return path.open(text_mode, newline="", encoding="utf-8")


def _is_pair_header(row: list[str]) -> bool:
    return set(row) == PAIR_COLUMNS


def _selected(row: list[str], index: int, *, column_name: str, row_number: int) -> str:
    try:
        return row[index].strip()
    except IndexError as exc:
        raise ValueError(
            f"Row {row_number} has no {column_name} at column index {index}: {row}"
        ) from exc


def tokenise_file(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    max_particles: int = 8,
    decode_format: Optional[str] = None,
    input_format: str = "auto",
    expression_column: int = 1,
    id_column: int = 0,
) -> None:
    src = _resolve_input_path(src)
    tok = ScatteringAmplitudeTokenizer(max_particles=max_particles)

    with _open_text(src, "r") as fin, _open_text(dst, "w") as fout:
        reader = csv.reader(fin)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"Input CSV is empty: {src}")

        if input_format == "auto":
            resolved_format = "pairs" if _is_pair_header(first_row) else "feyn"
        else:
            resolved_format = input_format

        if resolved_format == "pairs":
            if not _is_pair_header(first_row):
                raise ValueError(
                    f"Pair CSV must have columns {PAIR_COLUMNS}, got {first_row}"
                )
            _tokenise_pair_rows(
                reader,
                fout,
                tok,
                fieldnames=first_row,
                decode_format=decode_format,
            )
        elif resolved_format == "feyn":
            if decode_format is not None:
                raise ValueError("--decode-format is only supported for pairs CSVs")
            _tokenise_feyn_rows(
                itertools.chain([first_row], reader),
                fout,
                tok,
                expression_column=expression_column,
                id_column=id_column,
            )
        else:
            raise ValueError(f"Unsupported input format: {input_format}")


def _tokenise_pair_rows(
    rows,
    fout: TextIO,
    tok: ScatteringAmplitudeTokenizer,
    *,
    fieldnames: list[str],
    decode_format: Optional[str],
) -> None:
    writer = csv.DictWriter(fout, fieldnames=["simple", "scrambled"])
    writer.writeheader()

    simple_idx = fieldnames.index("simple")
    scrambled_idx = fieldnames.index("scrambled")
    skipped = 0
    for row_number, row in enumerate(rows, start=2):
        simple = _selected(
            row,
            simple_idx,
            column_name="simple",
            row_number=row_number,
        )
        scrambled = _selected(
            row,
            scrambled_idx,
            column_name="scrambled",
            row_number=row_number,
        )
        if decode_format is None:
            # Encode: infix -> token IDs
            writer.writerow({
                "simple": json.dumps(tok.encode_infix(simple)),
                "scrambled": json.dumps(tok.encode_infix(scrambled)),
            })
        else:
            # Decode: token IDs -> infix or prefix
            try:
                s_ids = json.loads(simple)
                t_ids = json.loads(scrambled)
                if decode_format == "infix":
                    s_str = tok.decode_infix(s_ids)
                    t_str = tok.decode_infix(t_ids)
                elif decode_format == "prefix":
                    s_str = tok.decode_prefix(s_ids)
                    t_str = tok.decode_prefix(t_ids)
                else:
                    raise ValueError(f"Unsupported decode format: {decode_format}")
            except ValueError:
                skipped += 1
                continue
            writer.writerow({"simple": s_str, "scrambled": t_str})

    if skipped:
        print(f"Skipped {skipped} rows due to decode errors.")


def _tokenise_feyn_rows(
    rows,
    fout: TextIO,
    tok: ScatteringAmplitudeTokenizer,
    *,
    expression_column: int,
    id_column: int,
) -> None:
    writer = csv.DictWriter(fout, fieldnames=["id", "tokens"])
    writer.writeheader()

    for row_number, row in enumerate(rows, start=1):
        if not row:
            continue
        row_id = _selected(row, id_column, column_name="id", row_number=row_number)
        expr = _selected(
            row,
            expression_column,
            column_name="expression",
            row_number=row_number,
        )
        writer.writerow({
            "id": row_id,
            "tokens": json.dumps(tok.encode_infix(expr)),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenise amplitude expressions in a CSV.")
    parser.add_argument("input_csv", type=pathlib.Path)
    parser.add_argument("output_csv", type=pathlib.Path)
    parser.add_argument("--max-particles", type=int, default=8)
    parser.add_argument(
        "--input-format",
        choices=["auto", "pairs", "feyn"],
        default="auto",
        help=(
            "CSV layout. 'pairs' expects simple,scrambled columns; 'feyn' expects "
            "headerless rows with an id column and one amplitude column."
        ),
    )
    parser.add_argument(
        "--expression-column",
        type=int,
        default=1,
        help="Zero-based expression column index for --input-format feyn.",
    )
    parser.add_argument(
        "--id-column",
        type=int,
        default=0,
        help="Zero-based id column index for --input-format feyn.",
    )
    parser.add_argument("--decode-format", type=str, default=None,
                        choices=["infix", "prefix"],
                        help="Decode token IDs back to expressions.")
    args = parser.parse_args()
    tokenise_file(args.input_csv, args.output_csv,
                  max_particles=args.max_particles,
                  decode_format=args.decode_format,
                  input_format=args.input_format,
                  expression_column=args.expression_column,
                  id_column=args.id_column)
