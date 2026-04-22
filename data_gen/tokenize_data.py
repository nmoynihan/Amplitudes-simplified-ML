#!/usr/bin/env python3
"""
tokenize_data.py — Tokenise a CSV of amplitude expressions.

Reads a CSV with columns (simple, scrambled) and writes a new CSV where each
cell contains the JSON-encoded list of integer token IDs.

Usage:
    python tokenize_data.py input.csv output.csv [--max-particles 8]
    python tokenize_data.py tokenised.csv decoded.csv --decode-format infix
"""
import argparse
import csv
import json
import pathlib

from Tokenizer import ScatteringAmplitudeTokenizer


def tokenise_file(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    max_particles: int = 8,
    decode_format: str | None = None,
) -> None:
    tok = ScatteringAmplitudeTokenizer(max_particles=max_particles)

    with src.open(newline="", encoding="utf-8") as fin, \
         dst.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        expected = {"simple", "scrambled"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                f"CSV must have columns {expected}, got {reader.fieldnames}"
            )

        writer = csv.DictWriter(fout, fieldnames=["simple", "scrambled"])
        writer.writeheader()

        skipped = 0
        for row in reader:
            if decode_format is None:
                # Encode: infix → token IDs
                writer.writerow({
                    "simple": json.dumps(tok.encode_infix(row["simple"])),
                    "scrambled": json.dumps(tok.encode_infix(row["scrambled"])),
                })
            else:
                # Decode: token IDs → infix or prefix
                try:
                    s_ids = json.loads(row["simple"])
                    t_ids = json.loads(row["scrambled"])
                    if decode_format == "infix":
                        s_str = tok.decode_infix(s_ids)
                        t_str = tok.decode_infix(t_ids)
                    elif decode_format == "prefix":
                        s_str = tok.decode_prefix(s_ids)
                        t_str = tok.decode_prefix(t_ids)
                    else:
                        raise ValueError(f"Unsupported decode format: {decode_format}")
                except (ValueError, KeyError) as e:
                    skipped += 1
                    continue
                writer.writerow({"simple": s_str, "scrambled": t_str})

        if skipped:
            print(f"Skipped {skipped} rows due to decode errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenise amplitude expressions in a CSV.")
    parser.add_argument("input_csv", type=pathlib.Path)
    parser.add_argument("output_csv", type=pathlib.Path)
    parser.add_argument("--max-particles", type=int, default=8)
    parser.add_argument("--decode-format", type=str, default=None,
                        choices=["infix", "prefix"],
                        help="Decode token IDs back to expressions.")
    args = parser.parse_args()
    tokenise_file(args.input_csv, args.output_csv,
                  max_particles=args.max_particles,
                  decode_format=args.decode_format)
