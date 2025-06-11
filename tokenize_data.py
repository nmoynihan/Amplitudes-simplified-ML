#!/usr/bin/env python3
"""
tokenise_csv.py
---------------

Read a CSV whose rows look like

    simple,scrambled
    13p_2 · p_2,Tr(F_1 · F_2) + p_3 · e_2
    …

tokenise both amplitude expressions with your ScatteringAmplitudeTokenizer
(from *Tokenizer.py*) and write a **new** CSV in which each cell contains the
space-separated integer IDs that the tokenizer returned.

Usage
-----
    python tokenise_csv.py input.csv output.csv [--max-particles 8]

`--max-particles` lets you keep the script in step with any future change in the
vocabulary size you decide to make.
"""

import csv
import argparse
import pathlib
import json
from Tokenizer import ScatteringAmplitudeTokenizer


def tokenise_file(src: pathlib.Path,
                  dst: pathlib.Path,
                  *,
                  max_particles: int = 8,
                  decode_format: str = None) -> None:
    """Tokenise every row of *src* and write the result to *dst*."""
    tok = ScatteringAmplitudeTokenizer(max_particles=max_particles)

    with src.open(newline="", encoding="utf-8") as fin, \
         dst.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        # We insist on the canonical two-column layout
        expected_cols = {"simple", "scrambled"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected_cols:
            raise ValueError(f"Input CSV must have exactly these columns: {expected_cols}, instead got: {reader.fieldnames}")

        writer = csv.DictWriter(fout, fieldnames=["simple", "scrambled"])
        writer.writeheader()

        for row in reader:
            if decode_format is None:
                simple_vec = tok.encode_infix(row["simple"])
                scrambled_vec = tok.encode_infix(row["scrambled"])
                writer.writerow({
                    "simple": json.dumps(simple_vec),
                    "scrambled": json.dumps(scrambled_vec),
                })
            else:
                simple_ids = json.loads(row["simple"])
                scrambled_ids = json.loads(row["scrambled"])

                if reader.line_num == 300:
                    print("Row", reader.line_num, ":", row)
                    print("Simple tokenized", simple_ids)
                    print("Simple infix", tok.decode_infix(simple_ids))
                    print("Simple prefix", tok.decode_prefix(simple_ids))
                    print("Scrambled tokenized", scrambled_ids)
                    #print("Scrambled infix", tok.decode_infix(scrambled_ids))
                    print("Scrambled prefix", tok.decode_prefix(scrambled_ids))

                try:
                    if decode_format == "infix":
                        row_simple = tok.decode_infix(simple_ids)
                        row_scrambled = tok.decode_infix(scrambled_ids)
                    elif decode_format == "prefix":
                        row_simple = tok.decode_prefix(simple_ids)
                        row_scrambled = tok.decode_prefix(scrambled_ids)
                    else:
                        raise ValueError("Unsupported decode format.")
                except ValueError as e:
                    print(f"Skipping row due to decode error: {e}")
                    continue

                writer.writerow({
                    "simple": row_simple,
                    "scrambled": row_scrambled,
                })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenise amplitude expressions held in a CSV file."
    )
    parser.add_argument("input_csv",  type=pathlib.Path, help="Path to the input CSV.")
    parser.add_argument("output_csv", type=pathlib.Path, help="Where to write the tokenised CSV.")
    parser.add_argument("--max-particles", type=int, default=8,
                        help="Size of the p_i / e_i / F_i families (default: 8).")
    parser.add_argument("--decode-format", type=str, default=None,
                        help="Optional: convert token IDs back to 'infix' or 'prefix'.")

    args = parser.parse_args()
    tokenise_file(args.input_csv, args.output_csv,
                  max_particles=args.max_particles,
                  decode_format=args.decode_format)