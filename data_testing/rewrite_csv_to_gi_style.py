#!/usr/bin/env python3
"""
Rewrite a raw simple/scrambled CSV into the GI-generator scrambled style.

This does not infer structure from the original scrambled column. It uses the
compact simple column as the source of truth, expands it with gen_data's F-block
rules, optionally applies the same scramble passes as generated data, and writes
a new raw CSV with columns simple,scrambled.

Example:
    python data_testing/rewrite_csv_to_gi_style.py \
        data/sqed_w0110_mM00M_den6_maxD2_20000_phys_oneshot.csv \
        data/sqed_w0110_mM00M_den6_maxD2_20000_phys_oneshot_gi_style_no_powers.csv \
        --skip-powers --min-scrambles 0 --max-scrambles 0
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_gen"))

import gen_data as gd


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def infer_n(expr: str) -> int:
    import re

    indices = [int(x) for x in re.findall(r"[pPeEfF]_(\d+)", expr)]
    if not indices:
        raise ValueError("Could not infer particle count from expression")
    return max(indices)


def rewrite_csv(
    input_csv: Path,
    output_csv: Path,
    *,
    max_rows: int | None,
    skip_powers: bool,
    min_scrambles: int,
    max_scrambles: int,
    seed: int,
    validate: bool,
    mass: float,
) -> None:
    random.seed(seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    rows_written = 0
    rows_skipped_power = 0
    rows_failed = 0

    with input_csv.open(newline="", encoding="utf-8") as fin, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        required = {"simple", "scrambled"}
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} has no CSV header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

        writer = csv.DictWriter(fout, fieldnames=["simple", "scrambled"])
        writer.writeheader()

        for row_idx, row in enumerate(reader, start=2):
            if max_rows is not None and rows_written >= max_rows:
                break
            rows_read += 1
            simple = (row.get("simple") or "").strip()
            if not simple:
                rows_failed += 1
                print(f"Skipping {input_csv}:{row_idx}: empty simple expression")
                continue

            if skip_powers and "^" in simple:
                rows_skipped_power += 1
                continue

            try:
                n_particles = infer_n(simple)
                expanded = gd.expand_simple_expression(simple).replace("**", "^")
                scrambled = gd.scramble(
                    expanded,
                    n_particles - 2,
                    n_particles,
                    min_scr=min_scrambles,
                    max_scr=max_scrambles,
                )
                if validate:
                    ok, reason = gd._validate_pair(simple, scrambled, n_particles, mass)
                    if not ok:
                        rows_failed += 1
                        print(f"Skipping {input_csv}:{row_idx}: validation failed: {reason}")
                        continue
            except Exception as exc:
                rows_failed += 1
                print(f"Skipping {input_csv}:{row_idx}: {type(exc).__name__}: {exc}")
                continue

            writer.writerow({"simple": simple, "scrambled": scrambled})
            rows_written += 1

    print(f"Read rows           : {rows_read}")
    print(f"Wrote rows          : {rows_written}")
    print(f"Skipped power rows  : {rows_skipped_power}")
    print(f"Failed rows         : {rows_failed}")
    print(f"Wrote GI-style CSV  : {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--skip-powers",
        action="store_true",
        help="Skip rows whose simple expression contains ^. Useful because the GI training data has no powers.",
    )
    parser.add_argument("--min-scrambles", type=int, default=0)
    parser.add_argument("--max-scrambles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--mass", type=float, default=2.0)
    args = parser.parse_args()

    rewrite_csv(
        resolve_path(args.input_csv),
        resolve_path(args.output_csv),
        max_rows=args.max_rows,
        skip_powers=args.skip_powers,
        min_scrambles=args.min_scrambles,
        max_scrambles=args.max_scrambles,
        seed=args.seed,
        validate=args.validate,
        mass=args.mass,
    )


if __name__ == "__main__":
    main()
