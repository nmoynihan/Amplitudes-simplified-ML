#!/usr/bin/env python3
"""
clean_data.py — Deduplicate a CSV file by all or selected columns.

Usage:
    python clean_data.py input.csv
    python clean_data.py input.csv -o output.csv --keep last
    python clean_data.py input.csv --inplace
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read CSV returning (headers, rows)."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return [], []
        return headers, list(reader)


def _make_key(
    row: Sequence[str],
    headers: Sequence[str],
    subset: Optional[Sequence[str]],
    case_insensitive: bool,
    strip_ws: bool,
) -> Tuple[str, ...]:
    if subset:
        indices = [headers.index(c) for c in subset]
        vals = [row[i] if i < len(row) else "" for i in indices]
    else:
        vals = list(row)
    out: list[str] = []
    for v in vals:
        if strip_ws:
            v = v.strip()
        if case_insensitive:
            v = v.lower()
        out.append(v)
    return tuple(out)


def dedupe(
    in_path: Path,
    out_path: Path,
    *,
    subset: Optional[Sequence[str]] = None,
    keep: str = "first",
    case_insensitive: bool = True,
    strip_ws: bool = True,
) -> Tuple[int, int]:
    """Deduplicate a CSV.  Returns (total_rows_incl_header, written_rows_incl_header)."""
    headers, rows = _read_csv(in_path)
    if not headers:
        out_path.write_text("")
        return 0, 0

    if subset:
        missing = [c for c in subset if c not in headers]
        if missing:
            raise SystemExit(f"Columns not found: {missing}\nAvailable: {headers}")

    def key(row):
        return _make_key(row, headers, subset, case_insensitive, strip_ws)

    if keep == "last":
        last_idx = {key(row): i for i, row in enumerate(rows)}
        kept = [row for i, row in enumerate(rows) if last_idx[key(row)] == i]
    else:
        seen: set[Tuple[str, ...]] = set()
        kept: list[list[str]] = []
        for row in rows:
            k = key(row)
            if k not in seen:
                seen.add(k)
                kept.append(row)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(kept)

    total = 1 + len(rows)
    written = 1 + len(kept)
    return total, written


def _infer_subset(headers: Sequence[str]) -> Optional[List[str]]:
    """Auto-detect subset columns from common header patterns."""
    if all(c in headers for c in ("simple", "scrambled")):
        return ["simple", "scrambled"]
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Deduplicate a CSV file.")
    p.add_argument("input", type=str)
    p.add_argument("-o", "--output", type=str)
    p.add_argument("-s", "--subset", action="append",
                   help="Column(s) for duplicate detection (repeatable, or comma-separated).")
    p.add_argument("--keep", choices=["first", "last"], default="first")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--case-insensitive", action="store_true", default=True)
    p.add_argument("--strip-whitespace", action="store_true", default=True)
    args = p.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"File not found: {in_path}", file=sys.stderr)
        return 2

    # Flatten subset arg
    subset: Optional[List[str]] = None
    if args.subset:
        subset = []
        for s in args.subset:
            subset.extend(c.strip() for c in s.split(",") if c.strip())
    if subset is None:
        headers, _ = _read_csv(in_path)
        subset = _infer_subset(headers)

    if args.inplace:
        out_path = in_path.with_suffix(in_path.suffix + ".tmp")
    else:
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + ".deduped" + in_path.suffix)

    try:
        total, written = dedupe(
            in_path, out_path,
            subset=subset, keep=args.keep,
            case_insensitive=args.case_insensitive,
            strip_ws=args.strip_whitespace,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.inplace:
        out_path.replace(in_path)

    removed = max(0, total - written)
    cols_str = ", ".join(subset) if subset else "<ALL>"
    print(f"Deduplication: {in_path}")
    print(f"  Columns: {cols_str}  Keep: {args.keep}")
    print(f"  Rows: {total} → {written}  (removed {removed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
