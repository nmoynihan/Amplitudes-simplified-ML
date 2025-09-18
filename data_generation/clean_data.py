#!/usr/bin/env python3
"""
CSV Deduplication Utility

Usage examples (PowerShell):
  # Keep first occurrence of each duplicate row across all columns
  python data_generation/clean_data.py path/to/input.csv

  # Deduplicate by a subset of columns, keeping the last occurrence
  python data_generation/clean_data.py path/to/input.csv -s id -s date --keep last

  # Case-insensitive and trim whitespace before comparing
  python data_generation/clean_data.py input.csv -s email --case-insensitive --strip-whitespace

  # In-place replacement (writes to a temporary file then replaces the original)
  python data_generation/clean_data.py input.csv --inplace

By default, writes to <input>.deduped.csv in the same folder and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _sniff_dialect(sample: bytes) -> Any:
	try:
		return csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"))
	except Exception:
		return csv.get_dialect("excel")


def _open_csv_reader(path: Path) -> Tuple[Any, List[str], Iterable[List[str]]]:
	"""Open a CSV file and return (dialect, headers, rows_iterable).

	Ensures UTF-8-sig is supported for BOM. Detects dialect from a sample.
	"""
	f = path.open("r", encoding="utf-8-sig", newline="")
	try:
		# Read a small sample to sniff dialect
		sample = f.read(8192)
		f.seek(0)
		dialect: Any = _sniff_dialect(sample.encode("utf-8", errors="ignore")) if sample else csv.get_dialect("excel")
		reader: Any = csv.reader(f, dialect)  # type: ignore[arg-type]
		try:
			headers = next(reader)
		except StopIteration:
			headers = []
			rows_iter = iter(())
		else:
			rows_iter = reader
		return dialect, headers, rows_iter
	finally:
		# Do not close file here; reader needs the handle. It will be closed when writer finishes.
		# We intentionally keep f open via the reader iterator.
		pass


def _open_csv_writer(path: Path, dialect: Any) -> Tuple[io.TextIOBase, Any]:
	f = path.open("w", encoding="utf-8", newline="")
	writer: Any = csv.writer(f, dialect)  # type: ignore[arg-type]
	return f, writer


def _read_headers_only(path: Path) -> Tuple[Any, List[str]]:
	"""Read only the header row and return (dialect, headers)."""
	with path.open("r", encoding="utf-8-sig", newline="") as f:
		sample = f.read(8192)
		f.seek(0)
		dialect: Any = _sniff_dialect(sample.encode("utf-8", errors="ignore")) if sample else csv.get_dialect("excel")
		reader: Any = csv.reader(f, dialect)  # type: ignore[arg-type]
		try:
			headers = next(reader)
		except StopIteration:
			headers = []
		return dialect, headers


def _normalize_value(v: str, case_insensitive: bool, strip_ws: bool) -> str:
	if strip_ws:
		v = v.strip()
	if case_insensitive:
		v = v.lower()
	return v


def _key_from_row(
	row: Sequence[str],
	headers: Sequence[str],
	subset_cols: Optional[Sequence[str]],
	case_insensitive: bool,
	strip_ws: bool,
) -> Tuple[str, ...]:
	if subset_cols:
		indices = [headers.index(c) for c in subset_cols]
		vals = [row[i] if i < len(row) else "" for i in indices]
	else:
		vals = list(row)
	return tuple(_normalize_value(v, case_insensitive, strip_ws) for v in vals)


def _validate_subset(headers: Sequence[str], subset_cols: Optional[Sequence[str]]):
	if not subset_cols:
		return
	missing = [c for c in subset_cols if c not in headers]
	if missing:
		raise SystemExit(f"Subset columns not found in CSV header: {missing}\nAvailable columns: {headers}")


def dedupe_keep_first(
	in_path: Path,
	out_path: Path,
	subset_cols: Optional[Sequence[str]],
	case_insensitive: bool,
	strip_ws: bool,
) -> Tuple[int, int]:
	"""Stream the input and write first occurrence of each key.

	Returns (rows_including_header, rows_written_including_header)
	"""
	dialect, headers, rows_iter = _open_csv_reader(in_path)
	if not headers:
		# empty file
		wf, _writer = _open_csv_writer(out_path, dialect)
		wf.close()
		return 0, 0

	_validate_subset(headers, subset_cols)

	seen = set()
	written = 0
	total = 1  # count header

	wf, writer = _open_csv_writer(out_path, dialect)
	try:
		writer.writerow(headers)
		written += 1
		for row in rows_iter:
			total += 1
			key = _key_from_row(row, headers, subset_cols, case_insensitive, strip_ws)
			if key in seen:
				continue
			seen.add(key)
			writer.writerow(row)
			written += 1
	finally:
		wf.close()

	return total, written


def dedupe_keep_last(
	in_path: Path,
	out_path: Path,
	subset_cols: Optional[Sequence[str]],
	case_insensitive: bool,
	strip_ws: bool,
) -> Tuple[int, int]:
	"""Two-pass approach to keep the last occurrence of each key.

	Returns (rows_including_header, rows_written_including_header)
	"""
	dialect, headers, rows_iter = _open_csv_reader(in_path)
	if not headers:
		wf, _writer = _open_csv_writer(out_path, dialect)
		wf.close()
		return 0, 0

	_validate_subset(headers, subset_cols)

	# First pass: record the last index for each key
	index = 0  # data-row index starting at 0 for the first data row
	last_index: Dict[Tuple[str, ...], int] = {}
	data_rows: List[List[str]] = []  # we'll reuse for second pass without rereading file twice
	for row in rows_iter:
		key = _key_from_row(row, headers, subset_cols, case_insensitive, strip_ws)
		last_index[key] = index
		data_rows.append(row)
		index += 1

	# Second pass: write only rows whose index is the last occurrence
	wf, writer = _open_csv_writer(out_path, dialect)
	written = 0
	try:
		writer.writerow(headers)
		written += 1
		for idx, row in enumerate(data_rows):
			key = _key_from_row(row, headers, subset_cols, case_insensitive, strip_ws)
			if last_index.get(key) == idx:
				writer.writerow(row)
				written += 1
	finally:
		wf.close()

	total = 1 + len(data_rows)
	return total, written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Deduplicate a CSV file by all or selected columns.")
	p.add_argument("input", type=str, help="Path to input CSV")
	p.add_argument("-o", "--output", type=str, help="Path to write deduplicated CSV. Defaults to <input>.deduped.csv")
	p.add_argument(
		"-s",
		"--subset",
		action="append",
		help=(
			"Column name to use for duplicate detection (may be provided multiple times). "
			"You can also pass a comma-separated list, e.g. -s colA,colB"
		),
	)
	p.add_argument("--keep", choices=["first", "last"], default="first", help="Keep the first or last occurrence for duplicates")
	p.add_argument("--inplace", action="store_true", help="Replace the input file in place (atomic via temp file)")
	# Defaults: case-insensitive, strip-whitespace
	p.add_argument("--case-insensitive", action="store_true", default=True, help="Compare values case-insensitively (default: on)")
	p.add_argument("--strip-whitespace", action="store_true", default=True, help="Strip leading/trailing whitespace before comparing (default: on)")
	return p.parse_args(argv)


def _flatten_subset_arg(subset_arg: Optional[Sequence[str]]) -> Optional[List[str]]:
	if not subset_arg:
		return None
	cols: List[str] = []
	for s in subset_arg:
		if s is None:
			continue
		cols.extend([c for c in (x.strip() for x in s.split(",")) if c])
	return cols or None


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)

	in_path = Path(args.input)
	if not in_path.exists():
		print(f"Input file not found: {in_path}", file=sys.stderr)
		return 2

	subset_cols = _flatten_subset_arg(args.subset)
	# Default subset columns:
	# 1) scambled,unscrambled if both present
	# 2) simple,scrambled if both present
	# 3) otherwise, all columns
	if subset_cols is None:
		_dialect, headers = _read_headers_only(in_path)
		# Respect exact names, as requested
		if all(c in headers for c in ("scambled", "unscrambled")):
			subset_cols = ["scambled", "unscrambled"]
		elif all(c in headers for c in ("simple", "scrambled")):
			subset_cols = ["simple", "scrambled"]
		else:
			subset_cols = None  # fall back to all columns

	if args.output and args.inplace:
		print("Cannot use --output and --inplace together.", file=sys.stderr)
		return 2

	if args.inplace:
		out_path = in_path.with_suffix(in_path.suffix + ".tmp_dedup")
	else:
		out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + ".deduped" + in_path.suffix)

	try:
		if args.keep == "first":
			total, written = dedupe_keep_first(
				in_path,
				out_path,
				subset_cols=subset_cols,
				case_insensitive=args.__dict__["case_insensitive"],
				strip_ws=args.__dict__["strip_whitespace"],
			)
		else:
			total, written = dedupe_keep_last(
				in_path,
				out_path,
				subset_cols=subset_cols,
				case_insensitive=args.__dict__["case_insensitive"],
				strip_ws=args.__dict__["strip_whitespace"],
			)

		if args.inplace:
			# Atomic-ish replace
			backup = in_path.with_suffix(in_path.suffix + ".bak")
			try:
				if backup.exists():
					backup.unlink()
				in_path.replace(backup)
				out_path.replace(in_path)
				# Cleanup backup if everything is fine
				backup.unlink(missing_ok=True)  # type: ignore[call-arg]
			except Exception:
				# Try to restore from backup on failure
				if not in_path.exists() and backup.exists():
					backup.replace(in_path)
				raise

		dupes_removed = max(0, total - written)
		cols_info = ", ".join(subset_cols) if subset_cols else "<ALL COLUMNS>"
		print(
			"\nDeduplication summary:\n"
			f"  Input:  {in_path}\n"
			f"  Output: {in_path if args.inplace else out_path}\n"
			f"  Columns used: {cols_info}\n"
			f"  Keep: {args.keep}\n"
			f"  Rows (incl header): {total} -> {written} (removed {dupes_removed})\n"
		)
		return 0
	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	sys.exit(main())


def dedupe_csv_file(
	path: str | Path,
	subset: Optional[Sequence[str]] = None,
	keep: str = "first",
	case_insensitive: bool = True,
	strip_whitespace: bool = True,
) -> Dict[str, Any]:
	"""
	Programmatic API: deduplicate a CSV file in-place.

	- path: path to the CSV file
	- subset: columns to use as the dedupe key; if None, uses defaults:
		['scambled','unscrambled'] if both present,
		else ['simple','scrambled'] if both present,
		else all columns.
	- keep: 'first' or 'last'
	- case_insensitive: normalize to lowercase before comparing (default True)
	- strip_whitespace: strip leading/trailing whitespace before comparing (default True)

	Returns a summary dict:
	  { 'input': str, 'rows_total': int, 'rows_written': int, 'removed': int, 'columns_used': List[str] | '<ALL COLUMNS>', 'keep': str }
	"""
	in_path = Path(path)
	if not in_path.exists():
		raise FileNotFoundError(f"Input file not found: {in_path}")

	# Determine subset if not provided
	subset_cols = list(subset) if subset is not None else None
	if subset_cols is None:
		_dialect, headers = _read_headers_only(in_path)
		if all(c in headers for c in ("scambled", "unscrambled")):
			subset_cols = ["scambled", "unscrambled"]
		elif all(c in headers for c in ("simple", "scrambled")):
			subset_cols = ["simple", "scrambled"]
		else:
			subset_cols = None

	# Perform dedupe to a temp file, then replace
	tmp_out = in_path.with_suffix(in_path.suffix + ".tmp_dedup")
	if keep not in ("first", "last"):
		raise ValueError("keep must be 'first' or 'last'")

	if keep == "first":
		total, written = dedupe_keep_first(
			in_path,
			tmp_out,
			subset_cols=subset_cols,
			case_insensitive=case_insensitive,
			strip_ws=strip_whitespace,
		)
	else:
		total, written = dedupe_keep_last(
			in_path,
			tmp_out,
			subset_cols=subset_cols,
			case_insensitive=case_insensitive,
			strip_ws=strip_whitespace,
		)

	backup = in_path.with_suffix(in_path.suffix + ".bak")
	try:
		if backup.exists():
			backup.unlink()
		in_path.replace(backup)
		tmp_out.replace(in_path)
		# Best-effort cleanup
		try:
			backup.unlink()
		except Exception:
			pass
	except Exception:
		# Try restore
		if not in_path.exists() and backup.exists():
			backup.replace(in_path)
		raise

	removed = max(0, total - written)
	return {
		"input": str(in_path),
		"rows_total": total,
		"rows_written": written,
		"removed": removed,
		"columns_used": subset_cols if subset_cols is not None else "<ALL COLUMNS>",
		"keep": keep,
	}

