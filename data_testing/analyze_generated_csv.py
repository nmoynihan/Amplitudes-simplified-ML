#!/usr/bin/env python3
"""
Analyze a raw generated amplitude CSV.

Expected input format:
    CSV with columns: simple, scrambled

Outputs:
1. Token-length statistics for the whole dataset.
2. Warnings for:
   - explicit mass terms M, M^n, M_i
   - manifest mass-dimension mismatches
   - spurious / unsupported pole structure
   - missing photon legs (all legs 2..N-1 should appear)

Usage:
    python data_testing/analyze_generated_csv.py path/to/file.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_gen"))

import gen_data as gd
from Tokenizer import ScatteringAmplitudeTokenizer
from kinematics import generate_kinematics


INDEX_RE = re.compile(r"[pPeEfF]_(\d+)")
MASS_RE = re.compile(r"\bM(?:_\d+)?(?:\s*\^\s*\d+)?\b")


@dataclass
class WarningEntry:
    row_id: int
    csv_line: int
    column: str
    issue: str
    details: str
    expression: str


def split_top_level(expr: str, sep: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def split_top_level_sum(expr: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and ch in "+-" and i > 0:
            prev = expr[i - 1]
            if prev not in "*/^(+":
                parts.append("".join(current).strip())
                current = [ch]
                continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def strip_outer_parens(s: str) -> str:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return s
    return s[1:-1].strip()


def infer_n(*exprs: str) -> int | None:
    used = [int(m.group(1)) for expr in exprs for m in INDEX_RE.finditer(expr)]
    return max(used) if used else None


def photon_legs_present(simple: str, scrambled: str, n: int) -> set[int]:
    present: set[int] = set()
    for expr in (simple, scrambled):
        for pat in (r"F_(\d+)", r"e_(\d+)"):
            for match in re.finditer(pat, expr):
                idx = int(match.group(1))
                if 2 <= idx <= n - 1:
                    present.add(idx)
    return present


def find_mass_terms(expr: str) -> list[str]:
    return sorted(set(m.group(0) for m in MASS_RE.finditer(expr)))


def check_spurious_poles(simple_expr: str, n: int) -> list[str]:
    issues: list[str] = []
    for term in split_top_level_sum(simple_expr):
        expr = term.lstrip("+-").strip()
        if "/" not in expr:
            continue
        num, den = expr.split("/", 1)
        num = strip_outer_parens(num)
        den = strip_outer_parens(den)
        den_factors = [f.strip() for f in split_top_level(den, "*") if f.strip()]
        canonical_denom = {gd._canon_pp(f) if gd._RE_pp.fullmatch(f) else f for f in den_factors}

        for factor in den_factors:
            match = gd._RE_pp.fullmatch(factor)
            if not match:
                issues.append(f"non-p·p denominator factor: {factor}")
                continue
            i, j = sorted(map(int, match.groups()))
            if {i, j} == {1, n} or (i in {1, n} and j in {1, n}):
                issues.append(f"scalar-scalar denominator factor: {factor}")

        num_factors = [f.strip() for f in split_top_level(num, "*") if f.strip()]
        for factor in num_factors:
            match = gd._RE_pFchainp.fullmatch(factor)
            if not match:
                continue
            photons = [int(x) for x in re.findall(r"F_(\d+)", match.group(2))]
            for photon in photons:
                if not any(re.search(fr"\bp_{photon}\b", d) for d in canonical_denom):
                    issues.append(f"photon leg {photon} appears in p·F...·p block without denominator support")
    return sorted(set(issues))


def numerically_compare(
    simple: str,
    scrambled: str,
    kinematics_samples: list[tuple[object, object]],
    *,
    tol_abs: float,
    tol_rel: float,
) -> tuple[bool, str]:
    try:
        worst_diff = 0.0
        worst_rel = 0.0
        worst_sample = -1
        worst_pair: tuple[float, float] | None = None
        for i, (momenta, pols) in enumerate(kinematics_samples):
            val_simple = gd.eval_infix_numeric(simple, momenta, pols)
            val_scrambled = gd.eval_infix_numeric(scrambled, momenta, pols)
            if not (math.isfinite(val_simple) and math.isfinite(val_scrambled)):
                return False, f"non-finite evaluation at sample {i}: simple={val_simple}, scrambled={val_scrambled}"
            diff = abs(val_simple - val_scrambled)
            scale = max(abs(val_simple), abs(val_scrambled), 1.0)
            rel = diff / scale
            if diff > worst_diff:
                worst_diff = diff
                worst_rel = rel
                worst_sample = i
                worst_pair = (val_simple, val_scrambled)
            if not (diff <= tol_abs or rel <= tol_rel):
                return (
                    False,
                    f"mismatch at sample {i}: simple={val_simple:.12g}, scrambled={val_scrambled:.12g}, "
                    f"abs_diff={diff:.3e}, rel_diff={rel:.3e}",
                )
        if worst_pair is None:
            return True, "ok"
        return (
            True,
            f"ok (worst sample {worst_sample}: simple={worst_pair[0]:.12g}, "
            f"scrambled={worst_pair[1]:.12g}, abs_diff={worst_diff:.3e}, rel_diff={worst_rel:.3e})",
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def analyze_csv(
    csv_path: Path,
    *,
    numeric_checks: int = 3,
    numeric_mass: float = 2.0,
    tol_abs: float = 1e-12,
    tol_rel: float = 1e-10,
    skip_numeric: bool = False,
) -> int:
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8)
    warnings: list[WarningEntry] = []
    simple_lengths: list[int] = []
    scrambled_lengths: list[int] = []
    combined_lengths: list[int] = []
    row_max_indices: list[int] = []
    rows: list[dict[str, str]] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"simple", "scrambled"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(f"Expected CSV columns {expected}, got {reader.fieldnames}")
        rows = list(reader)

    file_n = infer_n(*(row["simple"] for row in rows), *(row["scrambled"] for row in rows))
    if file_n is None:
        raise ValueError("Could not infer N from any expression in the CSV")
    kinematics_samples: list[tuple[object, object]] = []
    if not skip_numeric:
        kinematics_samples = [
            generate_kinematics(file_n, M=numeric_mass, seed=12345 + i)
            for i in range(numeric_checks)
        ]

    for row_id, row in enumerate(rows):
        csv_line = row_id + 2
        simple = row["simple"]
        scrambled = row["scrambled"]
        row_n = infer_n(simple, scrambled)
        if row_n is not None:
            row_max_indices.append(row_n)

        try:
            s_len = len(tokenizer.encode_infix(simple))
            simple_lengths.append(s_len)
            combined_lengths.append(s_len)
        except Exception as exc:
            warnings.append(
                WarningEntry(row_id, csv_line, "simple", "tokenization_error", str(exc), simple)
            )

        try:
            t_len = len(tokenizer.encode_infix(scrambled))
            scrambled_lengths.append(t_len)
            combined_lengths.append(t_len)
        except Exception as exc:
            warnings.append(
                WarningEntry(row_id, csv_line, "scrambled", "tokenization_error", str(exc), scrambled)
            )

        for col_name, expr in (("simple", simple), ("scrambled", scrambled)):
            masses = find_mass_terms(expr)
            if masses:
                warnings.append(
                    WarningEntry(
                        row_id,
                        csv_line,
                        col_name,
                        "mass_term",
                        f"Found mass terms: {', '.join(masses)}",
                        expr,
                    )
                )

        try:
            mass_dim = gd.manifest_mass_dimension(simple)
            target_dim = 4 - file_n
            if mass_dim != target_dim:
                warnings.append(
                    WarningEntry(
                        row_id,
                        csv_line,
                        "simple",
                        "mass_dimension_mismatch",
                        f"Found manifest dimension {mass_dim}, expected {target_dim}",
                        simple,
                    )
                )
        except Exception as exc:
            warnings.append(
                WarningEntry(
                    row_id,
                    csv_line,
                    "simple",
                    "mass_dimension_check_failed",
                    str(exc),
                    simple,
                )
            )

        present_photons = photon_legs_present(simple, scrambled, file_n)
        expected_photons = set(range(2, file_n))
        missing = sorted(expected_photons - present_photons)
        if missing:
            warnings.append(
                WarningEntry(
                    row_id,
                    csv_line,
                    "both",
                    "missing_photon_legs",
                    f"Missing photon legs: {missing}",
                    simple,
                )
            )

        for issue in check_spurious_poles(simple, file_n):
            warnings.append(
                WarningEntry(
                    row_id,
                    csv_line,
                    "simple",
                    "spurious_pole",
                    issue,
                    simple,
                )
            )

        if not skip_numeric:
            ok, details = numerically_compare(
                simple,
                scrambled,
                kinematics_samples,
                tol_abs=tol_abs,
                tol_rel=tol_rel,
            )
            if not ok:
                warnings.append(
                    WarningEntry(
                        row_id,
                        csv_line,
                        "both",
                        "numerical_mismatch",
                        details,
                        f"simple={simple} || scrambled={scrambled}",
                    )
                )

    print(f"Analyzed: {csv_path}")
    print()

    if combined_lengths:
        print("Token-length stats")
        print(
            f"  Combined  : min={min(combined_lengths)}, avg={statistics.mean(combined_lengths):.2f}, max={max(combined_lengths)}"
        )
        if simple_lengths:
            print(
                f"  Simple    : min={min(simple_lengths)}, avg={statistics.mean(simple_lengths):.2f}, max={max(simple_lengths)}"
            )
        if scrambled_lengths:
            print(
                f"  Scrambled : min={min(scrambled_lengths)}, avg={statistics.mean(scrambled_lengths):.2f}, max={max(scrambled_lengths)}"
        )
        print()

    unique_row_max = sorted(set(row_max_indices))
    print(f"File-wide inferred N: {file_n}")
    if unique_row_max:
        print(f"Row-wise max index values seen: {unique_row_max}")
        if len(unique_row_max) > 1:
            print("Note: some rows do not mention the highest leg explicitly; checks use the file-wide N.")
    print()

    if skip_numeric:
        print("Numerical checks: skipped")
    else:
        print(
            "Numerical checks: "
            f"{numeric_checks} phase-space points, M={numeric_mass}, "
            f"tol_abs={tol_abs:.1e}, tol_rel={tol_rel:.1e}"
        )
    print()

    if not warnings:
        print("No warnings found.")
        return 0

    print(f"Warnings found: {len(warnings)}")
    print()
    for item in warnings:
        print(f"[row {item.row_id} | csv line {item.csv_line} | {item.column} | {item.issue}]")
        print(f"  {item.details}")
        print(f"  {item.expression}")
        print()

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a raw generated amplitude CSV.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--numeric-checks", type=int, default=3)
    parser.add_argument("--numeric-mass", type=float, default=2.0)
    parser.add_argument("--tol-abs", type=float, default=1e-12)
    parser.add_argument("--tol-rel", type=float, default=1e-10)
    parser.add_argument("--skip-numeric", action="store_true")
    args = parser.parse_args()
    return analyze_csv(
        args.input_csv,
        numeric_checks=args.numeric_checks,
        numeric_mass=args.numeric_mass,
        tol_abs=args.tol_abs,
        tol_rel=args.tol_rel,
        skip_numeric=args.skip_numeric,
    )


if __name__ == "__main__":
    raise SystemExit(main())
