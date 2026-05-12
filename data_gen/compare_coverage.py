#!/usr/bin/env python3
"""
compare_coverage.py — compare two amplitude-expression CSV datasets.

The script expects raw CSV files with at least a `simple` column. It compares
structural coverage of the gauge-invariant target expressions, including term
counts, block mixtures, chain endpoints, coefficient patterns, denominator
powers, and repeated-denominator poles that are supported by hidden p.F/F.F
expansion factors.

Example:
    python compare_coverage.py data/new.csv data/old.csv \
      --name-a new --name-b old \
      --slice-b 150:300 \
      --out-prefix reports/new_vs_old

Outputs:
    <out-prefix>.txt
    <out-prefix>.png
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Counter, Iterable, Sequence

DOT = "·"

_RE_PP = re.compile(r"p_(\d+)\s*[·.]\s*p_(\d+)")
_RE_CHAIN = re.compile(r"p_(\d+)((?:\s*[·.]\s*F_\d+)+)\s*[·.]\s*p_(\d+)")
_RE_TRACE = re.compile(r"Tr\((F_\d+(?:\s*[·.,]\s*F_\d+)*)\)")
_RE_INT = re.compile(r"^[+-]?\d+$")


def p(i: int) -> str:
    return f"p_{i}"


def canon_pp_text(a: int | str, b: int | str) -> str:
    ia, ib = int(a), int(b)
    if ia > ib:
        ia, ib = ib, ia
    return f"p_{ia} {DOT} p_{ib}"


def strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                ok = False
                break
        if not ok:
            break
        s = s[1:-1].strip()
    return s


def normalize_expr(s: str) -> str:
    return s.strip().replace("**", "^").replace(".", DOT)


def split_top_level(s: str, sep: str) -> list[str]:
    """Split on a single-character separator at parenthesis depth zero."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf or not out:
        out.append("".join(buf).strip())
    return [x for x in out if x]


def split_top_level_sum(expr: str) -> list[str]:
    expr = normalize_expr(expr)
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
            if prev not in "*/^(+-":
                parts.append("".join(current).strip())
                current = [ch]
                continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def split_term_num_den(term_body: str) -> tuple[str, str]:
    """Return numerator, denominator for one top-level simple-expression term."""
    body = strip_outer_parens(normalize_expr(term_body))
    depth = 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "/" and depth == 0:
            return strip_outer_parens(body[:i]), strip_outer_parens(body[i + 1 :])
    return body, ""


def remove_leading_coefficient(term: str) -> tuple[int, str]:
    """Extract a top-level integer coefficient and the remaining term body."""
    s = normalize_expr(term).strip()
    sign = 1
    if s.startswith("+"):
        s = s[1:].strip()
    elif s.startswith("-"):
        sign = -1
        s = s[1:].strip()

    # Cases like 17*(...), 17*p_1·p_2, etc.
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "*" and depth == 0:
            maybe = s[:i].strip()
            if _RE_INT.fullmatch(maybe):
                return sign * int(maybe), s[i + 1 :].strip()
            break

    # Rare case: the whole term is just an integer.
    if _RE_INT.fullmatch(s):
        return sign * int(s), "1"
    return sign, s


def factor_power(factor: str) -> tuple[str, int]:
    f = strip_outer_parens(normalize_expr(factor))
    # Need to detect (base)^n before stripping all enclosing parens can erase the base.
    raw = normalize_expr(factor).strip()
    m = re.fullmatch(r"\((.+)\)\s*\^\s*(\d+)", raw)
    if m:
        return strip_outer_parens(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(.+?)\s*\^\s*(\d+)", f)
    if m:
        return strip_outer_parens(m.group(1)), int(m.group(2))
    return f, 1


def product_factors(expr: str) -> list[tuple[str, int]]:
    expr = strip_outer_parens(normalize_expr(expr))
    if not expr or expr == "1":
        return []
    out: list[tuple[str, int]] = []
    for factor in split_top_level(expr, "*"):
        if not factor:
            continue
        base, power = factor_power(factor)
        out.append((base, power))
    return out


def pp_multiplicities(expr: str) -> Counter[str]:
    counts: Counter[str] = collections.Counter()
    for base, power in product_factors(expr):
        m = _RE_PP.fullmatch(strip_outer_parens(base))
        if m:
            counts[canon_pp_text(m.group(1), m.group(2))] += power
    return counts


def block_factors(expr: str) -> list[dict]:
    blocks: list[dict] = []
    for base, power in product_factors(expr):
        b = strip_outer_parens(base)
        m = _RE_CHAIN.fullmatch(b)
        if m:
            left = int(m.group(1))
            photons = tuple(int(x) for x in re.findall(r"F_(\d+)", m.group(2)))
            right = int(m.group(3))
            for _ in range(power):
                blocks.append({"kind": "chain", "left": left, "photons": photons, "right": right})
            continue
        m = _RE_TRACE.fullmatch(b)
        if m:
            photons = tuple(int(x) for x in re.findall(r"F_(\d+)", m.group(1)))
            for _ in range(power):
                blocks.append({"kind": "trace", "photons": photons})
    return blocks


def hidden_pp_support_counts(blocks: Sequence[dict]) -> Counter[str]:
    """p.p factors that can appear inside expansions of p.F...F.p or Tr(F...).

    This deliberately tracks adjacency in F chains/traces, rather than explicit
    scalar numerator factors. For repeated denominator poles, m_D - 1 copies
    can be regarded as nontrivial expansion-spurious only if this support is
    present.
    """
    counts: Counter[str] = collections.Counter()
    for block in blocks:
        photons = tuple(block["photons"])
        if block["kind"] == "chain":
            left = int(block["left"])
            right = int(block["right"])
            if photons:
                if left != photons[0]:
                    counts[canon_pp_text(left, photons[0])] += 1
                for a, b in zip(photons, photons[1:]):
                    if a != b:
                        counts[canon_pp_text(a, b)] += 1
                if right != photons[-1]:
                    counts[canon_pp_text(photons[-1], right)] += 1
        elif block["kind"] == "trace" and len(photons) >= 2:
            cyc = list(photons) + [photons[0]]
            for a, b in zip(cyc, cyc[1:]):
                if a != b:
                    counts[canon_pp_text(a, b)] += 1
    return counts


def count_product_dim(expr: str) -> int:
    total = 0
    for base, power in product_factors(expr):
        b = strip_outer_parens(base)
        if _RE_INT.fullmatch(b):
            continue
        m = _RE_PP.fullmatch(b)
        if m:
            total += 2 * power
            continue
        m = _RE_TRACE.fullmatch(b)
        if m:
            total += len(re.findall(r"F_\d+", m.group(1))) * power
            continue
        m = _RE_CHAIN.fullmatch(b)
        if m:
            total += (len(re.findall(r"F_\d+", m.group(2))) + 2) * power
            continue
        # Fall back: if the factor is a parenthesized product, recurse.
        if "*" in b:
            total += count_product_dim(b) * power
            continue
    return total


def manifest_dimensions(simple_expr: str) -> list[int]:
    dims: list[int] = []
    for term in split_top_level_sum(simple_expr):
        _coeff, body = remove_leading_coefficient(term)
        num, den = split_term_num_den(body)
        dims.append(count_product_dim(num) - count_product_dim(den))
    return dims


@dataclass
class TermAnalysis:
    coeff: int
    body: str
    numerator: str
    denominator: str
    blocks: list[dict]
    denom_counts: Counter[str]
    numerator_pp_counts: Counter[str]
    hidden_support_counts: Counter[str]
    dim: int

    @property
    def has_denom_power(self) -> bool:
        return any(v >= 2 for v in self.denom_counts.values())

    @property
    def has_num_scalar_power(self) -> bool:
        return any(v >= 2 for v in self.numerator_pp_counts.values())

    @property
    def repeated_denom_factors(self) -> list[str]:
        return [k for k, v in self.denom_counts.items() if v >= 2]

    @property
    def unsupported_repeated_denoms(self) -> list[str]:
        bad: list[str] = []
        for pole, mult in self.denom_counts.items():
            if mult < 2:
                continue
            # One copy may be physical; extra copies should be hidden-expansion spurious.
            extra = mult - 1
            if self.hidden_support_counts.get(pole, 0) < extra:
                bad.append(pole)
        return bad

    @property
    def trivial_repeated_denoms(self) -> list[str]:
        return [pole for pole, mult in self.denom_counts.items() if mult >= 2 and self.numerator_pp_counts.get(pole, 0) > 0]

    @property
    def relaxed_endpoint_uses(self) -> int:
        count = 0
        for block in self.blocks:
            if block["kind"] != "chain":
                continue
            photons = set(block["photons"])
            # This is exactly what the older all-F-leg exclusion would forbid.
            if int(block["left"]) in photons:
                count += 1
            if int(block["right"]) in photons:
                count += 1
        return count

    def exact_signature(self) -> tuple:
        chain_sig = sorted(
            (b["left"], b["photons"], b["right"])
            for b in self.blocks
            if b["kind"] == "chain"
        )
        trace_sig = sorted(tuple(b["photons"]) for b in self.blocks if b["kind"] == "trace")
        den_sig = tuple(sorted(self.denom_counts.items()))
        num_pp_sig = tuple(sorted(self.numerator_pp_counts.items()))
        return (tuple(chain_sig), tuple(trace_sig), den_sig, num_pp_sig, self.dim)

    def coarse_signature(self) -> tuple:
        chain_lens = tuple(sorted(len(b["photons"]) for b in self.blocks if b["kind"] == "chain"))
        trace_lens = tuple(sorted(len(b["photons"]) for b in self.blocks if b["kind"] == "trace"))
        den_mults = tuple(sorted(self.denom_counts.values()))
        has_denom_power = self.has_denom_power
        has_num_power = self.has_num_scalar_power
        return (chain_lens, trace_lens, den_mults, has_denom_power, has_num_power, self.dim)


def analyze_term(term: str) -> TermAnalysis:
    coeff, body = remove_leading_coefficient(term)
    num, den = split_term_num_den(body)
    blocks = block_factors(num)
    denom_counts = pp_multiplicities(den)
    numerator_pp_counts = pp_multiplicities(num)
    hidden_support = hidden_pp_support_counts(blocks)
    dim = count_product_dim(num) - count_product_dim(den)
    return TermAnalysis(
        coeff=coeff,
        body=body,
        numerator=num,
        denominator=den,
        blocks=blocks,
        denom_counts=denom_counts,
        numerator_pp_counts=numerator_pp_counts,
        hidden_support_counts=hidden_support,
        dim=dim,
    )


@dataclass
class DatasetAnalysis:
    name: str
    path: Path
    rows: list[dict]
    terms_by_row: list[list[TermAnalysis]] = field(default_factory=list)
    parse_errors: list[tuple[int, str]] = field(default_factory=list)

    @property
    def terms(self) -> list[TermAnalysis]:
        return [t for row in self.terms_by_row for t in row]

    def row_count(self) -> int:
        return len(self.rows)

    def term_count_distribution(self) -> Counter[int]:
        return collections.Counter(len(row) for row in self.terms_by_row)

    def dimension_distribution(self) -> Counter[str]:
        c: Counter[str] = collections.Counter()
        for row in self.terms_by_row:
            dims = tuple(t.dim for t in row)
            c[str(dims)] += 1
        return c

    def block_counts(self) -> Counter[str]:
        c: Counter[str] = collections.Counter()
        for term in self.terms:
            for b in term.blocks:
                key = f"{b['kind']}{len(b['photons'])}"
                c[key] += 1
        return c

    def coefficient_distribution(self) -> Counter[str]:
        c: Counter[str] = collections.Counter()
        for row in self.terms_by_row:
            if not row:
                continue
            if all(abs(t.coeff) == 1 for t in row):
                c["all ±1"] += 1
            elif all(t.coeff > 0 for t in row):
                c["all positive"] += 1
            else:
                c["mixed/general"] += 1
        return c

    def row_flags(self) -> Counter[str]:
        c: Counter[str] = collections.Counter()
        for row in self.terms_by_row:
            if any(t.relaxed_endpoint_uses for t in row):
                c["relaxed_endpoint"] += 1
            if any(t.has_denom_power for t in row):
                c["denominator_power"] += 1
            if any(t.has_num_scalar_power for t in row):
                c["numerator_scalar_power"] += 1
            if any(t.unsupported_repeated_denoms for t in row):
                c["unsupported_repeated_denominator"] += 1
            if any(t.trivial_repeated_denoms for t in row):
                c["manifest_trivial_repeated_denominator"] += 1
        return c

    def exact_term_signatures(self) -> set[tuple]:
        return {t.exact_signature() for t in self.terms}

    def coarse_term_signatures(self) -> set[tuple]:
        return {t.coarse_signature() for t in self.terms}

    def unique_simple_count(self) -> int:
        return len({r.get("simple", "") for r in self.rows})

    def unique_scrambled_count(self) -> int:
        return len({r.get("scrambled", "") for r in self.rows if "scrambled" in r})


def read_csv_rows(path: Path, simple_col: str, slice_spec: str | None) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        if simple_col not in reader.fieldnames:
            raise ValueError(f"{path} has no {simple_col!r} column; found {reader.fieldnames}")
        rows = list(reader)
    if slice_spec:
        start, end = parse_slice(slice_spec)
        rows = rows[start:end]
    return rows


def parse_slice(spec: str) -> tuple[int | None, int | None]:
    if ":" not in spec:
        i = int(spec)
        return i, i + 1
    a, b = spec.split(":", 1)
    return (int(a) if a else None, int(b) if b else None)


def analyze_dataset(path: Path, name: str, simple_col: str, slice_spec: str | None) -> DatasetAnalysis:
    rows = read_csv_rows(path, simple_col, slice_spec)
    analysis = DatasetAnalysis(name=name, path=path, rows=rows)
    for idx, row in enumerate(rows):
        simple = row[simple_col]
        try:
            terms = [analyze_term(t) for t in split_top_level_sum(simple)]
            analysis.terms_by_row.append(terms)
        except Exception as exc:  # keep report useful on imperfect data
            analysis.parse_errors.append((idx, f"{type(exc).__name__}: {exc}"))
            analysis.terms_by_row.append([])
    return analysis


def pct(n: int | float, d: int | float) -> str:
    if d == 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def counter_table(counter: Counter, *, max_items: int = 20) -> str:
    if not counter:
        return "  (none)\n"
    lines = []
    for key, val in counter.most_common(max_items):
        lines.append(f"  {key}: {val}")
    if len(counter) > max_items:
        lines.append(f"  ... {len(counter) - max_items} more")
    return "\n".join(lines) + "\n"


def coverage_text(a: DatasetAnalysis, b: DatasetAnalysis) -> str:
    a_exact, b_exact = a.exact_term_signatures(), b.exact_term_signatures()
    a_coarse, b_coarse = a.coarse_term_signatures(), b.coarse_term_signatures()
    exact_ab = len(a_exact & b_exact)
    coarse_ab = len(a_coarse & b_coarse)
    return (
        f"Exact term-signature overlap: {exact_ab}\n"
        f"  {a.name} covers {exact_ab}/{len(b_exact)} unique exact {b.name} signatures ({pct(exact_ab, len(b_exact))})\n"
        f"  {b.name} covers {exact_ab}/{len(a_exact)} unique exact {a.name} signatures ({pct(exact_ab, len(a_exact))})\n"
        f"Coarse term-structure overlap: {coarse_ab}\n"
        f"  {a.name} covers {coarse_ab}/{len(b_coarse)} unique coarse {b.name} structures ({pct(coarse_ab, len(b_coarse))})\n"
        f"  {b.name} covers {coarse_ab}/{len(a_coarse)} unique coarse {a.name} structures ({pct(coarse_ab, len(a_coarse))})\n"
    )


def dataset_summary_text(ds: DatasetAnalysis) -> str:
    flags = ds.row_flags()
    terms = ds.terms
    repeated_total = sum(len(t.repeated_denom_factors) for t in terms)
    unsupported_total = sum(len(t.unsupported_repeated_denoms) for t in terms)
    trivial_total = sum(len(t.trivial_repeated_denoms) for t in terms)
    return (
        f"Dataset: {ds.name}\n"
        f"Path: {ds.path}\n"
        f"Rows: {ds.row_count()}\n"
        f"Terms parsed: {len(terms)}\n"
        f"Unique simple expressions: {ds.unique_simple_count()}\n"
        f"Unique scrambled expressions: {ds.unique_scrambled_count()}\n"
        f"Parse errors: {len(ds.parse_errors)}\n"
        f"Rows with relaxed endpoint usage: {flags['relaxed_endpoint']} ({pct(flags['relaxed_endpoint'], ds.row_count())})\n"
        f"Rows with denominator powers: {flags['denominator_power']} ({pct(flags['denominator_power'], ds.row_count())})\n"
        f"Rows with numerator scalar powers: {flags['numerator_scalar_power']} ({pct(flags['numerator_scalar_power'], ds.row_count())})\n"
        f"Repeated denominator factors: {repeated_total}\n"
        f"Unsupported repeated denominator factors: {unsupported_total}\n"
        f"Manifest-trivial repeated denominator factors: {trivial_total}\n"
        "Term-count distribution:\n"
        f"{counter_table(ds.term_count_distribution())}"
        "Coefficient row classes:\n"
        f"{counter_table(ds.coefficient_distribution())}"
        "Block counts:\n"
        f"{counter_table(ds.block_counts())}"
        "Manifest dimension tuple distribution:\n"
        f"{counter_table(ds.dimension_distribution(), max_items=10)}"
    )


def write_report(a: DatasetAnalysis, b: DatasetAnalysis, out_txt: Path) -> str:
    report = []
    report.append("Coverage comparison report")
    report.append("=" * 80)
    report.append("")
    report.append(dataset_summary_text(a))
    report.append("-" * 80)
    report.append(dataset_summary_text(b))
    report.append("-" * 80)
    report.append("Coverage")
    report.append(coverage_text(a, b))

    for ds in (a, b):
        if ds.parse_errors:
            report.append("-" * 80)
            report.append(f"Parse errors for {ds.name} (first 10)")
            for idx, msg in ds.parse_errors[:10]:
                report.append(f"  row {idx}: {msg}")

    text = "\n".join(report).rstrip() + "\n"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(text, encoding="utf-8")
    return text


def make_plots(a: DatasetAnalysis, b: DatasetAnalysis, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [a.name, b.name]
    fig, axes = plt.subplots(5, 1, figsize=(11, 18), constrained_layout=True)

    # 1. Dataset basics.
    basic_labels = ["rows", "terms", "unique simple", "unique scrambled"]
    basic_a = [a.row_count(), len(a.terms), a.unique_simple_count(), a.unique_scrambled_count()]
    basic_b = [b.row_count(), len(b.terms), b.unique_simple_count(), b.unique_scrambled_count()]
    x = range(len(basic_labels))
    width = 0.38
    axes[0].bar([i - width / 2 for i in x], basic_a, width, label=a.name)
    axes[0].bar([i + width / 2 for i in x], basic_b, width, label=b.name)
    axes[0].set_title("Dataset size and uniqueness")
    axes[0].set_xticks(list(x), basic_labels, rotation=20, ha="right")
    axes[0].legend()

    # 2. Term-count distribution.
    term_keys = sorted(set(a.term_count_distribution()) | set(b.term_count_distribution()))
    ca, cb = a.term_count_distribution(), b.term_count_distribution()
    x = range(len(term_keys))
    axes[1].bar([i - width / 2 for i in x], [ca.get(k, 0) for k in term_keys], width, label=a.name)
    axes[1].bar([i + width / 2 for i in x], [cb.get(k, 0) for k in term_keys], width, label=b.name)
    axes[1].set_title("Rows by number of top-level simple terms")
    axes[1].set_xticks(list(x), [str(k) for k in term_keys])
    axes[1].set_xlabel("top-level terms")
    axes[1].legend()

    # 3. Block counts.
    block_order = ["chain1", "chain2", "chain3", "chain4", "trace2", "trace3", "trace4"]
    ba, bb = a.block_counts(), b.block_counts()
    x = range(len(block_order))
    axes[2].bar([i - width / 2 for i in x], [ba.get(k, 0) for k in block_order], width, label=a.name)
    axes[2].bar([i + width / 2 for i in x], [bb.get(k, 0) for k in block_order], width, label=b.name)
    axes[2].set_title("Gauge-invariant block counts")
    axes[2].set_xticks(list(x), block_order, rotation=20, ha="right")
    axes[2].legend()

    # 4. Row-level feature flags.
    flag_order = [
        "relaxed_endpoint",
        "denominator_power",
        "numerator_scalar_power",
        "unsupported_repeated_denominator",
        "manifest_trivial_repeated_denominator",
    ]
    fa, fb = a.row_flags(), b.row_flags()
    x = range(len(flag_order))
    axes[3].bar([i - width / 2 for i in x], [fa.get(k, 0) for k in flag_order], width, label=a.name)
    axes[3].bar([i + width / 2 for i in x], [fb.get(k, 0) for k in flag_order], width, label=b.name)
    axes[3].set_title("Rows with structural features")
    axes[3].set_xticks(list(x), flag_order, rotation=25, ha="right")
    axes[3].legend()

    # 5. Coverage bars.
    a_exact, b_exact = a.exact_term_signatures(), b.exact_term_signatures()
    a_coarse, b_coarse = a.coarse_term_signatures(), b.coarse_term_signatures()
    exact_overlap = len(a_exact & b_exact)
    coarse_overlap = len(a_coarse & b_coarse)
    cov_labels = [
        f"{a.name} covers\n{b.name} exact",
        f"{b.name} covers\n{a.name} exact",
        f"{a.name} covers\n{b.name} coarse",
        f"{b.name} covers\n{a.name} coarse",
    ]
    cov_vals = [
        100.0 * exact_overlap / len(b_exact) if b_exact else 0.0,
        100.0 * exact_overlap / len(a_exact) if a_exact else 0.0,
        100.0 * coarse_overlap / len(b_coarse) if b_coarse else 0.0,
        100.0 * coarse_overlap / len(a_coarse) if a_coarse else 0.0,
    ]
    axes[4].bar(range(len(cov_labels)), cov_vals)
    axes[4].set_title("Unique term-signature coverage")
    axes[4].set_ylabel("coverage (%)")
    axes[4].set_ylim(0, 105)
    axes[4].set_xticks(range(len(cov_labels)), cov_labels)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare structural coverage of two amplitude CSV datasets.")
    parser.add_argument("csv_a", type=Path, help="First CSV file")
    parser.add_argument("csv_b", type=Path, help="Second CSV file")
    parser.add_argument("--name-a", default=None, help="Label for first dataset")
    parser.add_argument("--name-b", default=None, help="Label for second dataset")
    parser.add_argument("--slice-a", default=None, help="Optional row slice for first CSV, e.g. 0:150 or 150:")
    parser.add_argument("--slice-b", default=None, help="Optional row slice for second CSV, e.g. 0:150 or 150:")
    parser.add_argument("--simple-column", default="simple", help="Column containing raw simple expressions")
    parser.add_argument("--out-prefix", type=Path, default=Path("coverage_comparison"), help="Output prefix without extension")
    parser.add_argument("--no-plots", action="store_true", help="Only write the text report")
    args = parser.parse_args()

    name_a = args.name_a or args.csv_a.stem
    name_b = args.name_b or args.csv_b.stem
    a = analyze_dataset(args.csv_a, name_a, args.simple_column, args.slice_a)
    b = analyze_dataset(args.csv_b, name_b, args.simple_column, args.slice_b)

    out_txt = args.out_prefix.with_suffix(".txt")
    out_png = args.out_prefix.with_suffix(".png")
    text = write_report(a, b, out_txt)
    if not args.no_plots:
        try:
            make_plots(a, b, out_png)
        except Exception as exc:
            text += f"\nPlot generation failed: {type(exc).__name__}: {exc}\n"
            out_txt.write_text(text, encoding="utf-8")
            print(text)
            print(f"Wrote text report to {out_txt}")
            return 1

    print(text)
    print(f"Wrote text report to {out_txt}")
    if not args.no_plots:
        print(f"Wrote plot file to {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
