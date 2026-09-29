"""Reproduce the symbolic preparation used for the five-point worked example.

Run from the repository root with::

    python -m data_testing.ym_5pt_workflow.prepare --output-dir /tmp/ym5-prepared

This regenerates all 80 projected components and their 16 cyclic orbits, then
exports the historical 16 model inputs. The representatives were selected using
exploratory model predictions; this is a reproducible worked example, not an
unbiased model benchmark. No checkpoint or original investigation logs are read.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

import sympy as sp

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_gen.data_gen_ym.algebra import full_expand_expression, simplify_to_lowest_terms
from data_gen.data_gen_ym.generate_clean_4pt import parenthesize_for_semantic_tokenization

from . import algebra

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL = REPO_ROOT / "data/data_ym/gluon5feyn.csv.gz"
GAUGE_REFERENCES = (2, 3, 4, 5, 1)
SELECTED_REPRESENTATIVES = (
    "P001", "P067", "P003", "P004", "P037", "P017", "P019", "P025",
    "P050", "P034", "P018", "P026", "P035", "P054", "P038", "P039",
)
METHOD = "16 independently gauge-invariant scalar components, with five cyclic images of their weighted sum"
CHAIN = re.compile(r"p_([1-5])\s*·\s*(F_[1-5](?:\s*·\s*F_[1-5])*)\s*·\s*p_([1-5])")
DOT = re.compile(r"p_([1-5])\s*·\s*p_([1-5])")


def rotate(text: str, shift: int) -> str:
    return re.sub(r"\b([peF])_([1-5])\b", lambda m: f"{m[1]}_{(int(m[2])-1+shift)%5+1}", text)


def render(expr, names=None) -> str:
    """Render rational expressions without changing the original input order."""
    names = names or {}
    if isinstance(expr, sp.Symbol):
        if expr in names:
            return "(" + names[expr] + ")"
        if expr in algebra.LABELS:
            (a, i), (b, j) = algebra.LABELS[expr]
            return f"({a}_{i} · {b}_{j})"
        raise ValueError(f"Unknown scalar atom: {expr}")
    if expr.is_Integer:
        return str(expr)
    if expr.is_Rational:
        return f"({expr.p}/{expr.q})"
    if expr.is_Add:
        return "(" + " + ".join(render(x, names) for x in expr.args) + ")"
    if expr.is_Mul:
        numerator, denominator = expr.as_numer_denom()
        if denominator != 1:
            return f"({render(numerator, names)}/{render(denominator, names)})"
        return "(" + "*".join(render(x, names) for x in expr.args) + ")"
    if expr.is_Pow:
        return f"({render(expr.base, names)}^{render(expr.exp, names)})"
    raise ValueError(f"Unsupported expression: {expr}")


def chain_symbol(left, field_strengths, right, names):
    # p_i.F_i = F_i.p_i = 0 uses masslessness and transversality.
    if left == field_strengths[0] or right == field_strengths[-1]:
        return sp.Integer(0)
    if len(field_strengths) == 1 and left == right:
        return sp.Integer(0)
    symbol = sp.Symbol("C" + str(left) + "".join(map(str, field_strengths)) + str(right))
    names[symbol] = " · ".join([f"p_{left}"] + [f"F_{i}" for i in field_strengths] + [f"p_{right}"])
    return symbol


def construct(q, full):
    """Substitute e_i -> q_i.F_i/(q_i.p_i), with q_i a momentum label."""
    names, substitutions = {}, {}
    for symbol in full.free_symbols:
        (a, i), (b, j) = algebra.LABELS[symbol]
        if a == "e" and b == "p":
            substitutions[symbol] = chain_symbol(q[i-1], (i,), j, names) / algebra.atom("p", q[i-1], "p", i)
        elif a == b == "e":
            substitutions[symbol] = -chain_symbol(q[i-1], (i, j), q[j-1], names) / (
                algebra.atom("p", q[i-1], "p", i) * algebra.atom("p", q[j-1], "p", j)
            )
    return sp.expand(full.xreplace(substitutions)), names


def canonical(text: str):
    """Match orbits using commuting scalars, symmetric dots and F reversal.

    p_a.F_i1...F_in.p_b = (-1)^n p_b.F_in...F_i1.p_a.
    No on-shell or dimension-specific identities enter this orbit comparison.
    """
    atoms = {}

    def replace_chain(match):
        word = (int(match[1]), *(int(x) for x in re.findall(r"F_([1-5])", match[2])), int(match[3]))
        reverse = word[::-1]
        sign = (-1) ** (len(word)-2) if reverse < word else 1
        name = "C" + "".join(map(str, min(word, reverse)))
        atoms[name] = sp.Symbol(name)
        return ("-" if sign == -1 else "") + name

    def replace_dot(match):
        name = "P" + "".join(map(str, sorted((int(match[1]), int(match[2])))))
        atoms[name] = sp.Symbol(name)
        return name

    tree = ast.parse(DOT.sub(replace_dot, CHAIN.sub(replace_chain, text)).replace("^", "**"), mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in atoms:
            return atoms[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Integer:
                return left ** right
        raise ValueError(f"Unsupported orbit expression: {ast.dump(node)}")

    return sp.cancel(visit(tree))


def construct_components(full):
    projected, names = construct(GAUGE_REFERENCES, full)
    variables = sorted(projected.free_symbols & names.keys(), key=str)
    polynomial = sp.Poly(projected, *variables)
    components = []
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
    for monomial, coefficient in polynomial.terms():
        polarizations = sp.prod(v ** n for v, n in zip(variables, monomial))
        for coefficient_term in sp.Add.make_args(sp.expand(coefficient)):
            multiplier, rest = coefficient_term.as_coeff_Mul()
            reference = render(rest * polarizations, names)
            expanded = full_expand_expression(reference)
            expanded = re.sub(
                r"([pe])_(\d+)\s*·\s*([pe])_(\d+)",
                lambda m: "0" if m[2] == m[4] and (m[1] == "p" or m[3] == "p") else m[0],
                expanded,
            )
            source = parenthesize_for_semantic_tokenization(simplify_to_lowest_terms(expanded))
            name = f"P{len(components)+1:03d}"
            # Independent parser proves that the training-time expansion routine
            # has preserved the tensor expression supplied to it.
            if algebra.on_shell(algebra.parse(source) - algebra.parse(reference)) != 0:
                raise RuntimeError(f"Scalar expansion failed exact verification: {name}")
            components.append({
                "name": name, "expression": source, "reference": reference,
                "multiplier": str(multiplier), "input_tokens": len(tokenizer.encode_infix(source)),
                "target_tokens": len(tokenizer.encode_infix(reference)),
            })
    if len(components) != 80:
        raise RuntimeError(f"Expected the original 80 components; obtained {len(components)}")
    expanded_sum = sum(sp.Rational(c["multiplier"]) * algebra.parse(c["reference"]) for c in components)
    if algebra.on_shell(expanded_sum - full) != 0:
        raise RuntimeError("Projected components do not reconstruct the complete amplitude")
    return components


def group_orbits(components):
    lookup = {c["name"]: c for c in components}
    weighted = {c["name"]: canonical(f"({c['multiplier']})*({c['reference']})") for c in components}
    remaining = set(weighted)
    orbits, mapping = [], []
    while remaining:
        initial = lookup[min(remaining)]
        rotations = [canonical(f"({initial['multiplier']})*({rotate(initial['reference'], k)})") for k in range(5)]
        members = [name for name in sorted(remaining) if weighted[name] in rotations]
        selected = [name for name in SELECTED_REPRESENTATIVES if name in members]
        if len(members) != 5 or len(selected) != 1:
            raise RuntimeError(f"Unexpected orbit for {initial['name']}: {members}, selected={selected}")
        base = lookup[selected[0]]
        rotations = [canonical(f"({base['multiplier']})*({rotate(base['reference'], k)})") for k in range(5)]
        member_records = []
        for name in members:
            shift = rotations.index(weighted[name])
            ratio = sp.Rational(base["multiplier"]) / sp.Rational(lookup[name]["multiplier"])
            if canonical(lookup[name]["reference"]) != ratio * canonical(rotate(base["reference"], shift)):
                raise RuntimeError(f"Orbit identity failed for {name}")
            record = {
                "component": name, "base": base["name"], "orbit": len(orbits)+1,
                "rotation": shift, "reference_sign": str(ratio), "weighted_identity_exact": True,
            }
            member_records.append(record)
            mapping.append(record)
        orbits.append({
            "orbit": len(orbits)+1, "base": base["name"], "base_multiplier": base["multiplier"],
            "base_reference": base["reference"], "members": member_records,
        })
        remaining.difference_update(members)
    if tuple(o["base"] for o in orbits) != SELECTED_REPRESENTATIVES:
        raise RuntimeError("Regenerated orbit order differs from the historical prepared example")
    reconstructed = sum(
        canonical(f"({o['base_multiplier']})*({rotate(o['base_reference'], k)})")
        for o in orbits for k in range(5)
    )
    if sp.expand(sum(weighted.values()) - reconstructed) != 0:
        raise RuntimeError("Cyclic orbit sum failed exact verification")
    return orbits, sorted(mapping, key=lambda record: record["component"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-amplitude", type=Path, default=DEFAULT_FULL,
                        help="Complete gluon5feyn CSV (optionally gzip compressed); not the 12345 seed")
    args = parser.parse_args(argv)
    raw = args.full_amplitude.read_bytes()
    decoded = gzip.decompress(raw) if args.full_amplitude.suffix == ".gz" else raw
    rows = list(csv.reader(decoded.decode("utf-8").splitlines()))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise ValueError("The full-amplitude CSV must contain exactly one identifier,expression row")
    full = algebra.parse(rows[0][1])
    print("Constructing and exactly verifying 80 projected components...", flush=True)
    components = construct_components(full)
    print("Grouping into 16 cyclic orbits and verifying their weighted sum...", flush=True)
    orbits, mapping = group_orbits(components)
    lookup = {c["name"]: c for c in components}
    prepared = {"method": METHOD, "components": [
        {key: lookup[name][key] for key in ("name", "multiplier", "expression")}
        for name in SELECTED_REPRESENTATIVES
    ]}
    manifest = {
        "component_count": len(components), "orbit_count": len(orbits),
        "gauge_references": GAUGE_REFERENCES,
        "projection": "e_i -> q_i.F_i/(q_i.p_i), q_i=p_(i+1 mod 5)",
        "identity": "A_full = sum(k=0..4) rho_k(sum(j=1..16) multiplier_j * reference_j)",
        "exact_component_expansions": True, "exact_projected_sum_minus_full": "0",
        "exact_orbit_sum_minus_projected_sum": "0",
        "representative_selection": "Fixed historical representatives selected using exploratory greedy predictions, preferring the shortest successful input in each orbit; not an unbiased benchmark",
        "reference_use": "Tensor references are preparation/proof data only; prepared_components.json contains scalar model inputs without references",
        "full_amplitude_sha256": hashlib.sha256(raw).hexdigest(),
        "sympy_version": sp.__version__, "orbits": orbits, "mapping": mapping,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in (("prepared_components.json", prepared), ("projected_components.json", components), ("orbit_manifest.json", manifest)):
        (args.output_dir / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "model_inputs.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows((c["name"], c["expression"]) for c in prepared["components"])
    print(f"Verified preparation written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
