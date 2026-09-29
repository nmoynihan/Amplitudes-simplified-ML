"""Reconstruct cyclic completion and the audited 1,247-token model input.

The compact basis below is the exact result of the historical sparse search,
not a model output. ``derive`` re-runs that search. Every preparation, including
the fast path, independently proves the compact basis against the completed
seed and the bundled full-amplitude reference before exporting scalar dots.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import sympy as sp

from .algebra import (DEFAULT_REFERENCE, DEFAULT_SEED, atom, on_shell, parse,
                      read_expression, render, require_zero, rotate, s, scalar,
                      symbolic, t, tokenizer, use_data_gen, ward)

EXPECTED_MODEL_READY_SHA256 = "f73c1e6b8459876394697b26d4f2585b6f51ed0eee52fa8412dd5ba9fbc6ca0c"
HISTORICAL_SOLUTION = {
    "indices": [290, 338, 402],
    "names": ["p3F12p3*p1F34p1", "p2F13p2*p1F24p1", "p2F14p2*p1F23p1"],
    "coefficients": ["-16/(s*t + t**2)", "16/(s*t)", "-16/(s**2 + s*t)"],
}
# Representatives of the rational coefficients after dividing the search
# target A*s*t by s*t. Keeping this original off-shell layout reproduces the
# actual checkpoint input. Their on-shell equality is checked below.
DENOMINATORS = {
    "p3F12p3*p1F34p1": "(p_1 · p_2)*(p_1 · p_4)*(p_2 · p_3)*(p_1 · p_3)",
    "p2F13p2*p1F24p1": "(p_1 · p_2)*(p_3 · p_4)*(p_1 · p_4)*(p_2 · p_3)",
    "p2F14p2*p1F23p1": "(p_1 · p_2)*(p_3 · p_4)*(p_1 · p_4)*(p_1 · p_3)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def compact_targets(derivation: Path | None = None):
    """Consume a searched solution, preserving the historical pole spelling."""
    solutions = [HISTORICAL_SOLUTION]
    if derivation is not None:
        data = json.loads(Path(derivation).read_text(encoding="utf-8"))
        solutions = data["solutions"] if isinstance(data, dict) else data
    if not isinstance(solutions, list) or not solutions:
        raise ValueError("Derivation must contain at least one exact basis solution")
    solution = next((item for item in solutions if set(item.get("names", [])) == set(DENOMINATORS)), None)
    if solution is None:
        raise ValueError("This historical export requires the three paired-chain solution; "
                         "the supplied search result contains a different basis")
    coefficients = solution.get("coefficients", [])
    if len(coefficients) != len(solution["names"]):
        raise ValueError("Basis names and coefficients must have equal lengths")
    by_name = dict(zip(solution["names"], coefficients))
    targets = []
    for name in HISTORICAL_SOLUTION["names"]:
        coefficient_text = by_name[name]
        if not isinstance(coefficient_text, str) or not re.fullmatch(r"[st0-9+*/()\s-]+", coefficient_text):
            raise ValueError("Basis coefficients must be exact rational functions of s and t")
        coefficient = sp.sympify(coefficient_text, locals={"s": s, "t": t})
        require_zero(coefficient / (s * t) - 1 / symbolic(DENOMINATORS[name]),
                     f"Pole representative for {name}")
        blocks = []
        for block in name.split("*"):
            match = re.fullmatch(r"p([1-4])F([1-4]+)p([1-4])", block)
            if not match:
                raise ValueError(f"Unsupported searched tensor name: {block}")
            left, labels, right = match.groups()
            blocks.append(" · ".join([f"p_{left}", *(f"F_{label}" for label in labels), f"p_{right}"]))
        targets.append(f"({'*'.join(blocks)})/({DENOMINATORS[name]})")
    return targets, solution


def prepare(output_dir: Path, seed: Path | None = None, reference: Path | None = None,
            derivation: Path | None = None) -> dict:
    """Write completed amplitudes, derived basis exports and verified model input.

    ``derivation`` accepts derive's sparse_extended_result.json or full manifest.
    There is no checkpoint access, Torch import or model call in this stage.
    """
    use_data_gen()
    from data_gen_ym.algebra import simplify_to_lowest_terms
    from data_gen_ym.expr_model import expand_simple_term
    from data_gen_ym.generate_clean_4pt import parenthesize_for_semantic_tokenization

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path, reference_path = Path(seed or DEFAULT_SEED), Path(reference or DEFAULT_REFERENCE)
    seed_text, reference_text = read_expression(seed_path), read_expression(reference_path)
    grouped = f"(({seed_text})+({rotate(seed_text)}))/2"
    completed, reference_value = scalar(grouped), symbolic(reference_text)
    require_zero(on_shell(completed) - reference_value, "Cyclic completion versus full reference")
    reduced = sp.expand(on_shell(completed)).subs({s: 2 * atom("p", 1, "p", 2),
                                                 t: 2 * atom("p", 1, "p", 4)})
    variants = {
        "gluon4feyn1234_completed": parenthesize_for_semantic_tokenization(grouped),
        "gluon4feyn1234_completed_expanded": render(sp.expand(completed)),
        "gluon4feyn1234_completed_reduced": render(sp.expand(reduced)),
    }
    tok = tokenizer()
    completion_manifest = {"operation": "(S1234 + S2341)/2", "changes_seed_function": True,
                           "equals_existing_gluon4feyn_on_shell": True, "variants": []}
    for name, expression in variants.items():
        value = scalar(expression)
        require_zero(on_shell(value) - reference_value, f"Completed variant {name}")
        for leg in range(1, 5):
            require_zero(ward(value, leg), f"Ward identity for {name}, leg {leg}")
        ids = tok.encode_infix(expression)
        require_zero(symbolic(tok.decode_infix(ids)) - reference_value, f"Token roundtrip for {name}")
        write_csv(output_dir / f"{name}.csv", [[1, expression]])
        write_csv(output_dir / f"{name}_tok.csv", [["id", "tokens"], [1, json.dumps(ids)]])
        completion_manifest["variants"].append({"name": name, "tokens": len(ids),
                                                 "ward_symbolic": ["0"] * 4,
                                                 "symbolic_difference_from_reference": "0"})
    write_json(output_dir / "completion_manifest.json", completion_manifest)

    targets, solution = compact_targets(derivation)
    expansions = [expand_simple_term(target) for target in targets]
    for target, expansion in zip(targets, expansions):
        require_zero(parse(target) - scalar(expansion), "Independent tensor expansion versus training expansion")
    full_target = " + ".join(f"({target})" for target in targets)
    full_expansion = " + ".join(f"({expression})" for expression in expansions)
    require_zero(symbolic(full_target) - reference_value, "Three-term basis versus full amplitude")
    require_zero(symbolic(full_expansion) - on_shell(completed), "Expanded basis versus completed seed")
    basis_dir = output_dir / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(basis_dir / "three_component_pairs.csv", [["simple", "scrambled"], *zip(targets, expansions)])
    write_csv(basis_dir / "three_component_inputs.csv", [[f"U{i + 1}", expression] for i, expression in enumerate(expansions)])
    write_csv(basis_dir / "three_term_full_input.csv", [["A_three_blocks", full_expansion]])
    (basis_dir / "three_term_target.txt").write_text(full_target + "\n", encoding="utf-8")
    write_json(basis_dir / "three_term_exact_proof.json",
               {"targets": targets, "sum_minus_full_exactly_zero": True,
                "independent_tensor_expansion_verified": True, "solution": solution,
                "derivation_source": str(Path(derivation).resolve()) if derivation else "historical sparse-search result in prepare.py"})

    source = parenthesize_for_semantic_tokenization(simplify_to_lowest_terms(full_expansion))
    if "F_" in source or "Tr" in source:
        raise ValueError("Model source must contain scalar p/e dot products only")
    require_zero(symbolic(source) - reference_value, "Model source versus full reference")
    source_ids = tok.encode_infix(source)
    require_zero(symbolic(tok.decode_infix(source_ids)) - reference_value, "Model-source token roundtrip")
    source_path = output_dir / "gluon4feyn1234_model_ready.csv"
    tokens_path = output_dir / "gluon4feyn1234_model_ready_tok.csv"
    write_csv(source_path, [[1, source]])
    write_csv(tokens_path, [["id", "tokens"], [1, json.dumps(source_ids)]])
    digest = sha256(source_path)
    if len(source_ids) != 1247:
        raise ValueError(f"Historical model input must have 1247 tokens; found {len(source_ids)}")
    if digest != EXPECTED_MODEL_READY_SHA256:
        raise ValueError(f"Prepared model-input bytes differ from the audit: {digest}")
    manifest = {
        "method": "cyclic seed completion, exactly verified sparse basis, scalar expansion and semantic tokenization",
        "seed": str(seed_path.resolve()), "seed_sha256": sha256(seed_path),
        "reference": str(reference_path.resolve()), "reference_sha256": sha256(reference_path),
        "completion": completion_manifest, "basis_source": str(Path(derivation).resolve()) if derivation else "historical_search_result",
        "basis_terms": 3, "basis_exact_difference": "0", "independent_tensor_expansion_verified": True,
        "source": str(source_path.resolve()), "source_tokens": str(tokens_path.resolve()),
        "input_tokens": len(source_ids), "source_sha256": digest,
        "historical_model_ready_sha256_matches": True, "tokenizer_exact_difference": "0",
        "exact_difference_from_completed_amplitude": "0", "model_called": False,
        "scope": "Specific four-gluon example; cyclic completion changes the original seed function.",
    }
    write_json(output_dir / "preparation_manifest.json", manifest)
    return manifest
