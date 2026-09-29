"""Reproduce the prepared 16-call simplification of the completed YM amplitude."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import sympy as sp

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from . import algebra as au
from .verification import compare, evaluate, make_points

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data/data_ym"
PREPARED = HERE / "prepared_components.json"
REFERENCE_CHECKPOINT_SHA256 = "b0918473800c014699752f9a1d1a9e080c18c1e2580ba4e1c977e6ada5203179"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_expression(path):
    """Read one headerless id,expression row from CSV or gzipped CSV."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 1 or len(rows[0]) != 2 or not rows[0][1].strip():
        raise ValueError(f"Expected exactly one nonempty id,expression row: {path}")
    return rows[0][1].strip()


def rotate(expression, shift):
    return re.sub(
        r"\b([peF])_([1-5])\b",
        lambda m: f"{m[1]}_{(int(m[2]) - 1 + shift) % 5 + 1}", expression,
    )


def weighted(expression, multiplier):
    coefficient = sp.Rational(multiplier)
    # The prefix tokenizer joins adjacent digit tokens into multi-digit numbers.
    # Separating the denominator's digit with two unary operators preserves 2
    # when a prediction ends in a power, while remaining ordinary arithmetic.
    return f"(({coefficient.p})*({expression}))/(-(-{coefficient.q}))"


def reconstruct(predictions):
    core = " + ".join(
        "(" + weighted(row["prediction"], row["multiplier"]) + ")"
        for row in predictions
    )
    return core, " + ".join("(" + rotate(core, k) + ")" for k in range(5))


def require_zero(expression, description):
    if au.on_shell(expression) != 0:
        raise RuntimeError(f"Exact verification failed: {description}")


def token_roundtrip(expression, tokenizer, description):
    tokens = tokenizer.encode_infix(expression)
    if any(token in (0, 1, 2, 3) for token in tokens):
        raise ValueError(f"Unsupported or special token: {description}")
    require_zero(
        au.parse(tokenizer.decode_infix(tokens)) - au.parse(expression),
        f"{description} tokenizer roundtrip",
    )
    return tokens


def load_cases(path):
    cases = json.loads(Path(path).read_text())["components"]
    if len(cases) != 16 or len({row["name"] for row in cases}) != 16:
        raise ValueError("This workflow requires 16 uniquely named prepared components")
    for case in cases:
        if "F_" in case["expression"] or "Tr" in case["expression"]:
            raise ValueError("Model inputs must contain only scalar p/e contractions")
        if sp.Rational(case["multiplier"]) not in (sp.Rational(1, 2), sp.Rational(-1, 2)):
            raise ValueError("Expected the external coefficient +1/2 or -1/2")
    return cases


def verify_sources(cases, seed_text, full_text, tokenizer):
    seed, full = au.parse(seed_text), au.parse(full_text)
    require_zero(sum(au.cyclic(seed, k) for k in range(5)) - full, "cyclic seed completion")
    for leg in range(1, 6):
        require_zero(au.ward(full, leg), f"complete amplitude Ward identity, leg {leg}")
    sources = []
    for case in cases:
        source = au.parse(case["expression"])
        if au.on_shell(source) == 0:
            raise RuntimeError(f"{case['name']}: prepared input is identically zero")
        token_roundtrip(case["expression"], tokenizer, case["name"])
        sources.append(source)
    core = sum(sp.Rational(c["multiplier"]) * s for c, s in zip(cases, sources))
    completed = sum(au.cyclic(core, k) for k in range(5))
    require_zero(completed - full, "prepared inputs reconstruct full amplitude")
    return full, sources, completed


def run(args):
    started = time.monotonic()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("Use a new or empty output directory to avoid mixing run artifacts")
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
    cases = load_cases(args.components)
    seed_text, full_text = read_expression(args.seed_csv), read_expression(args.full_csv)
    full, sources, completed_sources = verify_sources(cases, seed_text, full_text, tokenizer)
    print("Cyclic completion, full Ward identities and prepared input identity verified.", flush=True)
    report = {
        "method": "16 scalar inputs; 16 greedy calls; weighted sum and five cyclic images",
        "changes_original_seed": True,
        "seed_completion": "sum of five cyclic seed images; no averaging",
        "assumptions": ["p_i^2=0", "sum_i p_i=0", "e_i.p_i=0"],
        "dimension_specific_identities_used": False,
        "artifacts": {key: {"path": str(path.resolve()), "sha256": digest(path)}
                      for key, path in (("seed", args.seed_csv), ("full", args.full_csv),
                                        ("prepared_components", args.components))},
        "exact_source_reconstruction_difference": "0",
        "full_ward_residuals": ["0"] * 5,
        "representative_selection": "Fixed representatives selected using exploratory inference before this reproduction",
        "limitation": "A prepared-input solution for this completed amplitude; not a general automatic simplifier or unbiased accuracy benchmark.",
        "verify_only": args.verify_only,
    }
    predictions = []
    if not args.verify_only:
        from .inference import FivePointPredictor
        predictor = FivePointPredictor(args.checkpoint, args.threads)
        for index, (case, source) in enumerate(zip(cases, sources), 1):
            row = predictor.predict(case["expression"], args.max_output_tokens)
            require_zero(au.parse(row["prediction"]) - source, f"{case['name']} actual prediction")
            row.update(name=case["name"], multiplier=case["multiplier"],
                       output_tokens=len(row["prediction_tokens"]), exact_difference_from_input="0")
            predictions.append(row)
            print(f"{index:2d}/16 {case['name']}: {row['input_tokens']} -> {row['output_tokens']} tokens; exact pass", flush=True)
        core_text, completed_text = reconstruct(predictions)
        completed = au.parse(completed_text)
        require_zero(completed - full, "reconstructed model output vs complete amplitude")
        core_ids = token_roundtrip(core_text, tokenizer, "exported cyclic core")
        completed_ids = token_roundtrip(completed_text, tokenizer, "exported completed amplitude")
        checkpoint_hash = digest(args.checkpoint)
        report.update(
            checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
            matches_reference_checkpoint=checkpoint_hash == REFERENCE_CHECKPOINT_SHA256,
            checkpoint_epoch=predictor.epoch, model_calls=len(predictions), decoding="greedy",
            cpu_threads=args.threads, maximum_decode_length=args.max_output_tokens,
            maximum_input_tokens_including_bos_eos=max(p["input_tokens"] + 2 for p in predictions),
            total_model_input_content_tokens=sum(p["input_tokens"] for p in predictions),
            total_model_output_content_tokens=sum(p["output_tokens"] for p in predictions),
            original_seed_content_tokens=len(tokenizer.encode_infix(seed_text)),
            original_completed_content_tokens=len(tokenizer.encode_infix(full_text)),
            rendered_cyclic_core_content_tokens=len(core_ids),
            rendered_completed_prediction_content_tokens=len(completed_ids),
            exact_model_output_reconstruction_difference="0", export_token_roundtrip_difference="0",
            predictions=predictions,
        )
    else:
        completed = completed_sources
        report.update(model_calls=0)

    points, kinematic_error = make_points(args.numeric_seed, args.samples_per_mode)
    numerical = compare(evaluate(completed, points), evaluate(full, points))
    if not numerical["passes"]:
        raise RuntimeError(f"Independent numerical checks failed: {numerical}")
    report.update(
        independent_numeric_points=len(points), independent_numeric_seed=args.numeric_seed,
        samples_per_mode=args.samples_per_mode, numerical_modes=["coulomb", "covariant"],
        numerical_polarizations="independent random transverse vectors; covariant mode adds random p_i shifts",
        maximum_kinematic_constraint_error=kinematic_error,
        numerical_comparison=numerical, numeric_atol=1e-9, numeric_rtol=1e-8,
        total_seconds=time.monotonic() - started,
        sympy_version=sp.__version__,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "model_ready_components.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows((c["name"], c["expression"]) for c in cases)
    if predictions:
        for name, expression in (
            ("gluon5feyn12345_completed_simplified", completed_text),
            ("gluon5feyn12345_gauge_invariant_cyclic_core", core_text),
        ):
            with (args.output_dir / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([1, expression])
        with (args.output_dir / "component_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "multiplier", "prediction"])
            writer.writerows((p["name"], p["multiplier"], p["prediction"]) for p in predictions)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['model_calls']} model calls; {len(points)} independent numerical checks passed.", flush=True)
    print(args.output_dir / "results.json", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="External five-point YM best_model.pt")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--components", type=Path, default=PREPARED)
    parser.add_argument("--seed-csv", type=Path, default=DATA / "gluon5feyn12345.csv.gz")
    parser.add_argument("--full-csv", type=Path, default=DATA / "gluon5feyn.csv.gz")
    parser.add_argument("--verify-only", action="store_true", help="Verify prepared inputs without running a model")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Decoder max_length (including BOS)")
    parser.add_argument("--numeric-seed", type=int, default=310921)
    parser.add_argument("--samples-per-mode", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.verify_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --verify-only is used")
    if args.samples_per_mode < 1 or args.threads < 1 or args.max_output_tokens < 2:
        parser.error("Positive samples/threads and at least two decoder tokens are required")
    run(args)


if __name__ == "__main__":
    main()
