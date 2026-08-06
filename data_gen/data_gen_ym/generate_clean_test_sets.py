#!/usr/bin/env python3
"""Generate held-out corrected Yang--Mills test sets at four and five points.

By default this launcher creates 200 4PT pairs and 200 5PT pairs.  It uses
fixed test-only seeds, the same strict cleaning/validation pipeline as the
training generators, and writes raw/tokenized CSVs, per-set reports, and one
combined manifest under ``data/data_ym/heldout``.  ``evaluate_model.py`` should
receive the raw expression CSV, not the ``_tok`` training/audit companion.

Run from the repository root::

    python -m data_gen.data_gen_ym.generate_clean_test_sets
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from . import generate_clean_4pt as _core


DEFAULT_SAMPLES = 200
DEFAULT_4PT_SEED = 4_020_004
DEFAULT_5PT_SEED = 5_020_005
DEFAULT_CANDIDATE_BATCH_SIZE = 400
DEFAULT_GENERATOR_BATCH_SIZE = 100
DEFAULT_OUTPUT_DIR = _core.REPO_ROOT / "data" / "data_ym" / "heldout"


def output_paths(
    output_dir: Path,
    *,
    n_particles: int,
    samples: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    stem = (
        f"ym_{n_particles}pt_heldout_{samples}_seed{seed}_canonical_nonzero"
    )
    return (
        output_dir / f"{stem}.csv.gz",
        output_dir / f"{stem}_tok.csv.gz",
        output_dir / f"{stem}.report.json",
    )


def default_manifest_path(
    output_dir: Path,
    *,
    samples: int,
    seed_4pt: int,
    seed_5pt: int,
) -> Path:
    return output_dir / (
        f"ym_4pt_5pt_heldout_{samples}_seeds{seed_4pt}_{seed_5pt}_manifest.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate corrected held-out Yang--Mills test pairs for both "
            "four-point and five-point models."
        )
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed-4pt", type=int, default=DEFAULT_4PT_SEED)
    parser.add_argument("--seed-5pt", type=int, default=DEFAULT_5PT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--jobs", default="auto")
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=DEFAULT_CANDIDATE_BATCH_SIZE,
    )
    parser.add_argument(
        "--generator-batch-size",
        type=int,
        default=DEFAULT_GENERATOR_BATCH_SIZE,
    )
    parser.add_argument("--max-candidates-factor", type=float, default=5.0)
    parser.add_argument("--zero-checks", type=int, default=3)
    parser.add_argument("--validation-checks", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-tokens-4pt", type=int, default=2048)
    parser.add_argument("--max-tokens-5pt", type=int, default=4096)
    parser.add_argument(
        "--only",
        choices=("both", "4pt", "5pt"),
        default="both",
        help="generate both test sets by default, or select one for debugging",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_fields = (
        "samples",
        "candidate_batch_size",
        "generator_batch_size",
        "zero_checks",
        "validation_checks",
        "max_tokens_4pt",
        "max_tokens_5pt",
    )
    for field in positive_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if (
        args.max_candidates_factor < 1
        or not math.isfinite(args.max_candidates_factor)
    ):
        parser.error("--max-candidates-factor must be finite and at least 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.seed_4pt == args.seed_5pt:
        parser.error("--seed-4pt and --seed-5pt must differ")


def _selected_points(selection: str) -> tuple[int, ...]:
    if selection == "4pt":
        return (4,)
    if selection == "5pt":
        return (5,)
    return (4, 5)


def _generation_args(
    launcher_args: argparse.Namespace,
    *,
    n_particles: int,
    raw_path: Path,
    token_path: Path,
    report_path: Path,
) -> argparse.Namespace:
    generation_args = _core.build_parser(
        n_particles=n_particles,
    ).parse_args([])
    generation_args.samples = launcher_args.samples
    generation_args.seed = (
        launcher_args.seed_4pt
        if n_particles == 4
        else launcher_args.seed_5pt
    )
    generation_args.jobs = launcher_args.jobs
    generation_args.candidate_batch_size = launcher_args.candidate_batch_size
    generation_args.generator_batch_size = launcher_args.generator_batch_size
    generation_args.max_candidates_factor = launcher_args.max_candidates_factor
    generation_args.zero_checks = launcher_args.zero_checks
    generation_args.validation_checks = launcher_args.validation_checks
    generation_args.progress_every = launcher_args.progress_every
    generation_args.max_tokens = (
        launcher_args.max_tokens_4pt
        if n_particles == 4
        else launcher_args.max_tokens_5pt
    )
    generation_args.raw_out = str(raw_path)
    generation_args.tok_out = str(token_path)
    generation_args.report_out = str(report_path)
    generation_args.overwrite = launcher_args.overwrite
    return generation_args


def _write_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    temporary = _core._temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _core._publish(temporary, path, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_test_sets(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path]:
    output_dir = args.output_dir.expanduser()
    selected = _selected_points(args.only)
    manifest_path = (
        args.manifest_out.expanduser()
        if args.manifest_out is not None
        else default_manifest_path(
            output_dir,
            samples=args.samples,
            seed_4pt=args.seed_4pt,
            seed_5pt=args.seed_5pt,
        )
    )

    paths_by_point = {
        n_particles: output_paths(
            output_dir,
            n_particles=n_particles,
            samples=args.samples,
            seed=args.seed_4pt if n_particles == 4 else args.seed_5pt,
        )
        for n_particles in selected
    }
    all_destinations = [
        destination
        for paths in paths_by_point.values()
        for destination in paths
    ] + [manifest_path]
    resolved = [path.resolve(strict=False) for path in all_destinations]
    if len(resolved) != len(set(resolved)):
        raise ValueError("test CSV, report, and manifest paths must be distinct")
    if not args.overwrite:
        existing = [path for path in all_destinations if path.exists()]
        if existing:
            raise FileExistsError(
                "test output already exists (use --overwrite to replace): "
                + ", ".join(str(path) for path in existing)
            )

    set_reports: dict[str, Any] = {}
    for n_particles in selected:
        raw_path, token_path, report_path = paths_by_point[n_particles]
        generation_args = _generation_args(
            args,
            n_particles=n_particles,
            raw_path=raw_path,
            token_path=token_path,
            report_path=report_path,
        )
        stats, report = _core.generate_to_files(
            generation_args,
            n_particles=n_particles,
            generator_name=f"clean_{n_particles}pt_yang_mills_test",
        )
        if stats.accepted != args.samples:
            raise RuntimeError(
                f"{n_particles}PT generator returned {stats.accepted}/"
                f"{args.samples} requested test rows"
            )
        set_reports[f"{n_particles}pt"] = {
            "seed": generation_args.seed,
            "raw_output": report["outputs"]["raw"],
            "tokenized_output": report["outputs"]["tokenized"],
            "report_path": str(report_path.resolve()),
            "stats": report["stats"],
        }

    manifest = {
        "manifest_schema_version": 1,
        "dataset_role": "held_out_test",
        "evaluate_model_input": "raw_output",
        "evaluate_model_existing_csv_max_rows": args.samples,
        "samples_per_point": args.samples,
        "selected_points": list(selected),
        "sets": set_reports,
    }
    _write_manifest(
        manifest_path,
        manifest,
        overwrite=args.overwrite,
    )
    return manifest, manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    try:
        manifest, manifest_path = generate_test_sets(args)
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote combined test-set manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
