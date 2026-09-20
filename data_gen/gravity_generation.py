#!/usr/bin/env python3
"""Generate gravity training data and a separate, audited benchmark test set.

Run ``python -m data_gen.gravity_generation --help`` from the repository root,
or execute this file directly from any directory. The default mode is ``data``;
``train``, ``eval``, ``full`` and ``smoke`` mirror run_gravity_100k.sh.

The benchmark comprises scrambles of two reserved paper amplitudes. It is never
passed to the trainer, including its validation split. Family exclusion uses
the generator's structural signatures (including particle relabelings), not a
proof of arbitrary algebraic inequivalence. Actual train/test expressions are
also required to be disjoint after tokenization, on either side of each pair.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_gen.data_gen_gravity.core import is_benchmark_leak
from data_gen.data_gen_gravity.generate import (
    Candidate,
    _quota,
    build_benchmarks,
    build_dataset_parallel,
    tokenise,
    write_metadata,
    write_raw,
)
from data_gen.data_gen_ym.generate_clean_4pt import _publish_all


class HoldoutGuard:
    """Reserve benchmark expressions and the compact origins of staged rows."""

    def __init__(self, benchmarks: Sequence[Candidate], max_tokens: int):
        if not benchmarks:
            raise ValueError("The held-out benchmark set must not be empty")
        self.tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8, max_sequence_length=max_tokens
        )
        self.expressions = {
            tuple(self.tokenizer.encode_infix(expression))
            for row in benchmarks
            for expression in (row.simple, row.scrambled)
        }

    def rejection_reason(self, row: Candidate) -> str | None:
        if not row.compact_origin:
            raise ValueError("Training candidate is missing its compact origin")
        if is_benchmark_leak(row.compact_origin, row.process):
            return "benchmark_family"
        for expression in (row.simple, row.scrambled):
            if tuple(self.tokenizer.encode_infix(expression)) in self.expressions:
                return "benchmark_expression"
        return None


def build_training_set(
    samples: int,
    benchmarks: Sequence[Candidate],
    *,
    jobs: int,
    seed: int,
    max_tokens: int,
    min_scr: int = 1,
    max_scr: int = 5,
    min_terms: int = 1,
    max_terms: int = 3,
    max_refill_rounds: int = 20,
) -> tuple[list[Candidate], dict[str, int]]:
    """Filter against the reserved test set, then refill exact balanced quotas."""
    guard = HoldoutGuard(benchmarks, max_tokens)
    desired = {(p, k): n for p, k, n in _quota(samples, "mixed", "mixed")}
    counts: Counter = Counter()
    rejected: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    output: list[Candidate] = []
    kwargs = dict(
        jobs=jobs, seed=seed, max_tokens=max_tokens,
        min_scr=min_scr, max_scr=max_scr, min_terms=min_terms,
        max_terms=max_terms, validate=True,
    )

    def accept(rows: Iterable[Candidate]) -> None:
        for row in rows:
            cell = (row.process, row.kind)
            if cell not in desired or counts[cell] >= desired[cell]:
                continue
            reason = guard.rejection_reason(row)
            if reason:
                rejected[reason] += 1
                continue
            key = (row.simple, row.scrambled)
            if key in seen:
                rejected["duplicate_pair"] += 1
                continue
            seen.add(key)
            output.append(row)
            counts[cell] += 1

    accept(build_dataset_parallel(samples, process="mixed", kind="mixed", **kwargs))
    for refill_round in range(max_refill_rounds):
        if all(counts[cell] == target for cell, target in desired.items()):
            break
        for cell_index, (cell, target) in enumerate(desired.items()):
            missing = target - counts[cell]
            if missing:
                refill = dict(kwargs)
                # Separate deterministic streams from the initial worker seeds.
                refill["seed"] = (
                    seed + 10**15 + refill_round * 10**10 + cell_index * 10**8
                )
                accept(build_dataset_parallel(
                    missing, process=cell[0], kind=cell[1], **refill
                ))
    if len(output) != samples or any(counts[c] != n for c, n in desired.items()):
        raise RuntimeError(
            f"Could not fill training quotas after holdout filtering: {dict(counts)}"
        )
    return output, dict(rejected)


def _read_token_rows(
    path: Path, max_tokens: int,
) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=8, max_sequence_length=max_tokens
    )
    expression_ids = set(tokenizer.vocab.values()) - {
        tokenizer.vocab[name]
        for name in ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "(", ")")
    }
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["simple", "scrambled"]:
            raise ValueError(f"Expected simple,scrambled columns in {path}")
        for line, row in enumerate(reader, 2):
            pair = []
            for column in ("simple", "scrambled"):
                values = json.loads(row[column])
                if (
                    not isinstance(values, list) or not values
                    or len(values) > max_tokens
                    or any(type(v) is not int or v not in expression_ids for v in values)
                ):
                    raise ValueError(f"Invalid token list in {path}:{line} ({column})")
                pair.append(tuple(values))
            yield pair[0], pair[1]


def audit_token_files(
    train_path: Path, test_path: Path, *, max_tokens: int = 4096,
) -> dict[str, int]:
    """Fail if any input OR target is shared, including across opposite columns."""
    reserved: set[tuple[int, ...]] = set()
    test_rows = 0
    for pair in _read_token_rows(test_path, max_tokens):
        reserved.update(pair)
        test_rows += 1
    if not test_rows:
        raise ValueError("The benchmark test file is empty")
    train_rows = 0
    for pair in _read_token_rows(train_path, max_tokens):
        train_rows += 1
        if any(expression in reserved for expression in pair):
            raise ValueError(f"Train/test expression overlap at training row {train_rows}")
    if not train_rows:
        raise ValueError("The training file is empty")
    return {"training_rows": train_rows, "test_rows": test_rows, "shared_expressions": 0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "output": (args.output_dir / "benchmark_evaluation.csv.gz").resolve(),
        "summary": (args.output_dir / "benchmark_evaluation_summary.json").resolve(),
    }


def _validate_output_paths(paths: Sequence[Path]) -> None:
    if len(set(paths)) != len(paths):
        raise ValueError("All training, benchmark, manifest, and evaluation output paths must differ")
    for path in paths:
        if path.exists() and not path.is_file():
            raise ValueError(f"Output destination is not a regular file: {path}")
        if any(other in path.parents for other in paths):
            raise ValueError("An output file cannot be the parent of another output")
        if any(parent.exists() and not parent.is_dir() for parent in path.parents):
            raise ValueError(f"Output parent is not a directory: {path}")
    existing = [path for path in paths if path.exists()]
    for index, path in enumerate(existing):
        if any(os.path.samefile(path, other) for other in existing[index + 1:]):
            raise ValueError("Output paths must not alias the same file")


def dataset_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir.resolve()
    if args.mode == "smoke":
        output_dir /= "smoke"
    stem = "smoke" if args.mode == "smoke" else "gravity_5pt_100k"
    defaults = {
        "raw_out": f"{stem}_raw.csv.gz",
        "tok_out": f"{stem}_tok.csv.gz",
        "metadata_out": f"{stem}_metadata.csv.gz",
        "bench_raw": "benchmarks_raw.csv.gz",
        "bench_tok": "benchmarks_tok.csv.gz",
        "bench_metadata": "benchmarks_metadata.csv.gz",
        "manifest_out": "generation_manifest.json",
    }
    paths = {
        key: (getattr(args, key) or output_dir / name).resolve()
        for key, name in defaults.items()
    }
    evaluation = _evaluation_paths(args) if args.mode in ("eval", "full") else {}
    _validate_output_paths((*paths.values(), *evaluation.values()))
    return paths


def generate_datasets(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; choose --output-dir or use --overwrite: "
            + ", ".join(existing)
        )
    print("Generating the reserved benchmark test set...", flush=True)
    benchmarks = build_benchmarks(
        scrambles_per_amplitude=args.benchmark_samples,
        seed=args.benchmark_seed, max_tokens=args.max_tokens, validate=True,
    )
    print(f"Generating {args.samples:,} training pairs with benchmark exclusion...", flush=True)
    training, rejected = build_training_set(
        args.samples, benchmarks, jobs=args.jobs, seed=args.seed,
        max_tokens=args.max_tokens, min_scr=args.min_scr, max_scr=args.max_scr,
        min_terms=args.min_terms, max_terms=args.max_terms,
    )
    # Stage and audit every artifact, then publish with rollback on failure.
    # Keep the manifest last so it describes a fully published release.
    staged: dict[str, Path] = {}
    try:
        for key, destination in paths.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=f".{destination.stem}.", suffix=destination.suffix,
                dir=destination.parent,
            )
            os.close(fd)
            staged[key] = Path(name)
        for rows, raw, tok, metadata in (
            (training, "raw_out", "tok_out", "metadata_out"),
            (benchmarks, "bench_raw", "bench_tok", "bench_metadata"),
        ):
            write_raw(rows, staged[raw])
            tokenise(rows, staged[tok], max_tokens=args.max_tokens)
            write_metadata(rows, staged[metadata])
        audit = audit_token_files(
            staged["tok_out"], staged["bench_tok"], max_tokens=args.max_tokens
        )
        if (audit["training_rows"] != args.samples
                or audit["test_rows"] != 2 * args.benchmark_samples):
            raise RuntimeError("Serialized dataset row counts do not match the requested sizes")
        manifest = {
            "schema_version": 1,
            "generator": "data_gen.gravity_generation",
            "training": {
                "rows": len(training), "seed": args.seed, "jobs": args.jobs,
                "counts": dict(Counter(f"{r.process}/{r.kind}" for r in training)),
                "min_scr": args.min_scr, "max_scr": args.max_scr,
                "min_terms": args.min_terms, "max_terms": args.max_terms,
            },
            "test": {
                "role": "held-out paper benchmarks", "rows": len(benchmarks),
                "seed": args.benchmark_seed,
                "rows_per_amplitude": args.benchmark_samples,
                "independent_amplitudes": 2,
            },
            "max_tokens": args.max_tokens,
            "holdout": {
                **audit, "rejected_training_candidates": rejected,
                "family_check": "compact-origin structural signatures and species-preserving relabelings",
                "expression_check": "token identity across both columns of both splits",
                "algebraic_inequivalence_proven": False,
            },
            "files": {
                key: {"path": str(paths[key]), "sha256": _sha256(staged[key])}
                for key in paths if key != "manifest_out"
            },
        }
        staged["manifest_out"].write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_all(tuple(staged.values()), tuple(paths.values()), overwrite=args.overwrite)
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
    print(
        f"Wrote {len(training):,} training and {len(benchmarks):,} test pairs; "
        "shared expressions: 0.", flush=True,
    )
    print(f"Manifest: {paths['manifest_out']}", flush=True)


def verify_saved_datasets(paths: dict[str, Path]) -> dict:
    """Recheck actual trainer inputs and require their generation audit."""
    manifest_path = paths["manifest_out"]
    if not manifest_path.is_file():
        raise ValueError(
            f"Missing generation manifest: {manifest_path}. Run data mode first "
            "to generate audited datasets (use --output-dir for a new directory)."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("generator") != "data_gen.gravity_generation"):
        raise ValueError(f"Unrecognized generation manifest: {manifest_path}")
    for key, path in paths.items():
        if key == "manifest_out":
            continue
        recorded = manifest["files"][key]
        if _sha256(path) != recorded["sha256"]:
            raise ValueError(f"Dataset changed since its holdout audit: {path}")
    audit = audit_token_files(
        paths["tok_out"], paths["bench_tok"], max_tokens=manifest["max_tokens"]
    )
    if (audit["training_rows"] != manifest["training"]["rows"]
            or audit["test_rows"] != manifest["test"]["rows"]):
        raise ValueError("Saved row counts do not match the generation manifest")
    return manifest


def training_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    command = [
        sys.executable, str(REPO_ROOT / "transformer/transformer_trainer.py"),
        "--run-name", args.run_name, "--data-files", str(paths["tok_out"]),
        "--max-length", str(args.max_tokens + 2), "--dynamic-padding", "--bucketing",
    ]
    settings = {
        "epochs": args.epochs, "batch-size": args.batch_size,
        "gradient-accumulation-steps": args.grad_accum_steps,
        "train-split": args.train_split, "embedding-dim": args.embedding_dim,
        "n-heads": args.n_heads, "n-enc-layers": args.n_enc_layers,
        "n-dec-layers": args.n_dec_layers, "head-ff-dim": args.head_ff_dim,
        "dropout": args.dropout, "learning-rate": args.learning_rate,
        "grad-clip": args.grad_clip, "label-smoothing": args.label_smoothing,
        "bucket-size-multiplier": args.bucket_size_multiplier,
        "num-workers": args.num_workers, "amp-dtype": args.amp_dtype,
    }
    for flag, value in settings.items():
        command.extend((f"--{flag}", str(value)))
    command.extend((
        "--amp" if args.amp else "--no-amp",
        "--pin-memory" if args.pin_memory else "--no-pin-memory",
    ))
    return command


def evaluation_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    evaluation = _evaluation_paths(args)
    # Also protect callers constructing an evaluation command directly.
    _validate_output_paths((*paths.values(), *evaluation.values()))
    model_path = (
        args.model_path or REPO_ROOT / "models" / args.run_name / "best_model.pt"
    ).resolve()
    return [
        sys.executable, "-m", "data_gen.data_gen_gravity.evaluate",
        "--model-path", str(model_path),
        "--raw", str(paths["bench_raw"]), "--tokenized", str(paths["bench_tok"]),
        "--metadata", str(paths["bench_metadata"]), "--max-output-tokens", "512",
        "--decoding-method", "beam", "--beam-size", str(args.beam_size),
        "--output", str(evaluation["output"]),
        "--summary", str(evaluation["summary"]),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode", nargs="?", default="data",
        choices=("data", "train", "eval", "full", "smoke"),
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/gravity")
    for flag, env in (
        ("raw-out", "RAW_OUT"), ("tok-out", "TOK_OUT"), ("metadata-out", "META_OUT"),
        ("bench-raw", "BENCH_RAW"), ("bench-tok", "BENCH_TOK"),
        ("bench-metadata", "BENCH_META"), ("manifest-out", "MANIFEST_OUT"),
    ):
        parser.add_argument(f"--{flag}", type=Path, default=os.environ.get(env))
    integer_options = {
        "samples": (100000, "SAMPLES"), "seed": (42, "SEED"),
        "jobs": (8, "JOBS"), "max-tokens": (4096, "MAX_TOKENS"),
        "benchmark-samples": (100, "BENCHMARK_SAMPLES"),
        "benchmark-seed": (240804720, "BENCHMARK_SEED"),
        "min-scr": (1, "MIN_SCR"), "max-scr": (5, "MAX_SCR"),
        "min-terms": (1, "MIN_TERMS"), "max-terms": (3, "MAX_TERMS"),
        "epochs": (60, "EPOCHS"), "batch-size": (2, "BATCH_SIZE"),
        "grad-accum-steps": (12, "GRAD_ACCUM_STEPS"), "beam-size": (8, "BEAM_SIZE"),
        "embedding-dim": (512, "EMBEDDING_DIM"), "n-heads": (8, "N_HEADS"),
        "n-enc-layers": (6, "N_ENC_LAYERS"), "n-dec-layers": (6, "N_DEC_LAYERS"),
        "head-ff-dim": (2048, "HEAD_FF_DIM"),
        "bucket-size-multiplier": (100, "BUCKET_SIZE_MULTIPLIER"),
        "num-workers": (2, "NUM_WORKERS"),
    }
    for flag, (default, env) in integer_options.items():
        parser.add_argument(f"--{flag}", type=int, default=os.environ.get(env, default))
    for flag, default, env in (
        ("train-split", 0.9, "TRAIN_SPLIT"), ("dropout", 0.1, "DROPOUT"),
        ("learning-rate", 3e-4, "LEARNING_RATE"), ("grad-clip", 1.0, "GRAD_CLIP"),
        ("label-smoothing", 0.1, "LABEL_SMOOTHING"),
    ):
        parser.add_argument(f"--{flag}", type=float, default=os.environ.get(env, default))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "gravity_5pt_mixed_100k"))
    parser.add_argument("--model-path", type=Path, default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=os.environ.get("AMP_DTYPE", "bf16"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing dataset outputs after successful generation and audit",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "smoke":
        # Set mode defaults, then parse again so explicit CLI overrides still win.
        parser.set_defaults(
            samples=os.environ.get("SMOKE_SAMPLES", 16), benchmark_samples=5,
            jobs=1, epochs=1, grad_accum_steps=1, batch_size=4,
            run_name=os.environ.get("SMOKE_RUN_NAME", "gravity_smoke"),
            embedding_dim=32, n_heads=4, n_enc_layers=1, n_dec_layers=1,
            head_ff_dim=64, dropout=0.0, num_workers=0, train_split=0.75,
            amp=False, pin_memory=False,
        )
        args = parser.parse_args(argv)
    positive = (
        "samples", "jobs", "max_tokens", "benchmark_samples", "min_scr", "max_scr",
        "min_terms", "max_terms", "epochs", "batch_size", "grad_accum_steps",
        "beam_size", "embedding_dim", "n_heads", "n_enc_layers", "n_dec_layers",
        "head_ff_dim", "bucket_size_multiplier",
    )
    for field in positive:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.benchmark_samples % 5:
        parser.error("--benchmark-samples is per amplitude and must be divisible by 5")
    if args.seed == args.benchmark_seed:
        parser.error("--seed and --benchmark-seed must differ")
    if args.min_scr > args.max_scr or args.min_terms > args.max_terms:
        parser.error("minimum scramble depth / term count must not exceed the maximum")
    if not 0 < args.train_split < 1:
        parser.error("--train-split must be between 0 and 1; it applies only to the main dataset")
    if args.num_workers < 0 or args.embedding_dim % args.n_heads:
        parser.error("--num-workers must be non-negative; --embedding-dim must be divisible by --n-heads")
    for field in ("dropout", "label_smoothing"):
        value = getattr(args, field)
        if not math.isfinite(value) or not 0 <= value < 1:
            parser.error(f"--{field.replace('_', '-')} must be between 0 (inclusive) and 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error("--grad-clip must be finite and non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = dataset_paths(args)
    if args.mode in ("data", "full", "smoke"):
        generate_datasets(args, paths)
    if args.mode in ("train", "eval", "full", "smoke"):
        manifest = verify_saved_datasets(paths)
        if args.mode != "eval" and args.max_tokens < manifest["max_tokens"]:
            raise ValueError(
                "--max-tokens must cover the saved dataset's token cap "
                "to prevent training truncation"
            )
        env = os.environ.copy()
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        if args.mode == "smoke":
            env["CUDA_VISIBLE_DEVICES"] = ""
        if args.mode in ("train", "full", "smoke"):
            subprocess.run(training_command(args, paths), cwd=REPO_ROOT, env=env, check=True)
        if args.mode in ("eval", "full"):
            subprocess.run(evaluation_command(args, paths), cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
