#!/usr/bin/env python3
"""Generate canonical, numerically nonzero cyclic Yang--Mills training pairs.

Defaults use the external-order conventions of the reference files
``data/data_ym/gluon4feyn1234.csv.gz`` (1,2,3,4) and
``data/data_ym/gluon5feyn12345.csv.gz`` (1,2,3,4,5). Those files contain expanded
dot products, not compact F-blocks. Their filenames specify the external cycles;
the per-block invariant below uses those same cycles. These defaults are defined
in ``notation.default_cyclic_order`` and recorded in logs and split manifests.

Each trace or open chain contains field-strength labels in a forward cyclic
subsequence of (1, ..., N): labels may be omitted and the sequence may wrap
around N to 1, but it may not permute or reverse that order. Each external
field strength occurs exactly once across all blocks of a numerator monomial.
The invariant applies to each block independently, not to the concatenation
of commuting scalar factors. Open-chain momentum endpoints are contractions,
not additional field strengths, and retain the original endpoint sampling.

The CLI uses :mod:`generate_clean_4pt`'s strict cleaning and publication engine,
with cyclic-aware canonicalization. Equivalent trace and chain representations
are chosen only when their F labels retain the forward cyclic order. Exact
zeros and numerical zero subsets are removed, nonzero targets are checked on
deterministic kinematic grids, and target/source equivalence is checked on an
independent grid. Raw expressions are parenthesized before tokenization.

Rejected candidates and duplicate cleaned pairs are replaced until exactly
``--samples`` pairs have been accepted. If ``--max-candidates-factor`` is
exhausted, generation fails without publishing a partial pool. The pool and its
JSON report are published together; existing files require ``--overwrite``.
The programmatic ``build_dataset`` helpers below remain candidate builders.

Physical denominator channels already obey cyclic adjacency, including (N, 1);
the treatment of cancellable denominator factors is unchanged. Both supported
dataset modes (``oneshot`` and ``step``) use the existing Yang--Mills pair builder.

The existing N >= 6 limitation remains: genuine multiparticle poles
(p_i + ... + p_j)^2 are not modelled by the scalar-product pole representation.
Numerator ordering does not make this a complete N >= 6 physical pole model.

From the repository root::

    python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 4 --samples 1000 --jobs 1
    python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 5 --samples 1000 --jobs 1

The historical ``python -m data_gen_ym.yang_mills_cyclic_generation`` invocation
from ``data_gen/`` and direct execution of this file are also supported.
Default CSV filenames contain ``cyclic`` and are distinct from the original
generator's defaults. A JSON ``.report.json`` file is placed beside the raw CSV.
``--report-out`` sets its path; the old ``--log-out`` spelling is an alias.
``--batch-size`` aliases ``--generator-batch-size``, and ``--mass`` aliases
``--energy-scale``. Cleaning and validation cannot be disabled.

Add ``--test-size 100`` to hold out 100 rows whose tokenised simple targets and
scrambled inputs each appear only once in the pool. Repeated targets, including
different scrambled variants, stay together in training. Every source row is
preserved, and both train/test target and input intersections are checked to be
empty. ``--samples`` is the total source pool size, including held-out rows.
Split files and a verification manifest are
written to ``cyclic_train_test/`` beside the raw pool, or to
``--split-output-dir``. Selection uses ``--split-seed`` (default: 20260401),
and existing split outputs are protected unless ``--split-overwrite`` is set.
Splitting requires N >= 4, tokenisation, enough eligible singleton targets, and
at least one training row; without ``--test-size``, only the clean pool is published.
The safeguard compares token sequences, not general algebraic equivalence.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence


# The shared generator imports Tokenizer as a top-level module. Keep that
# historical import available under both package layouts and in spawn workers.
_DATA_GEN_DIR = Path(__file__).resolve().parent.parent
if str(_DATA_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_GEN_DIR))

if __package__:
    from . import generate as _generate
    from . import generate_clean_4pt as _clean
    from . import split_cyclic_train_test as _split
else:  # python data_gen/data_gen_ym/yang_mills_cyclic_generation.py
    from data_gen_ym import generate as _generate
    from data_gen_ym import generate_clean_4pt as _clean
    from data_gen_ym import split_cyclic_train_test as _split


DEFAULT_RAW_OUT_TEMPLATE = "gi_cyclic_{N}pt_{NSAMPS}k.csv"
DEFAULT_TOK_OUT_TEMPLATE = "gi_cyclic_{N}pt_tok_{NSAMPS}k.csv"
DEFAULT_LOG_OUT_TEMPLATE = "gen_data_cyclic_{N}pt_{NSAMPS}k.log"


def build_dataset(N: int, num_samples: int, **kwargs) -> list[tuple[str, str]]:
    """Build fixed-order pairs using all options of ``generate.build_dataset``.

    The ``cyclic_order`` policy is always enabled by this entry point.
    """
    return _generate.build_dataset(N, num_samples, cyclic_order=True, **kwargs)


def build_dataset_batched(N: int, num_samples: int, **kwargs) -> list[tuple[str, str]]:
    """Build fixed-order batches using the shared builder's batching options."""
    return _generate.build_dataset_batched(N, num_samples, cyclic_order=True, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = _clean.build_parser()
    parser.description = (
        "Generate exactly the requested number of unique, canonical, numerically "
        "nonzero Yang--Mills pairs with forward cyclic F-block ordering."
    )
    parser.epilog = (
        "For N >= 6 the pole model still omits genuine multiparticle channels. "
        "The nonzero and equivalence gates are numerical checks, not symbolic proofs."
    )
    parser.add_argument("N", nargs="?", type=int, default=4)
    parser.add_argument(
        "--batch-size", dest="generator_batch_size", type=int,
        default=argparse.SUPPRESS, help="Alias for --generator-batch-size.",
    )
    parser.add_argument(
        "--mass", dest="energy_scale", type=float,
        default=argparse.SUPPRESS, help="Alias for --energy-scale.",
    )
    parser.add_argument(
        "--log-out", dest="report_out", default=argparse.SUPPRESS,
        help="Compatibility alias for --report-out; writes a JSON generation report.",
    )
    parser.add_argument(
        "--spurious-repeat-probability", dest="denom_repeat_probability", type=float,
        default=argparse.SUPPRESS, help="Alias for --denom-repeat-probability.",
    )
    parser.add_argument(
        "--dataset-kind", choices=("oneshot", "step"), default="oneshot",
        help="Compatibility selector; both modes use the same pair builder.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Unsupported: canonical nonzero generation always validates pairs.",
    )
    parser.add_argument(
        "--test-size", type=int, default=0,
        help="Hold out this many singleton targets with unique scrambled inputs.",
    )
    parser.add_argument("--split-seed", type=int, default=_split.DEFAULT_SPLIT_SEED)
    parser.add_argument("--split-output-dir", type=Path)
    parser.add_argument("--split-overwrite", action="store_true")
    return parser


def _report_path(raw_path: str) -> str:
    path = Path(raw_path)
    name = path.name
    for suffix in (".csv.gz", ".csv"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return str(path.with_name(name + ".report.json"))


def _validate_split_destinations(args: argparse.Namespace) -> None:
    """Catch output conflicts before a potentially long generation run."""
    destinations = _split.output_paths(
        args.split_output_dir, n_particles=args.N, source_rows=args.samples,
        test_size=args.test_size, seed=args.split_seed,
    )
    pool_paths = {Path(path).expanduser().resolve() for path in (
        args.raw_out, args.tok_out, args.report_out,
    )}
    if any(path.resolve() in pool_paths for path in destinations.all()):
        raise ValueError("split outputs must be distinct from pool and report outputs")
    if not args.split_overwrite:
        existing = [str(path) for path in destinations.all() if _clean._lexists(path)]
        if existing:
            raise FileExistsError(
                "split output already exists (use --split-overwrite to replace): "
                + ", ".join(existing)
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.N < 4:
        parser.error("canonical cyclic generation requires N >= 4")
    if args.no_validate:
        parser.error("--no-validate is incompatible with canonical nonzero generation")
    _clean._validate_args(args, parser, n_particles=args.N)
    if args.test_size < 0:
        parser.error("--test-size must be non-negative")
    if args.test_size >= args.samples:
        parser.error("--test-size must be smaller than --samples")
    if args.test_size and args.no_tokenise:
        parser.error("--test-size requires tokenisation; remove --no-tokenise")
    if args.no_progress:
        args.progress_every = 0

    names = {"N": args.N, "NSAMPS": args.samples // 1000}
    args.raw_out = args.raw_out or DEFAULT_RAW_OUT_TEMPLATE.format(**names)
    args.tok_out = args.tok_out or DEFAULT_TOK_OUT_TEMPLATE.format(**names)
    args.report_out = args.report_out or _report_path(args.raw_out)
    args.split_output_dir = (
        args.split_output_dir or Path(args.raw_out).parent / "cyclic_train_test"
    ).expanduser().resolve()
    if args.test_size:
        try:
            _validate_split_destinations(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))

    try:
        stats, report = _clean.generate_to_files(
            args, n_particles=args.N, cyclic_order=True,
            generator_name=f"clean_cyclic_{args.N}pt_yang_mills",
        )
        print(
            f"accepted {stats.accepted:,} canonical nonzero cyclic pairs from "
            f"{stats.candidates_generated:,} generated candidates"
        )
        print(f"  raw: {report['raw_output']}")
        if report["token_output"]:
            print(f"  tokenized: {report['token_output']}")
        print(f"  generation report: {Path(args.report_out).expanduser().resolve()}")
        print(f"  cyclic order: {','.join(map(str, _split.default_cyclic_order(args.N)))}")
        if args.test_size:
            split_report = _split.split_cyclic_dataset(
                Path(args.raw_out), Path(args.tok_out), n_particles=args.N,
                test_size=args.test_size, seed=args.split_seed,
                output_dir=args.split_output_dir,
                tokenizer_max_particles=args.tokenizer_max_particles,
                overwrite=args.split_overwrite,
            )
            for label, output in split_report["outputs"].items():
                print(f"  {label}: {output['rows']} rows -> {output['path']}")
            for side in ("target", "input"):
                print(
                    f"  train/test {side} overlap: "
                    f"{split_report['verification'][f'train_test_{side}_overlap']}"
                )
            print(f"  split manifest: {split_report['manifest_path']}")
    except (ArithmeticError, csv.Error, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
