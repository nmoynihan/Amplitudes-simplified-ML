"""Command-line entry point for the reproducible four-point workflow."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for mode, help_text in (
        ("run", "Prepare, independently audit, run the checkpoint and verify its prediction"),
        ("verify", "Reconstruct and independently audit the input, without a model"),
        ("prepare", "Reconstruct the completed amplitude and model-ready scalar input"),
        ("derive", "Repeat the sparse symbolic basis search"),
        ("audit", "Independently audit an existing scalar CSV"),
    ):
        sub = commands.add_parser(mode, help=help_text)
        sub.add_argument("--output-dir", type=Path, required=True,
                         help="New or empty directory for this run")
        if mode != "audit":
            sub.add_argument("--seed", type=Path, help="Override the repository cyclic seed")
            sub.add_argument("--reference", type=Path, help="Override the repository full amplitude")
        if mode in {"run", "verify", "prepare"}:
            selection = sub.add_mutually_exclusive_group()
            selection.add_argument("--derive", action="store_true",
                                   help="Repeat the basis search before preparation")
            selection.add_argument("--derivation", type=Path,
                                   help="Use a sparse_extended_result.json from the derive command")
        if mode == "run":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--threads", type=int, default=4)
            sub.add_argument("--max-length", type=int, default=160,
                             help="Greedy sequence limit including BOS (default: 160)")
        if mode == "audit":
            sub.add_argument("--input", type=Path, required=True,
                             help="Headerless id,expression scalar-contraction CSV")
    args = parser.parse_args()
    from .workflow import execute
    try:
        result = execute(
            args.command, args.output_dir,
            checkpoint=getattr(args, "checkpoint", None),
            source=getattr(args, "input", None),
            seed=getattr(args, "seed", None), reference=getattr(args, "reference", None),
            derive_basis=getattr(args, "derive", False),
            derivation=getattr(args, "derivation", None),
            threads=getattr(args, "threads", 4), max_length=getattr(args, "max_length", 160),
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Workflow failed: {error}\n")
    if args.command == "run":
        print(f"Verified greedy prediction: {result['input_tokens']} -> {result['output_tokens']} content tokens")
        print(f"Exact identity and {result['independent_numeric_points']} numerical checks passed")
    print(f"{args.command}: passed; {result['model_calls']} model calls")
    print(args.output_dir.resolve() / "results.json")


if __name__ == "__main__":
    main()
