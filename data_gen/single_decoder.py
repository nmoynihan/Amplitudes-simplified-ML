#!/usr/bin/env python3
"""
Decode one tokenized amplitude expression.

The input should be a single JSON/Python-style list of integer token IDs, e.g.

    [4, 21, 25, 26]

Usage:
    python single_decoder.py "[4, 21, 25, 26]"
    python single_decoder.py tokens.txt
    echo "[4, 21, 25, 26]" | python single_decoder.py -
"""
import argparse
import ast
import pathlib
import sys

from Tokenizer import ScatteringAmplitudeTokenizer


def _read_token_text(source: str) -> str:
    if source == "-":
        return sys.stdin.read().strip()

    source = source.strip()
    if source.startswith("["):
        return source

    path = pathlib.Path(source)
    try:
        path_exists = path.exists()
    except OSError:
        path_exists = False
    if path_exists:
        return path.read_text(encoding="utf-8").strip()

    return source


def _parse_token_list(text: str) -> list[int]:
    try:
        tokens = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Input must be a Python/JSON-style list of token IDs") from exc

    if not isinstance(tokens, list) or not all(isinstance(tok, int) for tok in tokens):
        raise ValueError("Input must be a list containing only integers")

    return tokens


def decode_single(
    token_source: str,
    *,
    max_particles: int = 8,
    decode_format: str = "infix",
) -> str:
    token_text = _read_token_text(token_source)
    tokens = _parse_token_list(token_text)
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=max_particles)

    if decode_format == "infix":
        return tokenizer.decode_infix(tokens)
    if decode_format == "prefix":
        return tokenizer.decode_prefix(tokens)

    raise ValueError(f"Unsupported decode format: {decode_format}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode one tokenized amplitude expression.")
    parser.add_argument(
        "tokens",
        help="Token list literal, '-' for stdin, or path to a text file containing one token list.",
    )
    parser.add_argument("--max-particles", type=int, default=8)
    parser.add_argument(
        "--decode-format",
        choices=["infix", "prefix"],
        default="infix",
        help="Output format for the decoded expression.",
    )
    args = parser.parse_args()

    print(
        decode_single(
            args.tokens,
            max_particles=args.max_particles,
            decode_format=args.decode_format,
        )
    )


if __name__ == "__main__":
    main()
