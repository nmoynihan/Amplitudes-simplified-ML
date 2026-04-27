#!/usr/bin/env python3
"""
Evaluate a step model directly on fresh held-out one-step pairs.

This is different from iterative simplification. Each generated row is already
one training-policy example:

    input  = scrambled current expression
    target = one verified shorter expression

The evaluator asks the model for one batch of candidates per row and reports
whether the output is syntactically valid, exactly matches the held-out target,
or is at least numerically equivalent to the input and shorter.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
DATA_TESTING_DIR = ROOT / "data_testing"

sys.path.insert(0, str(ROOT / "data_gen"))
sys.path.insert(0, str(ROOT / "transformer"))

import gen_data as gd
from Tokenizer import ScatteringAmplitudeTokenizer
from kinematics import generate_kinematics
from transformer_functions import (
    TransformerRegressor,
    clean_seq,
    decode_with_model,
    load_transformer_model,
)


SPECIAL_TOKENS = {"pad": 0, "bos": 2, "eos": 3}


@dataclass(frozen=True)
class DecodeConfig:
    decoding_method: str
    beam_size: int
    max_beams_to_check: int | None
    p_nucleus: float
    temperature: float


def path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a step model on fresh held-out one-step pairs."
    )
    parser.add_argument("--model-path", type=str, default="models/step_model/best_model.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--tokenizer-max-particles", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-subdir", type=str, default="outputs")
    parser.add_argument("--output-stem", type=str, default=None)
    parser.add_argument("--min-terms", type=int, default=1)
    parser.add_argument("--max-terms", type=int, default=2)
    parser.add_argument("--min-scr", type=int, default=1)
    parser.add_argument("--max-scr", type=int, default=5)
    parser.add_argument(
        "--scrambles",
        nargs="*",
        default=None,
        choices=["all", "none", *gd.DEFAULT_SCRAMBLES],
        help="Scrambles used to generate fresh held-out step data. Omit for all.",
    )
    parser.add_argument("--decoding-method", choices=["greedy", "beam", "nucleus"], default="beam")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--max-beams-to-check", type=int, default=None)
    parser.add_argument("--p-nucleus", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--numeric-samples", type=int, default=3)
    parser.add_argument("--numeric-seed", type=int, default=42)
    parser.add_argument("--mass", type=float, default=2.0)
    parser.add_argument("--print-examples", type=int, default=3)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE='cuda' but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def strip_special_tokens(seq: list[int]) -> list[int]:
    cleaned = clean_seq(
        seq,
        pad_token=SPECIAL_TOKENS["pad"],
        eos_token=SPECIAL_TOKENS["eos"],
    )
    if cleaned and cleaned[0] == SPECIAL_TOKENS["bos"]:
        cleaned = cleaned[1:]
    if cleaned and cleaned[-1] == SPECIAL_TOKENS["eos"]:
        cleaned = cleaned[:-1]
    return cleaned


def unique_sequences(seqs: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for seq in seqs:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            out.append(seq)
    return out


def safe_decode(tokenizer: ScatteringAmplitudeTokenizer, seq: list[int]) -> tuple[bool, str]:
    if not seq:
        return False, ""
    try:
        return True, tokenizer.decode_infix(seq)
    except Exception:
        return False, ""


def token_len(tokenizer: ScatteringAmplitudeTokenizer, expr: str) -> int:
    return len(tokenizer.encode_infix(expr))


def source_tensor(tokens: list[int]) -> torch.Tensor:
    seq = [SPECIAL_TOKENS["bos"], *tokens, SPECIAL_TOKENS["eos"]]
    return torch.tensor([seq], dtype=torch.long)


def decode_candidates(
    model,
    tokenizer: ScatteringAmplitudeTokenizer,
    source_expr: str,
    cfg: DecodeConfig,
) -> list[list[int]]:
    source_tokens = tokenizer.encode_infix(source_expr)
    decoded, all_beams = decode_with_model(
        model,
        source_tensor(source_tokens),
        max_length=len(source_tokens) + 2,
        decoding_method=cfg.decoding_method,
        beam_size=cfg.beam_size,
        p_nucleus=cfg.p_nucleus,
        temperature_nucleus=cfg.temperature,
        bos_token=SPECIAL_TOKENS["bos"],
        eos_token=SPECIAL_TOKENS["eos"],
        pad_token=SPECIAL_TOKENS["pad"],
    )

    candidates = [
        strip_special_tokens(
            clean_seq(
                decoded[0].tolist(),
                pad_token=SPECIAL_TOKENS["pad"],
                eos_token=SPECIAL_TOKENS["eos"],
            )
        )
    ]
    if all_beams is not None and all_beams:
        beams = [strip_special_tokens(seq) for seq in all_beams[0]]
        if cfg.max_beams_to_check is not None:
            beams = beams[: cfg.max_beams_to_check]
        candidates.extend(beams)
    return unique_sequences(candidates)


def precompute_kinematics(args: argparse.Namespace) -> list[tuple[Any, Any]]:
    return [
        generate_kinematics(
            args.n_particles,
            M=args.mass,
            pol_mode=pol_mode,
            seed=args.numeric_seed + mode_idx * args.numeric_samples + i,
        )
        for mode_idx, pol_mode in enumerate(("coulomb", "covariant"))
        for i in range(args.numeric_samples)
    ]


def numerically_equivalent(
    expr_a: str,
    expr_b: str,
    cached_kinematics: list[tuple[Any, Any]],
) -> bool:
    try:
        for momenta, pols in cached_kinematics:
            val_a = gd.eval_infix_numeric(expr_a, momenta, pols)
            val_b = gd.eval_infix_numeric(expr_b, momenta, pols)
            if not (math.isfinite(val_a) and math.isfinite(val_b)):
                return False
            diff = abs(val_a - val_b)
            scale = max(abs(val_a), abs(val_b), 1.0)
            if not (diff <= 1e-12 or diff / scale <= 1e-10):
                return False
        return True
    except Exception:
        return False


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    stem = args.output_stem or f"step_pair_eval_{args.n_particles}pt"
    out_dir = DATA_TESTING_DIR / args.output_subdir
    return {
        "dir": out_dir,
        "raw": out_dir / f"{stem}.csv",
        "tok": out_dir / f"{stem}_tok.csv",
        "gen_log": out_dir / f"{stem}.log",
        "results": out_dir / f"{stem}_results.csv",
        "summary": out_dir / f"{stem}_summary.csv",
    }


def generate_step_data(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    print(f"Generating {args.num_samples} held-out one-step {args.n_particles}-point pairs...")
    pairs = gd.build_step_dataset(
        args.n_particles,
        args.num_samples,
        max_scr=args.max_scr,
        min_scr=args.min_scr,
        seed=args.seed,
        use_denominators=True,
        validate=True,
        M=args.mass,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
        log_path=str(paths["gen_log"]),
        max_tokens=args.max_tokens,
        tokenizer_max_particles=args.tokenizer_max_particles,
        scrambles=args.scrambles,
    )
    pairs, removed = gd.dedupe_pairs(pairs)
    pairs = pairs[: args.num_samples]
    gd.write_csv(pairs, str(paths["raw"]))
    gd.tokenise_csv(
        str(paths["raw"]),
        str(paths["tok"]),
        max_particles=args.tokenizer_max_particles,
        max_sequence_length=args.max_tokens,
    )
    print(f"Wrote raw step pairs to {paths['raw']}")
    print(f"Wrote tokenised step pairs to {paths['tok']}")
    print(f"Dedupe removed {removed} duplicate pairs")


def load_raw_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_row(
    row_id: int,
    row: dict[str, str],
    model,
    tokenizer: ScatteringAmplitudeTokenizer,
    cfg: DecodeConfig,
    cached_kinematics: list[tuple[Any, Any]],
) -> dict[str, Any]:
    target_expr = row["simple"]
    source_expr = row["scrambled"]
    target_tokens = tokenizer.encode_infix(target_expr)
    source_len = token_len(tokenizer, source_expr)

    candidates = decode_candidates(model, tokenizer, source_expr, cfg)
    counts = {
        "candidate_sequences_checked": len(candidates),
        "candidate_valid_decode_count": 0,
        "candidate_malformed_count": 0,
        "candidate_exact_token_count": 0,
        "candidate_equivalent_count": 0,
        "candidate_shorter_count": 0,
        "candidate_accepted_count": 0,
    }
    top1_expr = ""
    top1_parse_ok = 0
    top1_exact = 0
    top1_equiv = 0
    top1_shorter = 0
    best_expr = ""
    best_len: int | None = None

    for idx, seq in enumerate(candidates):
        ok, pred_expr = safe_decode(tokenizer, seq)
        if idx == 0:
            top1_expr = pred_expr
            top1_parse_ok = int(ok)
        if not ok:
            counts["candidate_malformed_count"] += 1
            continue

        counts["candidate_valid_decode_count"] += 1
        try:
            pred_tokens = tokenizer.encode_infix(pred_expr)
            pred_len = len(pred_tokens)
        except Exception:
            counts["candidate_malformed_count"] += 1
            continue

        exact = pred_tokens == target_tokens
        equiv = numerically_equivalent(pred_expr, source_expr, cached_kinematics)
        shorter = pred_len < source_len

        counts["candidate_exact_token_count"] += int(exact)
        counts["candidate_equivalent_count"] += int(equiv)
        counts["candidate_shorter_count"] += int(shorter)
        accepted = equiv and shorter
        counts["candidate_accepted_count"] += int(accepted)

        if idx == 0:
            top1_exact = int(exact)
            top1_equiv = int(equiv)
            top1_shorter = int(shorter)

        if accepted and (best_len is None or pred_len < best_len):
            best_expr = pred_expr
            best_len = pred_len

    return {
        "row_id": row_id,
        "source_expr": source_expr,
        "target_expr": target_expr,
        "target_token_count": len(target_tokens),
        "source_token_count": source_len,
        "top1_expr": top1_expr,
        "top1_parse_ok": top1_parse_ok,
        "top1_exact_token_match": top1_exact,
        "top1_num_eq_source": top1_equiv,
        "top1_shorter": top1_shorter,
        "any_exact_token_match": int(counts["candidate_exact_token_count"] > 0),
        "any_valid_equiv_shorter": int(counts["candidate_accepted_count"] > 0),
        "best_accepted_expr": best_expr,
        "best_accepted_token_count": best_len if best_len is not None else "",
        **counts,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"total_examples": 0}

    def sum_int(key: str) -> int:
        return sum(int(row[key]) for row in rows)

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in rows) / total

    return {
        "total_examples": total,
        "top1_parse_ok": sum_int("top1_parse_ok"),
        "top1_exact_token_matches": sum_int("top1_exact_token_match"),
        "top1_num_eq_source": sum_int("top1_num_eq_source"),
        "top1_shorter": sum_int("top1_shorter"),
        "any_exact_token_match": sum_int("any_exact_token_match"),
        "any_valid_equiv_shorter": sum_int("any_valid_equiv_shorter"),
        "avg_candidates_checked": avg("candidate_sequences_checked"),
        "avg_valid_decodes": avg("candidate_valid_decode_count"),
        "avg_malformed": avg("candidate_malformed_count"),
        "total_accepted_candidates": sum_int("candidate_accepted_count"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def print_summary(summary: dict[str, Any]) -> None:
    total = int(summary.get("total_examples", 0))
    print("\nHeld-out step-pair summary")
    print(f"  total examples              : {total}")
    if total == 0:
        return
    print(f"  top1 parse ok               : {summary['top1_parse_ok']} / {total}")
    print(f"  top1 exact token matches    : {summary['top1_exact_token_matches']} / {total}")
    print(f"  top1 num-eq source          : {summary['top1_num_eq_source']} / {total}")
    print(f"  top1 shorter                : {summary['top1_shorter']} / {total}")
    print(f"  any exact in candidates     : {summary['any_exact_token_match']} / {total}")
    print(f"  any valid equiv shorter     : {summary['any_valid_equiv_shorter']} / {total}")
    print(f"  avg candidates checked      : {summary['avg_candidates_checked']:.2f}")
    print(f"  avg valid decodes           : {summary['avg_valid_decodes']:.2f}")
    print(f"  avg malformed candidates    : {summary['avg_malformed']:.2f}")
    print(f"  total accepted candidates   : {summary['total_accepted_candidates']}")


def print_examples(rows: list[dict[str, Any]], count: int) -> None:
    if not rows:
        return
    print(f"\nExample step-pair results ({min(count, len(rows))} rows):")
    for row in rows[:count]:
        print(f"  row {row['row_id']}")
        print(f"    source : {row['source_expr'][:180]}")
        print(f"    target : {row['target_expr'][:180]}")
        print(f"    top1   : {row['top1_expr'][:180]}")
        print(
            f"    parse={row['top1_parse_ok']} exact={row['top1_exact_token_match']} "
            f"equiv={row['top1_num_eq_source']} shorter={row['top1_shorter']} "
            f"any_accept={row['any_valid_equiv_shorter']}"
        )


def main() -> None:
    args = parse_args()
    paths = output_paths(args)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    device = resolve_device(args.device)
    model_path = path_arg(args.model_path)
    print(f"Using device: {device}")
    print(f"Model path: {model_path}")
    print(f"Evaluation samples: {args.num_samples} fresh held-out step pairs")
    print(f"Decoding: {args.decoding_method}, beam_size={args.beam_size}")

    generate_step_data(args, paths)
    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=args.tokenizer_max_particles,
        max_sequence_length=args.max_tokens,
    )
    cached_kinematics = precompute_kinematics(args)

    loaded = load_transformer_model(TransformerRegressor, str(model_path), device=device)
    model = loaded["model"]
    model.eval()

    cfg = DecodeConfig(
        decoding_method=args.decoding_method,
        beam_size=args.beam_size,
        max_beams_to_check=args.max_beams_to_check,
        p_nucleus=args.p_nucleus,
        temperature=args.temperature,
    )
    rows = [
        evaluate_row(row_id, row, model, tokenizer, cfg, cached_kinematics)
        for row_id, row in enumerate(load_raw_rows(paths["raw"]))
    ]

    summary = summarize(rows)
    write_csv(paths["results"], rows)
    write_summary(paths["summary"], summary)
    print_summary(summary)
    print_examples(rows, args.print_examples)
    print(f"\nWrote detail results to {paths['results']}")
    print(f"Wrote summary to {paths['summary']}")
    print(f"Total wall time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
