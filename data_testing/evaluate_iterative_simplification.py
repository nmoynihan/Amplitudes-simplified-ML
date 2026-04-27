#!/usr/bin/env python3
"""
Evaluate a step-model by repeatedly applying verified simplification moves.

The model is trained on one-step pairs where:
    input  = current expression
    target = one shorter equivalent expression

At inference time this script loops:
    current -> model candidates -> verifier -> best shorter candidate -> current

It writes per-row trajectories plus a compact summary under data_testing/outputs.
"""
from __future__ import annotations

import csv
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_TESTING_DIR = ROOT / "data_testing"
OUTPUT_SUBDIR = "outputs"

MODEL_PATH = ROOT / "models" / "step_model" / "best_model.pt"
DEVICE = "auto"  # "auto", "cpu", "cuda"

N_PARTICLES = 4
NUM_SAMPLES = 16
GENERATION_SEED = 123
TOKENIZER_MAX_PARTICLES = 8
MAX_TOKENS = 2048

DATA_FILENAME_STEM = "iterative_eval_4pt"
RAW_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}.csv"
TOK_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_tok.csv"
GEN_LOG_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}.log"
DETAIL_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_results.csv"
SUMMARY_CSV_PATH = DATA_TESTING_DIR / OUTPUT_SUBDIR / f"{DATA_FILENAME_STEM}_summary.csv"

GEN_MIN_TERMS = 1
GEN_MAX_TERMS = 2
GEN_MIN_SCRAMBLES = 2
GEN_MAX_SCRAMBLES = 5
GEN_VALIDATE = True
GEN_MASS = 2.0
GEN_SCRAMBLES = None

MAX_SIMPLIFICATION_STEPS = 8
PRINT_EXAMPLES = 3

NUMERIC_EQUIV_SAMPLES = 3
NUMERIC_EQUIV_SEED = 42
NUMERIC_EQUIV_MASS = 2.0
NUMERIC_EQUIV_POL_MODES = ("coulomb", "covariant")
NUMERIC_TOL_ABS = 1e-12
NUMERIC_TOL_REL = 1e-10

SPECIAL_TOKENS = {"pad": 0, "bos": 2, "eos": 3}


@dataclass(frozen=True)
class IterativeDecodeConfig:
    name: str = "beam"
    decoding_method: str = "beam"  # "greedy", "beam", or "nucleus"
    beam_size: int = 8
    max_length: int | None = None
    max_beams_to_check: int | None = None
    nucleus_attempts: int = 8
    p_nucleus: float = 0.99
    temp_min: float = 0.7
    temp_max: float = 1.8


DECODE_CONFIG = IterativeDecodeConfig()


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _refresh_output_paths() -> None:
    global RAW_CSV_PATH, TOK_CSV_PATH, GEN_LOG_PATH, DETAIL_CSV_PATH, SUMMARY_CSV_PATH
    base = DATA_TESTING_DIR / OUTPUT_SUBDIR
    RAW_CSV_PATH = base / f"{DATA_FILENAME_STEM}.csv"
    TOK_CSV_PATH = base / f"{DATA_FILENAME_STEM}_tok.csv"
    GEN_LOG_PATH = base / f"{DATA_FILENAME_STEM}.log"
    DETAIL_CSV_PATH = base / f"{DATA_FILENAME_STEM}_results.csv"
    SUMMARY_CSV_PATH = base / f"{DATA_FILENAME_STEM}_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a one-step simplification model on fresh generated data."
    )
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=GENERATION_SEED)
    parser.add_argument("--tokenizer-max-particles", type=int, default=TOKENIZER_MAX_PARTICLES)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--output-subdir", type=str, default=OUTPUT_SUBDIR)
    parser.add_argument("--output-stem", type=str, default=None)
    parser.add_argument("--min-terms", type=int, default=GEN_MIN_TERMS)
    parser.add_argument("--max-terms", type=int, default=GEN_MAX_TERMS)
    parser.add_argument("--min-scr", type=int, default=GEN_MIN_SCRAMBLES)
    parser.add_argument("--max-scr", type=int, default=GEN_MAX_SCRAMBLES)
    parser.add_argument(
        "--scrambles",
        nargs="*",
        default=GEN_SCRAMBLES,
        choices=["all", "none", *gd.DEFAULT_SCRAMBLES],
        help="Scrambles used to generate fresh evaluation data. Omit for all.",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_SIMPLIFICATION_STEPS)
    parser.add_argument("--print-examples", type=int, default=PRINT_EXAMPLES)
    parser.add_argument("--decoding-method", choices=["greedy", "beam", "nucleus"], default=DECODE_CONFIG.decoding_method)
    parser.add_argument("--beam-size", type=int, default=DECODE_CONFIG.beam_size)
    parser.add_argument("--max-beams-to-check", type=int, default=DECODE_CONFIG.max_beams_to_check)
    parser.add_argument("--nucleus-attempts", type=int, default=DECODE_CONFIG.nucleus_attempts)
    parser.add_argument("--p-nucleus", type=float, default=DECODE_CONFIG.p_nucleus)
    parser.add_argument("--temp-min", type=float, default=DECODE_CONFIG.temp_min)
    parser.add_argument("--temp-max", type=float, default=DECODE_CONFIG.temp_max)
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global MODEL_PATH, DEVICE, N_PARTICLES, NUM_SAMPLES, GENERATION_SEED
    global TOKENIZER_MAX_PARTICLES, MAX_TOKENS, OUTPUT_SUBDIR, DATA_FILENAME_STEM
    global GEN_MIN_TERMS, GEN_MAX_TERMS, GEN_MIN_SCRAMBLES, GEN_MAX_SCRAMBLES, GEN_SCRAMBLES
    global MAX_SIMPLIFICATION_STEPS, PRINT_EXAMPLES, DECODE_CONFIG

    MODEL_PATH = _path_arg(args.model_path)
    DEVICE = args.device
    N_PARTICLES = args.n_particles
    NUM_SAMPLES = args.num_samples
    GENERATION_SEED = args.seed
    TOKENIZER_MAX_PARTICLES = args.tokenizer_max_particles
    MAX_TOKENS = args.max_tokens
    OUTPUT_SUBDIR = args.output_subdir
    DATA_FILENAME_STEM = args.output_stem or f"iterative_eval_{N_PARTICLES}pt"
    GEN_MIN_TERMS = args.min_terms
    GEN_MAX_TERMS = args.max_terms
    GEN_MIN_SCRAMBLES = args.min_scr
    GEN_MAX_SCRAMBLES = args.max_scr
    GEN_SCRAMBLES = args.scrambles
    MAX_SIMPLIFICATION_STEPS = args.max_steps
    PRINT_EXAMPLES = args.print_examples
    DECODE_CONFIG = IterativeDecodeConfig(
        name=args.decoding_method,
        decoding_method=args.decoding_method,
        beam_size=args.beam_size,
        max_beams_to_check=args.max_beams_to_check,
        nucleus_attempts=args.nucleus_attempts,
        p_nucleus=args.p_nucleus,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
    )
    _refresh_output_paths()


# ============================================================================
# Imports from repo modules
# ============================================================================

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


# ============================================================================
# Helpers
# ============================================================================


def resolve_device() -> str:
    if DEVICE == "cpu":
        return "cpu"
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE='cuda' but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_output_dirs() -> None:
    (DATA_TESTING_DIR / OUTPUT_SUBDIR).mkdir(parents=True, exist_ok=True)


def strip_special_tokens(seq: list[int], *, keep_eos: bool = False) -> list[int]:
    cleaned = clean_seq(
        seq,
        pad_token=SPECIAL_TOKENS["pad"],
        eos_token=SPECIAL_TOKENS["eos"],
    )
    if cleaned and cleaned[0] == SPECIAL_TOKENS["bos"]:
        cleaned = cleaned[1:]
    if not keep_eos and cleaned and cleaned[-1] == SPECIAL_TOKENS["eos"]:
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


def safe_decode_infix(tokenizer: ScatteringAmplitudeTokenizer, seq: list[int]) -> tuple[bool, str, str]:
    if not seq:
        return False, "", "empty prediction"
    try:
        return True, tokenizer.decode_infix(seq), ""
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def encode_expr(tokenizer: ScatteringAmplitudeTokenizer, expr: str) -> list[int]:
    return tokenizer.encode_infix(expr)


def source_tensor(tokens: list[int]) -> torch.Tensor:
    seq = [SPECIAL_TOKENS["bos"], *tokens, SPECIAL_TOKENS["eos"]]
    return torch.tensor([seq], dtype=torch.long)


def token_len(tokenizer: ScatteringAmplitudeTokenizer, expr: str) -> int:
    return len(encode_expr(tokenizer, expr))


def temp_schedule(cfg: IterativeDecodeConfig) -> list[float]:
    if cfg.decoding_method != "nucleus":
        return [cfg.temp_min]
    n = max(1, cfg.nucleus_attempts)
    if n == 1:
        return [cfg.temp_min]
    step = (cfg.temp_max - cfg.temp_min) / (n - 1)
    return [cfg.temp_min + i * step for i in range(n)]


def load_raw_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_token_rows(path: Path) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "simple": json.loads(row["simple"]),
                    "scrambled": json.loads(row["scrambled"]),
                }
            )
    return rows


def precompute_kinematics() -> list[tuple[Any, Any]]:
    return [
        generate_kinematics(
            N_PARTICLES,
            M=NUMERIC_EQUIV_MASS,
            pol_mode=pol_mode,
            seed=NUMERIC_EQUIV_SEED + mode_idx * NUMERIC_EQUIV_SAMPLES + i,
        )
        for mode_idx, pol_mode in enumerate(NUMERIC_EQUIV_POL_MODES)
        for i in range(NUMERIC_EQUIV_SAMPLES)
    ]


def numerically_equivalent_exprs(
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
            if not (diff <= NUMERIC_TOL_ABS or diff / scale <= NUMERIC_TOL_REL):
                return False
        return True
    except Exception:
        return False


def generate_test_data() -> None:
    print(f"Generating {NUM_SAMPLES} direct {N_PARTICLES}-point evaluation samples...")
    pairs = gd.build_dataset(
        N_PARTICLES,
        NUM_SAMPLES,
        max_scr=GEN_MAX_SCRAMBLES,
        min_scr=GEN_MIN_SCRAMBLES,
        seed=GENERATION_SEED,
        use_denominators=True,
        validate=GEN_VALIDATE,
        M=GEN_MASS,
        min_terms=GEN_MIN_TERMS,
        max_terms=GEN_MAX_TERMS,
        log_path=str(GEN_LOG_PATH),
        max_tokens=MAX_TOKENS,
        tokenizer_max_particles=TOKENIZER_MAX_PARTICLES,
        scrambles=GEN_SCRAMBLES,
    )
    pairs, removed = gd.dedupe_pairs(pairs)
    pairs = pairs[:NUM_SAMPLES]
    gd.write_csv(pairs, str(RAW_CSV_PATH))
    gd.tokenise_csv(
        str(RAW_CSV_PATH),
        str(TOK_CSV_PATH),
        max_particles=TOKENIZER_MAX_PARTICLES,
        max_sequence_length=MAX_TOKENS,
    )
    print(f"Wrote raw data to {RAW_CSV_PATH}")
    print(f"Wrote tokenised data to {TOK_CSV_PATH}")
    print(f"Dedupe removed {removed} duplicate pairs")


def load_model(device: str):
    print(f"Loading step model from {MODEL_PATH}")
    loaded = load_transformer_model(
        TransformerRegressor,
        str(MODEL_PATH),
        device=device,
    )
    model = loaded["model"]
    model.eval()
    return model


def decode_candidate_sequences(
    model,
    tokenizer: ScatteringAmplitudeTokenizer,
    current_expr: str,
    cfg: IterativeDecodeConfig,
) -> list[list[int]]:
    current_tokens = encode_expr(tokenizer, current_expr)
    src = source_tensor(current_tokens)
    max_length = cfg.max_length or (len(current_tokens) + 2)

    candidates: list[list[int]] = []
    for temperature in temp_schedule(cfg):
        decoded, all_beams = decode_with_model(
            model,
            src,
            max_length=max_length,
            decoding_method=cfg.decoding_method,
            beam_size=cfg.beam_size,
            p_nucleus=cfg.p_nucleus,
            temperature_nucleus=temperature,
            bos_token=SPECIAL_TOKENS["bos"],
            eos_token=SPECIAL_TOKENS["eos"],
            pad_token=SPECIAL_TOKENS["pad"],
        )
        top1_full = clean_seq(
            decoded[0].tolist(),
            pad_token=SPECIAL_TOKENS["pad"],
            eos_token=SPECIAL_TOKENS["eos"],
        )
        candidates.append(strip_special_tokens(top1_full))

        if all_beams is not None and all_beams:
            beam_candidates = [strip_special_tokens(seq) for seq in all_beams[0]]
            if cfg.max_beams_to_check is not None:
                beam_candidates = beam_candidates[: cfg.max_beams_to_check]
            candidates.extend(beam_candidates)

    return unique_sequences(candidates)


def pick_best_step(
    tokenizer: ScatteringAmplitudeTokenizer,
    current_expr: str,
    candidate_sequences: list[list[int]],
    cached_kinematics: list[tuple[Any, Any]],
) -> tuple[str | None, dict[str, Any]]:
    current_len = token_len(tokenizer, current_expr)
    best_expr: str | None = None
    best_len: int | None = None
    counts = {
        "candidate_sequences_checked": len(candidate_sequences),
        "candidate_valid_decode_count": 0,
        "candidate_malformed_count": 0,
        "candidate_not_equivalent_count": 0,
        "candidate_not_shorter_count": 0,
        "candidate_accepted_count": 0,
    }

    for seq in candidate_sequences:
        decode_ok, cand_expr, _ = safe_decode_infix(tokenizer, seq)
        if not decode_ok:
            counts["candidate_malformed_count"] += 1
            continue
        try:
            cand_len = token_len(tokenizer, cand_expr)
        except Exception:
            counts["candidate_malformed_count"] += 1
            continue

        counts["candidate_valid_decode_count"] += 1
        if not numerically_equivalent_exprs(cand_expr, current_expr, cached_kinematics):
            counts["candidate_not_equivalent_count"] += 1
            continue
        if cand_len >= current_len:
            counts["candidate_not_shorter_count"] += 1
            continue

        counts["candidate_accepted_count"] += 1
        if best_len is None or cand_len < best_len:
            best_expr = cand_expr
            best_len = cand_len

    return best_expr, counts


def stop_reason_from_counts(counts: dict[str, Any]) -> str:
    if counts["candidate_sequences_checked"] == 0:
        return "no_candidate_found"
    if counts["candidate_valid_decode_count"] == 0:
        return "malformed_output"
    if counts["candidate_not_equivalent_count"] > 0:
        return "not_numerically_equivalent"
    if counts["candidate_not_shorter_count"] > 0:
        return "equivalent_but_not_shorter"
    return "no_candidate_found"


def simplify_iteratively(
    model,
    tokenizer: ScatteringAmplitudeTokenizer,
    start_expr: str,
    target_simple_expr: str,
    cached_kinematics: list[tuple[Any, Any]],
    cfg: IterativeDecodeConfig,
) -> dict[str, Any]:
    current_expr = start_expr
    initial_len = token_len(tokenizer, start_expr)
    trajectory = [start_expr]
    aggregate_counts = {
        "candidate_sequences_checked": 0,
        "candidate_valid_decode_count": 0,
        "candidate_malformed_count": 0,
        "candidate_not_equivalent_count": 0,
        "candidate_not_shorter_count": 0,
        "candidate_accepted_count": 0,
    }
    stop_reason = "max_steps"

    for _step in range(MAX_SIMPLIFICATION_STEPS):
        candidate_sequences = decode_candidate_sequences(model, tokenizer, current_expr, cfg)
        best_expr, counts = pick_best_step(
            tokenizer,
            current_expr,
            candidate_sequences,
            cached_kinematics,
        )
        for key in aggregate_counts:
            aggregate_counts[key] += int(counts[key])

        if best_expr is None:
            stop_reason = stop_reason_from_counts(counts)
            break

        current_expr = best_expr
        trajectory.append(current_expr)

    final_len = token_len(tokenizer, current_expr)
    target_tokens = encode_expr(tokenizer, target_simple_expr)
    final_tokens = encode_expr(tokenizer, current_expr)
    target_norm = tokenizer.decode_infix(target_tokens)
    final_norm = tokenizer.decode_infix(final_tokens)

    return {
        "final_expr": current_expr,
        "initial_token_count": initial_len,
        "final_token_count": final_len,
        "target_simple_token_count": len(target_tokens),
        "token_reduction": initial_len - final_len,
        "accepted_steps": len(trajectory) - 1,
        "stop_reason": stop_reason,
        "exact_token_match": int(final_tokens == target_tokens),
        "exact_string_match": int(final_norm == target_norm),
        "num_eq_target_simple": int(
            numerically_equivalent_exprs(current_expr, target_simple_expr, cached_kinematics)
        ),
        "num_eq_input_scrambled": int(
            numerically_equivalent_exprs(current_expr, start_expr, cached_kinematics)
        ),
        "improved": int(final_len < initial_len),
        "trajectory": trajectory,
        **aggregate_counts,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"total_examples": 0}

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in rows) / total

    return {
        "total_examples": total,
        "exact_token_matches": sum(int(r["exact_token_match"]) for r in rows),
        "exact_string_matches": sum(int(r["exact_string_match"]) for r in rows),
        "num_eq_target_simple": sum(int(r["num_eq_target_simple"]) for r in rows),
        "num_eq_input_scrambled": sum(int(r["num_eq_input_scrambled"]) for r in rows),
        "improved_examples": sum(int(r["improved"]) for r in rows),
        "improved_not_exact": sum(
            int(r["improved"]) and not int(r["exact_token_match"]) for r in rows
        ),
        "avg_token_reduction": avg("token_reduction"),
        "avg_accepted_steps": avg("accepted_steps"),
        "avg_candidates_checked": avg("candidate_sequences_checked"),
    }


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [k for k in rows[0].keys() if k != "trajectory"]
        fieldnames.append("trajectory_json")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: v for k, v in row.items() if k != "trajectory"}
            out["trajectory_json"] = json.dumps(row["trajectory"])
            writer.writerow(out)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def print_summary(summary: dict[str, Any]) -> None:
    total = int(summary.get("total_examples", 0))
    print("\nIterative simplification summary")
    print(f"  total examples          : {total}")
    if total == 0:
        return
    print(f"  exact token matches     : {summary['exact_token_matches']} / {total}")
    print(f"  exact string matches    : {summary['exact_string_matches']} / {total}")
    print(f"  num-eq target simple    : {summary['num_eq_target_simple']} / {total}")
    print(f"  num-eq input scrambled  : {summary['num_eq_input_scrambled']} / {total}")
    print(f"  improved examples       : {summary['improved_examples']} / {total}")
    print(f"  improved but not exact  : {summary['improved_not_exact']} / {total}")
    print(f"  avg token reduction     : {summary['avg_token_reduction']:.2f}")
    print(f"  avg accepted steps      : {summary['avg_accepted_steps']:.2f}")
    print(f"  avg candidates checked  : {summary['avg_candidates_checked']:.2f}")


def print_examples(rows: list[dict[str, Any]], count: int) -> None:
    if not rows:
        return
    print(f"\nExample iterative results ({min(count, len(rows))} rows):")
    for row in rows[:count]:
        print(f"  row {row['row_id']}")
        print(f"    target simple : {row['target_simple'][:180]}")
        print(f"    initial       : {row['input_scrambled'][:180]}")
        print(f"    final         : {row['final_expr'][:180]}")
        print(
            f"    steps={row['accepted_steps']} reduction={row['token_reduction']} "
            f"stop={row['stop_reason']} exact={row['exact_token_match']}"
        )


def main() -> None:
    configure_from_args(parse_args())

    t0 = time.perf_counter()
    ensure_output_dirs()

    device = resolve_device()
    print(f"Using device: {device}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Evaluation samples: {NUM_SAMPLES} fresh {N_PARTICLES}-point examples")
    print(f"Decoding: {DECODE_CONFIG.decoding_method}, beam_size={DECODE_CONFIG.beam_size}")

    generate_test_data()
    raw_rows = load_raw_rows(RAW_CSV_PATH)
    token_rows = load_token_rows(TOK_CSV_PATH)
    if len(raw_rows) != len(token_rows):
        raise ValueError("Raw and tokenised row counts do not match")

    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=TOKENIZER_MAX_PARTICLES,
        max_sequence_length=MAX_TOKENS,
    )
    cached_kinematics = precompute_kinematics()
    model = load_model(device)

    detail_rows: list[dict[str, Any]] = []
    for row_id, raw_row in enumerate(raw_rows):
        result = simplify_iteratively(
            model,
            tokenizer,
            raw_row["scrambled"],
            raw_row["simple"],
            cached_kinematics,
            DECODE_CONFIG,
        )
        detail_rows.append(
            {
                "row_id": row_id,
                "input_scrambled": raw_row["scrambled"],
                "target_simple": raw_row["simple"],
                **result,
            }
        )

    summary = summarize(detail_rows)
    write_detail_csv(DETAIL_CSV_PATH, detail_rows)
    write_summary_csv(SUMMARY_CSV_PATH, summary)
    print_summary(summary)
    print_examples(detail_rows, PRINT_EXAMPLES)
    print(f"\nWrote detail results to {DETAIL_CSV_PATH}")
    print(f"Wrote summary to {SUMMARY_CSV_PATH}")
    print(f"Total wall time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
