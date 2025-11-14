#!/usr/bin/env python3
"""
Roundtrip tokenization check
----------------------------

Generate (simple, scrambled) amplitude pairs, tokenize them with
ScatteringAmplitudeTokenizer, detokenize back to infix, and verify that
numeric evaluation matches before vs after for both expressions.

This targets the issue where expressions numerically match pre-tokenization
but not post-detokenization.

Usage (examples)
  python roundtrip_tokenization_check.py --N 5 --count 50 --samples 3 --seed 123

Notes
  - Imports modules directly from the local 'data_generation' folder.
  - Uses gen_data.eval_infix_numeric for numeric evaluation and
    Tokenizer.numerically_equivalent for convenience/comparison.
"""

from __future__ import annotations
import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# Ensure we can import local generation modules without a package install
ROOT = Path(__file__).resolve().parent.parent
DATA_GEN_DIR = ROOT / "data_generation"
if str(DATA_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_DIR))

# Local imports from data_generation dir
from Tokenizer import ScatteringAmplitudeTokenizer, numerically_equivalent  # type: ignore
import gen_data as gd  # type: ignore
import kinematics as km  # type: ignore


@dataclass
class RoundtripFailure:
    pair_index: int
    which: str  # 'simple' or 'scrambled'
    original: str
    decoded: str
    details: dict


def infer_N_from_exprs(*exprs: str, default: int = 5) -> int:
    """Infer minimal N required by tokens in expressions.

    Rules:
      - p_i, M_i require N >= i
      - e_i, F_i require N >= i+1 (since photons are 2..N-1)
    """
    need = default
    for s in exprs:
        for m in re.finditer(r"([pPeEfFM])_(\d+)", s):
            kind = m.group(1)
            i = int(m.group(2))
            if kind in ('p', 'P', 'M'):
                need = max(need, i)
            else:  # e/E/F/f
                need = max(need, i + 1)
    return need


def check_roundtrip(
    pairs: List[Tuple[str, str]],
    *,
    samples: int = 3,
    mass_M: float = 2.0,
    tol_abs: float = 1e-12,
    tol_rel: float = 1e-10,
    seed: int | None = 123,
    max_particles_tokenizer: int = 16,
    max_fail_examples: int = 5,
):
    tok = ScatteringAmplitudeTokenizer(max_particles=max_particles_tokenizer)

    n = len(pairs)
    fail_simple: List[RoundtripFailure] = []
    fail_scrambled: List[RoundtripFailure] = []

    for i, (simple, scrambled) in enumerate(pairs):
        # Infer N from the expressions (robust if denominators/Tr appear)
        N = infer_N_from_exprs(simple, scrambled, default=5)

        # Tokenize both
        try:
            simple_ids = tok.encode_infix(simple)
            scrambled_ids = tok.encode_infix(scrambled)
        except Exception as e:
            # Treat tokenization errors as failures with minimal details
            fail_simple.append(RoundtripFailure(
                pair_index=i, which="simple", original=simple, decoded=f"<encode error: {e}>", details={}
            ))
            fail_scrambled.append(RoundtripFailure(
                pair_index=i, which="scrambled", original=scrambled, decoded=f"<encode error: {e}>", details={}
            ))
            continue

        # Decode back to infix (this is the path used later when training/IO)
        simple_dec = tok.decode_infix(simple_ids)
        scrambled_dec = tok.decode_infix(scrambled_ids)

        # Compare numerically: before vs after, using lists of IDs for detokenized path
        try:
            ok_s, det_s = numerically_equivalent(
                tok, simple, simple_ids, N,
                samples=samples, M=mass_M, tol_abs=tol_abs, tol_rel=tol_rel,
                seed=None if seed is None else seed + i,
                return_details=True,
            )
        except Exception as e:
            ok_s = False
            det_s = {"error": repr(e), "expr_a": simple, "expr_b_decoded": simple_dec, "N": N}
        if not ok_s:
            fail_simple.append(RoundtripFailure(
                pair_index=i, which="simple", original=simple, decoded=simple_dec, details=det_s
            ))

        try:
            ok_t, det_t = numerically_equivalent(
                tok, scrambled, scrambled_ids, N,
                samples=samples, M=mass_M, tol_abs=tol_abs, tol_rel=tol_rel,
                seed=None if seed is None else seed + 10_000 + i,
                return_details=True,
            )
        except Exception as e:
            ok_t = False
            det_t = {"error": repr(e), "expr_a": scrambled, "expr_b_decoded": scrambled_dec, "N": N}
        if not ok_t:
            fail_scrambled.append(RoundtripFailure(
                pair_index=i, which="scrambled", original=scrambled, decoded=scrambled_dec, details=det_t
            ))

    # Summary
    print("\n===== Roundtrip Tokenization Check =====")
    print(f"Total pairs tested: {n}")
    print(f"Simple   roundtrip failures:   {len(fail_simple)}")
    print(f"Scrambled roundtrip failures: {len(fail_scrambled)}")

    def _print_examples(label: str, fails: List[RoundtripFailure]):
        if not fails:
            return
        print(f"\n-- Examples of {label} failures (up to {max_fail_examples}) --")
        for f in fails[:max_fail_examples]:
            print(f"[pair {f.pair_index}] {f.which}:")
            print(f"  original: {f.original}")
            print(f"  decoded : {f.decoded}")
            d = f.details
            if d and d.get("samples"):
                samp = d["samples"][0]
                print(f"  abs_diff={samp['abs_diff']:.3e} rel_diff={samp['rel_diff']:.3e} seed={samp.get('seed')}")
                print(f"  values: a={samp['value_a']:.6g}  b={samp['value_b']:.6g}")
            elif d and d.get("error"):
                print(f"  error: {d['error']}")

        # Print deeper debug for the first failure: expanded numeric strings
        try:
            f0 = fails[0]
            N = infer_N_from_exprs(f0.original, f0.decoded)
            mom, pol = km.generate_kinematics(N, M=mass_M, seed=(seed or 0) + 999)
            num_a = gd.to_numeric_string(f0.original, mom, pol)
            num_b = gd.to_numeric_string(f0.decoded, mom, pol)
            print("\n  -- Numeric-expanded strings for first failure --")
            print("  original-num:", num_a[:300], ("..." if len(num_a) > 300 else ""))
            print("  decoded-num :", num_b[:300], ("..." if len(num_b) > 300 else ""))
            va = gd.eval_infix_numeric(f0.original, mom, pol)
            vb = gd.eval_infix_numeric(f0.decoded, mom, pol)
            diff = abs(va - vb)
            scale = max(abs(va), abs(vb), 1.0)
            rel = diff / scale
            print(f"  direct-eval: a={va:.12g} b={vb:.12g} | abs={diff:.3e} rel={rel:.3e}")
        except Exception:
            pass

    _print_examples("simple", fail_simple)
    _print_examples("scrambled", fail_scrambled)

    return len(fail_simple) == 0 and len(fail_scrambled) == 0


def main():
    ap = argparse.ArgumentParser(description="Check numeric equivalence before vs after tokenization+detokenization.")
    ap.add_argument("--N", type=int, default=5, help="Target number of legs for data generation (upper bound; actual pairs may use fewer).")
    ap.add_argument("--count", type=int, default=25, help="Number of pairs to generate and test.")
    ap.add_argument("--samples", type=int, default=3, help="Kinematics samples per check.")
    ap.add_argument("--seed", type=int, default=123, help="Base seed for generation and checks.")
    ap.add_argument("--mass", type=float, default=2.0, help="Mass M for scalars in kinematics.")
    ap.add_argument("--tol-abs", type=float, default=1e-12, dest="tol_abs", help="Absolute tolerance.")
    ap.add_argument("--tol-rel", type=float, default=1e-10, dest="tol_rel", help="Relative tolerance.")
    ap.add_argument("--min-scr", type=int, default=0, dest="min_scr", help="Minimum scrambles during generation.")
    ap.add_argument("--max-scr", type=int, default=3, dest="max_scr", help="Maximum scrambles during generation.")
    ap.add_argument("--no-denominators", action="store_true", help="Generate without denominators (debug option).")
    ap.add_argument("--max-particles-tokenizer", type=int, default=16, help="Tokenizer family size for p_i/e_i/F_i/M_i.")
    args = ap.parse_args()

    # Build dataset with the provided seed. We keep generation validation on to ensure (simple,scrambled) are equivalent pre-tokenization.
    pairs = gd.build_dataset(
        args.N,
        num_samples=args.count,
        max_scr=args.max_scr,
        min_scr=args.min_scr,
        seed=args.seed,
        use_denominators=not args.no_denominators,
        validate=True,
        M=args.mass,
    )

    ok = check_roundtrip(
        pairs,
        samples=args.samples,
        mass_M=args.mass,
        tol_abs=args.tol_abs,
        tol_rel=args.tol_rel,
        seed=args.seed,
        max_particles_tokenizer=args.max_particles_tokenizer,
    )

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
