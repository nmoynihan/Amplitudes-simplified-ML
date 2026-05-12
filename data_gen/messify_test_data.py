#!/usr/bin/env python3
"""
messify_test_data.py — Convert clean test CSV into training-distribution form.

The training data produced by gen_data.py uses Python-style "**" for powers
and has polarisation vectors appearing in both orders within dot products
(both "e_i · p_j" and "p_j · e_i"). The sqed_oneshot_150.csv test set is in
a cleaner canonical form ("^" for powers, polarisations always first).

This script mechanically rewrites the SCRAMBLED column to mimic the training
distribution, leaving SIMPLE (the targets) untouched. Use it as a sanity
check: if your model's accuracy jumps significantly on the messified version,
the diagnosis (token-level distribution mismatch) is confirmed.

Usage:
    python messify_test_data.py sqed_oneshot_150.csv messified.csv
    python messify_test_data.py sqed_oneshot_150.csv messified.csv \
        --swap-prob 0.5 --seed 0
"""
import argparse
import csv
import random
import re
from pathlib import Path

# Match an "e_i · p_j" pair (polarisation first). We will randomly swap some
# of these to "p_j · e_i" to mimic the order-mixed training distribution.
POL_DOT_RE = re.compile(r"(e_\d+)\s*·\s*(p_\d+)")


def messify(expr: str, swap_prob: float, rng: random.Random) -> str:
    """Convert one expression from clean form to training-like form."""
    # 1. "^" -> "**" everywhere (training uses Python-style powers).
    out = expr.replace("^", "**")

    # 2. Randomly flip "e_i · p_j" to "p_j · e_i" with probability swap_prob.
    #    This mimics the unsorted polarisation orders the scramblers produce.
    def _maybe_swap(m: re.Match) -> str:
        if rng.random() < swap_prob:
            return f"{m.group(2)} · {m.group(1)}"
        return m.group(0)

    out = POL_DOT_RE.sub(_maybe_swap, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("input_csv", type=Path,
                    help="Clean test CSV with columns (simple, scrambled).")
    ap.add_argument("output_csv", type=Path,
                    help="Output CSV in training-like form.")
    ap.add_argument("--swap-prob", type=float, default=0.5,
                    help="Probability of swapping each 'e_i · p_j' pair "
                         "(default 0.5 -> ~half end up reversed, similar "
                         "mix to training).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_in = n_swapped = 0

    with args.input_csv.open(newline="", encoding="utf-8") as fin, \
         args.output_csv.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        if reader.fieldnames is None or set(reader.fieldnames) != {"simple", "scrambled"}:
            raise SystemExit(
                f"Expected columns ('simple', 'scrambled'); got {reader.fieldnames}"
            )

        writer = csv.DictWriter(fout, fieldnames=["simple", "scrambled"])
        writer.writeheader()

        for row in reader:
            n_in += 1
            new_scrambled = messify(row["scrambled"], args.swap_prob, rng)
            if "**" in new_scrambled or new_scrambled != row["scrambled"]:
                n_swapped += 1
            writer.writerow({
                "simple": row["simple"],          # targets unchanged
                "scrambled": new_scrambled,       # inputs mangled to look training-like
            })

    print(f"Wrote {n_in} rows -> {args.output_csv} "
          f"({n_swapped} scrambled inputs were modified).")


if __name__ == "__main__":
    main()
