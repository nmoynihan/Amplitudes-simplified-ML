#!/usr/bin/env python3
"""
relabel.py — Relabel particle indices in amplitude expressions for data augmentation.

Supports:
    • Arbitrary permutations of photon labels {2, …, N−1}  (the full S_{N−2})
    • Scalar swap  1 ↔ N
    • Predefined cyclic shifts for backward compatibility

Usage:
    # Apply a random photon permutation
    python relabel.py input.csv output.csv --N 6 --mode random

    # Apply all cyclic permutations (gives N−2 augmented copies)
    python relabel.py input.csv output.csv --N 6 --mode cyclic

    # Apply a specific permutation (space-separated: target of 2,3,4,5 → 3,5,2,4)
    python relabel.py input.csv output.csv --N 6 --perm 3 5 2 4

    # Also swap the two scalar legs 1 ↔ N
    python relabel.py input.csv output.csv --N 6 --mode random --swap-scalars
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import random
import re
from itertools import permutations
from typing import Sequence

# Matches e_#, p_#, F_# where # is one or more digits
_PATTERN = re.compile(r"\b([epF])_(\d+)\b")


def _build_idx_map(perm_of_photons: Sequence[int], N: int, swap_scalars: bool = False) -> dict[str, str]:
    """Build a digit→digit map from a permutation of photon labels.

    perm_of_photons: a sequence of length N−2 giving the *targets* for
        photon labels [2, 3, …, N−1].  E.g. for N=6, perm_of_photons=(3,5,2,4)
        means 2→3, 3→5, 4→2, 5→4.
    """
    photon_labels = list(range(2, N))
    assert len(perm_of_photons) == len(photon_labels)
    idx_map: dict[str, str] = {}
    for src, dst in zip(photon_labels, perm_of_photons):
        idx_map[str(src)] = str(dst)
    if swap_scalars:
        idx_map["1"] = str(N)
        idx_map[str(N)] = "1"
    return idx_map


def _apply_map(text: str, idx_map: dict[str, str]) -> str:
    def _repl(m: re.Match) -> str:
        var, digit = m.group(1), m.group(2)
        return f"{var}_{idx_map.get(digit, digit)}"
    return _PATTERN.sub(_repl, text)


def remap_csv(
    in_path: str,
    out_path: str,
    idx_map: dict[str, str],
) -> None:
    """Read a CSV, remap indices, write a new CSV."""
    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    remapped = [[_PATTERN.sub(lambda m: f"{m.group(1)}_{idx_map.get(m.group(2), m.group(2))}", cell)
                  for cell in row] for row in rows]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(remapped)


def remap_csv_multi(
    in_path: str,
    out_path: str,
    idx_maps: list[dict[str, str]],
) -> None:
    """Apply multiple permutations, concatenating all augmented copies into one CSV."""
    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    header = rows[0]
    data_rows = rows[1:] if rows[0][0].strip().lower() in ("simple", "scrambled", "scambled") else rows

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if rows[0] != data_rows[0]:
            w.writerow(header)
        for idx_map in idx_maps:
            for row in data_rows:
                w.writerow([_apply_map(cell, idx_map) for cell in row])


def cyclic_permutations(N: int) -> list[dict[str, str]]:
    """All cyclic permutations of photon labels {2, …, N−1}."""
    photons = list(range(2, N))
    n = len(photons)
    maps: list[dict[str, str]] = []
    for shift in range(n):
        perm = photons[shift:] + photons[:shift]
        maps.append(_build_idx_map(perm, N))
    return maps


def random_permutation(N: int, swap_scalars: bool = False) -> dict[str, str]:
    """A single random permutation of photon labels."""
    photons = list(range(2, N))
    perm = photons[:]
    random.shuffle(perm)
    return _build_idx_map(perm, N, swap_scalars)


def all_permutations(N: int) -> list[dict[str, str]]:
    """All (N−2)! permutations of photon labels (can be large!)."""
    photons = list(range(2, N))
    return [_build_idx_map(list(p), N) for p in permutations(photons)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relabel amplitude expressions for data augmentation.")
    parser.add_argument("input_csv", type=pathlib.Path)
    parser.add_argument("output_csv", type=pathlib.Path)
    parser.add_argument("--N", type=int, required=True, help="Total number of external legs.")
    parser.add_argument("--mode", choices=["random", "cyclic", "all", "fixed"],
                        default="fixed",
                        help="Permutation mode (default: fixed = single cyclic shift for compat).")
    parser.add_argument("--perm", type=int, nargs="+", default=None,
                        help="Explicit permutation targets for photon labels 2…N−1.")
    parser.add_argument("--swap-scalars", action="store_true",
                        help="Also swap scalar legs 1 ↔ N.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    N = args.N

    if args.perm:
        # Explicit permutation
        idx_map = _build_idx_map(args.perm, N, args.swap_scalars)
        remap_csv(str(args.input_csv), str(args.output_csv), idx_map)

    elif args.mode == "cyclic":
        maps = cyclic_permutations(N)
        remap_csv_multi(str(args.input_csv), str(args.output_csv), maps)

    elif args.mode == "random":
        idx_map = random_permutation(N, args.swap_scalars)
        remap_csv(str(args.input_csv), str(args.output_csv), idx_map)

    elif args.mode == "all":
        maps = all_permutations(N)
        print(f"Generating all {len(maps)} permutations — this may be large.")
        remap_csv_multi(str(args.input_csv), str(args.output_csv), maps)

    else:
        # Legacy: single cyclic shift  (1→5, 2→1, 3→2, 4→3, 5→4  for N=6)
        photons = list(range(2, N))
        shifted = photons[1:] + photons[:1]
        idx_map = _build_idx_map(shifted, N, swap_scalars=args.swap_scalars)
        # Also shift scalar 1 → N in the old convention
        idx_map["1"] = str(N)
        idx_map[str(N)] = str(N - 1)
        remap_csv(str(args.input_csv), str(args.output_csv), idx_map)

    print(f"Done: {args.input_csv} → {args.output_csv}")
