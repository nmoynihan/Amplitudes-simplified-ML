"""Re-run the historical sparse field-strength basis search.

This is symbolic preprocessing, not a learned model prediction. The candidate
family and ordering follow the 20 September 2026 ``derive_extended.py`` search.
A numerical point screens candidates; every accepted coefficient is solved
and verified exactly as a rational function of s and t. The search is optional
because normal preparation re-verifies its compact historical answer.
"""
from __future__ import annotations

import itertools
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp

from .algebra import (DEFAULT_REFERENCE, DEFAULT_SEED, dot, read_expression,
                      require_zero, rotate, s, symbolic, t)


@lru_cache(maxsize=None)
def trace(labels: tuple[int, ...]):
    value = 0
    for bits in itertools.product((0, 1), repeat=len(labels)):
        left = [("e" if bit else "p", j) for bit, j in zip(bits, labels)]
        right = [("p" if bit else "e", j) for bit, j in zip(bits, labels)]
        value += (-1) ** sum(bits) * sp.prod(
            dot(*right[i], *left[(i + 1) % len(labels)]) for i in range(len(labels)))
    return sp.expand(value)


@lru_cache(maxsize=None)
def chain(left: int, labels: tuple[int, ...], right: int):
    value = 0
    for bits in itertools.product((0, 1), repeat=len(labels)):
        ls = [("e" if bit else "p", j) for bit, j in zip(bits, labels)]
        rs = [("p" if bit else "e", j) for bit, j in zip(bits, labels)]
        value += ((-1) ** sum(bits) * dot("p", left, *ls[0]) *
                  sp.prod(dot(*rs[i], *ls[i + 1]) for i in range(len(labels) - 1)) *
                  dot(*rs[-1], "p", right))
    return sp.expand(value)


def trace_name(labels):
    return "Tr(" + "".join(map(str, labels)) + ")"


def chain_name(left, labels, right):
    return f"p{left}F" + "".join(map(str, labels)) + f"p{right}"


def candidate_basis():
    """Enumerate traces and products of momentum-ended field-strength chains."""
    candidates = [("Tr1234", trace((1, 2, 3, 4)))]
    pairings = (((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3)))
    candidates.extend((trace_name(i) + trace_name(j), trace(i) * trace(j)) for i, j in pairings)
    for size in (1, 2, 4):
        for subset in itertools.combinations(range(1, 5), size):
            rest = tuple(j for j in range(1, 5) if j not in subset)
            for shift in range(size):
                labels = subset[shift:] + subset[:shift]
                for left in range(1, 5):
                    for right in range(1, 5):
                        if left == labels[0] or right == labels[-1]:
                            continue
                        numerator = chain(left, labels, right)
                        if rest:
                            numerator *= trace(rest)
                        candidates.append((chain_name(left, labels, right) +
                                           (trace_name(rest) if rest else ""), sp.expand(numerator)))

    def chain_options(subset):
        options = []
        for shift in range(len(subset)):
            labels = subset[shift:] + subset[:shift]
            lefts = [j for j in range(1, 5) if j != labels[0]][:2]
            rights = [j for j in range(1, 5) if j != labels[-1]][:2]
            for left in lefts:
                for right in rights:
                    value = chain(left, labels, right)
                    if value != 0:
                        options.append((chain_name(left, labels, right), value))
        return options

    def add_products(parts):
        for blocks in itertools.product(*(chain_options(part) for part in parts)):
            candidates.append(("*".join(block[0] for block in blocks),
                               sp.expand(sp.prod(block[1] for block in blocks))))

    for leg in range(1, 5):
        add_products([tuple(j for j in range(1, 5) if j != leg), (leg,)])
    for other in (2, 3, 4):
        add_products([(1, other), tuple(j for j in range(1, 5) if j not in (1, other))])
    for subset in itertools.combinations(range(1, 5), 2):
        add_products([subset, *[(j,) for j in range(1, 5) if j not in subset]])
    add_products([(j,) for j in range(1, 5)])
    return candidates


def coefficient_matrix(expressions, variables):
    polynomials = [sp.Poly(sp.expand(expression), *variables).as_dict() for expression in expressions]
    monomials = sorted(set().union(*(polynomial for polynomial in polynomials)))
    return sp.Matrix([[polynomial.get(monomial, 0) for polynomial in polynomials]
                      for monomial in monomials])


def derive(output_dir: Path, seed: Path | None = None, reference: Path | None = None,
           max_solutions: int = 1) -> dict:
    """Search supports of sizes 1, 2, 3 and export the first exact solutions.

    ``coefficients`` multiply basis numerators in ``A*s*t``, as in the original
    search. ``sparse_extended_result.json`` is accepted by ``prepare``.
    """
    if max_solutions < 1:
        raise ValueError("max_solutions must be positive")
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_expression = read_expression(seed or DEFAULT_SEED)
    amplitude = symbolic(f"(({seed_expression})+({rotate(seed_expression)}))/2")
    require_zero(amplitude - symbolic(read_expression(reference or DEFAULT_REFERENCE)),
                 "Cyclic completion must equal the full reference before deriving a basis")
    target = sp.expand(amplitude * s * t)
    traces = [trace(labels) for labels in ((1, 2, 3, 4), (1, 3, 2, 4), (1, 2, 4, 3))]
    trace_products = [trace(a) * trace(b) for a, b in
                      (((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3)))]
    all_variables = sorted(set().union(*(x.free_symbols for x in traces + trace_products + [target])), key=str)
    trace_matrix = coefficient_matrix(traces + trace_products + [target], all_variables)
    trace_relation = sp.linsolve((trace_matrix[:, :-1], trace_matrix[:, -1]))
    restricted = sp.linsolve((trace_matrix[:, [0, 3, 4, 5]], trace_matrix[:, -1]))

    candidates = candidate_basis()
    variables = sorted(set().union(*(value.free_symbols for _, value in candidates), target.free_symbols) - {s, t}, key=str)
    matrix = coefficient_matrix([value for _, value in candidates] + [target], variables)
    print(f"Basis search: {len(candidates)} candidates, {matrix.rows} polarization monomials", flush=True)
    generic = matrix.subs({s: 2, t: 3})
    indices, seen = [], set()
    for index in range(len(candidates)):
        column = tuple(generic[:, index])
        first = next((x for x in column if x != 0), None)
        if first is None:
            continue
        key = tuple(x / first for x in column)
        if key not in seen:
            seen.add(key)
            indices.append(index)
    unique = generic[:, indices]
    _, rows = unique.T.rref()
    screen = np.asarray(unique[list(rows), :], dtype=float)
    rhs = np.asarray(generic[list(rows), -1], dtype=float).reshape(-1)
    print(f"Basis search: {len(indices)} distinct columns, rank {len(rows)}", flush=True)
    solutions, screened, exact_attempts = [], 0, 0
    for size in (1, 2, 3):
        print(f"Basis search: testing supports of size {size}", flush=True)
        for choice in itertools.combinations(range(len(indices)), size):
            screened += 1
            array = screen[:, choice]
            approximate = np.linalg.lstsq(array, rhs, rcond=None)[0]
            if np.linalg.norm(array @ approximate - rhs) > 1e-8:
                continue
            selected = [indices[index] for index in choice]
            exact_attempts += 1
            exact = sp.linsolve((matrix[:, selected], matrix[:, -1]))
            if exact == sp.EmptySet:
                continue
            coefficients = list(exact)[0]
            if any(coefficient.free_symbols - {s, t} for coefficient in coefficients):
                continue
            require_zero(sum(coeff * candidates[index][1] for index, coeff in zip(selected, coefficients)) - target,
                         "Derived basis representation")
            solution = {"indices": selected, "names": [candidates[index][0] for index in selected],
                        "coefficients": [str(coefficient) for coefficient in coefficients]}
            solutions.append(solution)
            print(f"Basis search: exact solution {solution['names']}", flush=True)
            if len(solutions) >= max_solutions:
                break
        if solutions:
            break
    if not solutions:
        raise RuntimeError("No exact representation of size at most three was found")
    result_path = output_dir / "sparse_extended_result.json"
    result_path.write_text(json.dumps(solutions, indent=2) + "\n", encoding="utf-8")
    manifest = {"method": "enumerated F-tensor basis; numerical support screening; exact rational solve",
                "target": "A*s*t", "candidate_count": len(candidates), "polarization_monomials": matrix.rows,
                "generic_screen_point": {"s": 2, "t": 3}, "distinct_columns": len(indices), "rank": len(rows),
                "supports_screened": screened, "exact_solve_attempts": exact_attempts,
                "trace_relation": str(trace_relation), "restricted_trace_relation": str(restricted),
                "solutions": solutions, "solution_file": str(result_path.resolve()),
                "exact_difference_from_completed_amplitude": "0", "seconds": time.monotonic() - started,
                "scope": "Sparse solution found using the original candidate family and generic-point screening; not a proof of global minimality."}
    (output_dir / "derivation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
