#!/usr/bin/env python3
"""Generate four-point Yang--Mills pairs with zero-free canonical targets.

This is a stricter front end around :mod:`data_gen_ym.generate`.  The original
generator validates ``simple == scrambled`` numerically, but equality alone
allows both sides to be zero and does not remove algebraically redundant terms.
For every candidate this module therefore:

1. canonicalises four-point scalar products, traces, and open F-chains;
2. cancels common rational factors and combines identical top-level terms;
3. removes exact and numerically proven zero subsets of the compact target;
4. rejects targets that are still numerically zero; and
5. explicitly groups dot chains before prefix tokenization, avoiding the
   historical ``·``/``*`` precedence ambiguity; and
6. validates the final target against the scrambled source on an independent
   deterministic kinematic grid before writing raw and tokenised CSV files.

The outputs are built as sibling temporary files and published only after the
requested number of accepted rows has been produced.

Recommended invocation from the repository root::

    python -m data_gen.data_gen_ym.generate_clean_4pt \
        --samples 500000 --seed 42 --jobs auto

The historical ``data_gen_ym`` top-level package layout is also supported when
running from ``data_gen/``::

    python -m data_gen_ym.generate_clean_4pt --samples 1000 --jobs 1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TextIO


MODULE_PATH = Path(__file__).resolve()
DATA_GEN_DIR = MODULE_PATH.parent.parent
REPO_ROOT = DATA_GEN_DIR.parent

# ``data_gen_ym.generate`` historically imports ``Tokenizer`` as a top-level
# module.  Adding data_gen itself keeps that import working for both supported
# module layouts and for spawned multiprocessing workers.
if str(DATA_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_DIR))

try:  # ``python -m data_gen.data_gen_ym.generate_clean_4pt``
    from ..Tokenizer import ScatteringAmplitudeTokenizer
    from ..filter_antisymmetry_zeros import (
        ExpressionSyntaxError,
        OnShellAssumptions,
        _join_signed_terms,
        _strip_term_prefix,
        product_factors,
        split_term_num_den,
        split_top_level_sum,
        zero_factor_reasons,
    )
    from ..numeric_utils import numeric_values_close
except ImportError:  # ``cd data_gen && python -m data_gen_ym...``
    from Tokenizer import ScatteringAmplitudeTokenizer
    from filter_antisymmetry_zeros import (
        ExpressionSyntaxError,
        OnShellAssumptions,
        _join_signed_terms,
        _strip_term_prefix,
        product_factors,
        split_term_num_den,
        split_top_level_sum,
        zero_factor_reasons,
    )
    from numeric_utils import numeric_values_close

from .generate import build_dataset_batched, _resolve_jobs
from .kinematics import generate_kinematics
from .algebra import _BinOp, _DotChain, _Num, _Parser, _UnaryOp, _Vec
from .notation import (
    DEFAULT_MAX_ATTEMPTS_FACTOR,
    DEFAULT_MAX_TERMS,
    DEFAULT_MIN_TERMS,
    DEFAULT_USE_DENOMINATORS,
    DENOM_REPEAT_PROBABILITY,
    OLD_STYLE_PROBABILITY,
    SCALAR_POWER_PROBABILITY,
    UNIT_PROBABILITY,
    _RE_pFchainp,
    _RE_pp,
    _RE_TrN,
)
from .numerics import _strict_tokenize, eval_infix_numeric
from .scramble import _SCRAMBLER_BY_NAME, normalise_scramble_names


N_PARTICLES = 4
DEFAULT_SAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_ENERGY_SCALE = 2.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_5PT_MAX_TOKENS = 4096
DEFAULT_CANDIDATE_BATCH_SIZE = 1000
DEFAULT_GENERATOR_BATCH_SIZE = 1000
DEFAULT_MAX_CANDIDATES_FACTOR = 5.0
DEFAULT_ZERO_CHECKS = 3
DEFAULT_VALIDATION_CHECKS = 3
DEFAULT_ZERO_SEED = 17011
DEFAULT_VALIDATION_SEED = 91009
DEFAULT_ZERO_TOL = 1e-12
DEFAULT_ZERO_REL_TOL = 1e-10
DEFAULT_TOL_ABS = 1e-10
DEFAULT_TOL_REL = 1e-8
DEFAULT_POL_MODES = ("coulomb", "covariant")
KINEMATIC_MODE_SEED_STRIDE = 100_003


@lru_cache(maxsize=None)
def _ym_assumptions(n_particles: int) -> OnShellAssumptions:
    return OnShellAssumptions.all_massless_ym(n_particles)


@dataclass(frozen=True, order=True)
class FactorKey:
    """Hashable exact representation of one canonical compact factor."""

    kind: str
    labels: tuple[int, ...]


@dataclass(frozen=True, order=True)
class RationalKey:
    """Canonical numerator and denominator exponent maps for one term."""

    numerator: tuple[tuple[FactorKey, int], ...]
    denominator: tuple[tuple[FactorKey, int], ...]


@dataclass(frozen=True)
class CanonicalizationResult:
    expression: str | None
    input_terms: int
    exact_zero_terms_removed: int
    combined_terms_removed: int


@dataclass(frozen=True)
class KinematicPoint:
    name: str
    momenta: Any
    polarisations: Any


@dataclass(frozen=True)
class PreparedPair:
    simple: str
    scrambled: str
    simple_tokens: list[int]
    scrambled_tokens: list[int]
    exact_zero_terms_removed: int
    numerical_zero_terms_removed: int
    combined_terms_removed: int


class TokenizationError(ValueError):
    """Raised when an otherwise valid pair cannot be tokenized faithfully."""


@dataclass
class GenerationStats:
    requested: int
    candidates_generated: int = 0
    accepted: int = 0
    generation_batches: int = 0
    exact_zero_terms_removed: int = 0
    numerical_zero_terms_removed: int = 0
    combined_terms_removed: int = 0
    exact_zero_targets_rejected: int = 0
    numerical_zero_targets_rejected: int = 0
    syntax_rejections: int = 0
    numerical_rejections: int = 0
    token_rejections: int = 0
    duplicate_rejections: int = 0


def _rotations(word: Sequence[int]) -> Iterator[tuple[int, ...]]:
    values = tuple(word)
    for offset in range(len(values)):
        yield values[offset:] + values[:offset]


def _validate_particle_labels(
    labels: Sequence[int],
    *,
    context: str,
    n_particles: int,
) -> None:
    invalid = tuple(label for label in labels if not 1 <= label <= n_particles)
    if invalid:
        raise ExpressionSyntaxError(
            f"{n_particles}-point {context} label out of range: {invalid}"
        )


def _choose_oriented_word(
    candidates: Iterable[tuple[tuple[int, ...], int]],
) -> tuple[int, tuple[int, ...]] | None:
    """Choose the smallest word, returning ``None`` if it equals its negative."""

    materialised = list(candidates)
    if not materialised:
        raise ValueError("an oriented word needs at least one candidate")
    best = min(word for word, _sign in materialised)
    signs = {sign for word, sign in materialised if word == best}
    if signs == {-1, 1}:
        return None
    return next(iter(signs)), best


def _canonical_trace(
    labels: Sequence[int],
    *,
    n_particles: int,
) -> tuple[int, FactorKey] | None:
    word = tuple(labels)
    if len(word) < 2:
        raise ExpressionSyntaxError("Tr requires at least two field strengths")
    _validate_particle_labels(
        word,
        context="trace",
        n_particles=n_particles,
    )
    reverse_sign = -1 if len(word) % 2 else 1
    oriented = [(rotation, 1) for rotation in _rotations(word)]
    oriented.extend(
        (rotation, reverse_sign)
        for rotation in _rotations(tuple(reversed(word)))
    )
    chosen = _choose_oriented_word(oriented)
    if chosen is None:
        return None
    sign, canonical_word = chosen
    return sign, FactorKey("trace", canonical_word)


def _canonical_chain(
    left: int,
    labels: Sequence[int],
    right: int,
    *,
    n_particles: int,
) -> tuple[int, FactorKey] | None:
    word = tuple(labels)
    if not word:
        raise ExpressionSyntaxError("an open F-chain needs at least one field strength")
    all_labels = (left, *word, right)
    _validate_particle_labels(
        all_labels,
        context="open-chain",
        n_particles=n_particles,
    )
    forward = (left, *word, right)
    reverse = (right, *reversed(word), left)
    reverse_sign = -1 if len(word) % 2 else 1
    chosen = _choose_oriented_word(((forward, 1), (reverse, reverse_sign)))
    if chosen is None:
        return None
    sign, canonical_word = chosen
    return sign, FactorKey("chain", canonical_word)


def _canonical_pp(
    left: int,
    right: int,
    *,
    n_particles: int,
) -> FactorKey | None:
    _validate_particle_labels(
        (left, right),
        context="momentum",
        n_particles=n_particles,
    )
    if left == right:
        return None  # every external momentum is massless
    pair = tuple(sorted((left, right)))
    if n_particles == 4:
        complement = tuple(
            label for label in range(1, n_particles + 1) if label not in pair
        )
        # Only at massless four point does a pair equal its complementary pair.
        pair = min(pair, tuple(sorted(complement)))
    return FactorKey("pp", pair)


def canonicalize_factor(
    factor: str,
    *,
    n_particles: int = N_PARTICLES,
) -> tuple[int, FactorKey] | None:
    """Return orientation sign/key, or ``None`` for a proven-zero factor."""

    text = factor.strip()
    match = _RE_pp.fullmatch(text)
    if match:
        key = _canonical_pp(
            int(match.group(1)),
            int(match.group(2)),
            n_particles=n_particles,
        )
        return None if key is None else (1, key)

    match = _RE_TrN.fullmatch(text)
    if match:
        labels = tuple(int(value) for value in re.findall(r"F_(\d+)", match.group(1)))
        _validate_particle_labels(
            labels,
            context="trace",
            n_particles=n_particles,
        )
        if zero_factor_reasons(
            text,
            assumptions=_ym_assumptions(n_particles),
        ):
            return None
        return _canonical_trace(labels, n_particles=n_particles)

    match = _RE_pFchainp.fullmatch(text)
    if match:
        labels = tuple(int(value) for value in re.findall(r"F_(\d+)", match.group(2)))
        left, right = int(match.group(1)), int(match.group(3))
        _validate_particle_labels(
            (left, *labels, right),
            context="open-chain",
            n_particles=n_particles,
        )
        # p_i·F_a·F_i·F_b·p_i vanishes for arbitrary antisymmetric F_a,F_b:
        # expanding F_i=p_i∧e_i leaves p_i·F_a·p_i or p_i·F_b·p_i.
        if (
            left == right
            and len(labels) == 3
            and labels[1] == left
        ):
            return None
        if zero_factor_reasons(
            text,
            assumptions=_ym_assumptions(n_particles),
        ):
            return None
        return _canonical_chain(
            left,
            labels,
            right,
            n_particles=n_particles,
        )

    raise ExpressionSyntaxError(f"unsupported compact Yang--Mills factor: {text!r}")


def _add_product_factors(
    product: str,
    factors: Counter[FactorKey],
    coefficient: Fraction,
    *,
    inverted: bool,
    n_particles: int,
) -> tuple[Fraction, bool]:
    if not product:
        return coefficient, False
    for base, power in product_factors(product):
        if re.fullmatch(r"\d+", base):
            if power == 0:
                continue
            numeric = Fraction(int(base)) ** power
            coefficient = coefficient / numeric if inverted else coefficient * numeric
            continue
        canonical = canonicalize_factor(base, n_particles=n_particles)
        if power == 0:
            continue
        if canonical is None:
            if inverted:
                raise ExpressionSyntaxError("zero factor appears in a denominator")
            return Fraction(0), True
        sign, key = canonical
        if sign < 0 and power % 2:
            coefficient *= -1
        factors[key] += power
    return coefficient, False


def _canonicalize_term(
    term: str,
    *,
    n_particles: int,
) -> tuple[Fraction, RationalKey] | None:
    integer_coefficient, body = _strip_term_prefix(term)
    coefficient = Fraction(integer_coefficient)
    numerator_text, denominator_text = split_term_num_den(body)
    numerator: Counter[FactorKey] = Counter()
    denominator: Counter[FactorKey] = Counter()

    coefficient, is_zero = _add_product_factors(
        numerator_text,
        numerator,
        coefficient,
        inverted=False,
        n_particles=n_particles,
    )
    if is_zero or coefficient == 0:
        return None
    coefficient, denominator_zero = _add_product_factors(
        denominator_text,
        denominator,
        coefficient,
        inverted=True,
        n_particles=n_particles,
    )
    if denominator_zero:
        raise ExpressionSyntaxError("zero denominator")

    for key in tuple(numerator):
        cancelled = min(numerator[key], denominator[key])
        if cancelled:
            numerator[key] -= cancelled
            denominator[key] -= cancelled
        if numerator[key] == 0:
            del numerator[key]
        if denominator[key] == 0:
            del denominator[key]

    return coefficient, RationalKey(
        tuple(sorted(numerator.items())),
        tuple(sorted(denominator.items())),
    )


def _render_factor(key: FactorKey) -> str:
    if key.kind == "pp":
        left, right = key.labels
        return f"p_{left} · p_{right}"
    if key.kind == "trace":
        return "Tr(" + " · ".join(f"F_{label}" for label in key.labels) + ")"
    if key.kind == "chain":
        left, *middle, right = key.labels
        return " · ".join(
            [f"p_{left}", *(f"F_{label}" for label in middle), f"p_{right}"]
        )
    raise AssertionError(f"unknown factor kind: {key.kind}")


def _render_power(key: FactorKey, power: int) -> str:
    factor = _render_factor(key)
    return factor if power == 1 else f"({factor})^{power}"


def _render_rational_term(coefficient: Fraction, key: RationalKey) -> str:
    magnitude = abs(coefficient)
    numerator_parts = [
        _render_power(factor, power) for factor, power in key.numerator
    ]
    denominator_parts = [
        _render_power(factor, power) for factor, power in key.denominator
    ]

    if magnitude.numerator != 1 or not numerator_parts:
        numerator_parts.insert(0, str(magnitude.numerator))
    if magnitude.denominator != 1:
        denominator_parts.insert(0, str(magnitude.denominator))

    numerator = "*".join(numerator_parts)
    if denominator_parts:
        return f"({numerator})/({'*'.join(denominator_parts)})"
    return numerator


def canonicalize_simple_expression(
    expression: str,
    *,
    n_particles: int = N_PARTICLES,
) -> CanonicalizationResult:
    """Canonicalise and combine a compact Yang--Mills expression exactly."""

    terms = split_top_level_sum(expression)
    combined: dict[RationalKey, Fraction] = {}
    zero_terms = 0
    nonzero_terms = 0
    for term in terms:
        canonical = _canonicalize_term(term, n_particles=n_particles)
        if canonical is None:
            zero_terms += 1
            continue
        coefficient, key = canonical
        combined[key] = combined.get(key, Fraction(0)) + coefficient
        nonzero_terms += 1

    combined = {key: value for key, value in combined.items() if value}
    if not combined:
        return CanonicalizationResult(None, len(terms), zero_terms, nonzero_terms)

    pieces: list[str] = []
    for key, coefficient in sorted(combined.items(), key=lambda item: item[0]):
        body = _render_rational_term(coefficient, key)
        if not pieces:
            pieces.append(body if coefficient > 0 else f"-{body}")
        else:
            pieces.append(("+ " if coefficient > 0 else "- ") + body)
    return CanonicalizationResult(
        " ".join(pieces),
        len(terms),
        zero_terms,
        max(0, nonzero_terms - len(combined)),
    )


def parenthesize_for_semantic_tokenization(expression: str) -> str:
    """Render the numerical parser's AST with every operation explicit.

    The historical tokenizer assigns ``·`` the same precedence as ``*`` and
    ``/``, while the Yang--Mills numerical grammar treats a complete dot chain
    as one atomic factor.  Fully parenthesizing the parsed tree makes the two
    grammars agree without changing the vocabulary or old-model tokenizer.
    """

    def render(node: Any) -> str:
        if isinstance(node, _Num):
            if not math.isfinite(node.value) or not node.value.is_integer():
                raise ExpressionSyntaxError(
                    "tokenized Yang--Mills data requires finite integer literals"
                )
            return str(int(node.value))
        if isinstance(node, _Vec):
            return f"{node.tag}_{node.idx}"
        if isinstance(node, _DotChain):
            is_trace = bool(node.parts) and node.parts[-1] is _DotChain._TR
            parts = node.parts[:-1] if is_trace else node.parts
            if not parts or any(not isinstance(part, _Vec) for part in parts):
                raise ExpressionSyntaxError("malformed dot chain during tokenization")
            body = " · ".join(render(part) for part in parts)
            return f"Tr({body})" if is_trace else f"({body})"
        if isinstance(node, _UnaryOp):
            if node.op != "-":
                raise ExpressionSyntaxError(
                    f"unsupported unary operator during tokenization: {node.op!r}"
                )
            return f"(-({render(node.operand)}))"
        if isinstance(node, _BinOp):
            if node.op not in {"+", "-", "*", "/", "**"}:
                raise ExpressionSyntaxError(
                    f"unsupported binary operator during tokenization: {node.op!r}"
                )
            if node.op == "*":
                factors: list[Any] = []

                def collect_factors(current: Any) -> None:
                    if isinstance(current, _BinOp) and current.op == "*":
                        collect_factors(current.left)
                        collect_factors(current.right)
                    else:
                        factors.append(current)

                collect_factors(node)
                return "(" + "*".join(render(factor) for factor in factors) + ")"
            if node.op == "**":
                # Power already binds more tightly than scalar arithmetic in
                # the tokenizer; avoiding a redundant outer pair also keeps
                # the compact canonicalizer's factor grammar idempotent.
                return f"{render(node.left)}^{render(node.right)}"
            return f"({render(node.left)}{node.op}{render(node.right)})"
        raise ExpressionSyntaxError(
            f"unsupported AST node during tokenization: {type(node).__name__}"
        )

    normalized_terms: list[str] = []
    for raw_term in split_top_level_sum(expression):
        term = raw_term.strip()
        sign = ""
        if term.startswith(("+", "-")):
            sign, term = term[0], term[1:].lstrip()
        if not term:
            raise ExpressionSyntaxError("empty signed term during tokenization")

        tokens = _strict_tokenize(term)
        parser = _Parser(tokens)
        tree = parser.parse()
        if parser.i != len(tokens):
            raise ExpressionSyntaxError(
                "cannot normalize expression for tokenization: parser did not "
                f"consume all tokens ({parser.i}/{len(tokens)})"
            )
        rendered = render(tree)
        if sign == "-":
            normalized_terms.append(f"-({rendered})")
        else:
            normalized_terms.append(sign + rendered)

    return _join_signed_terms(normalized_terms)


def build_kinematic_points(
    *,
    base_seed: int,
    checks_per_mode: int,
    energy_scale: float,
    pol_modes: Sequence[str] = DEFAULT_POL_MODES,
    n_particles: int = N_PARTICLES,
) -> tuple[KinematicPoint, ...]:
    points: list[KinematicPoint] = []
    for mode_index, pol_mode in enumerate(pol_modes):
        for check_index in range(checks_per_mode):
            seed = base_seed + KINEMATIC_MODE_SEED_STRIDE * mode_index + check_index
            momenta, polarisations = generate_kinematics(
                n_particles,
                E_scale=energy_scale,
                pol_mode=pol_mode,
                seed=seed,
            )
            points.append(
                KinematicPoint(
                    f"{pol_mode}:seed={seed}",
                    momenta,
                    polarisations,
                )
            )
    return tuple(points)


def _kinematic_seed_ranges(
    base_seed: int,
    checks_per_mode: int,
    pol_modes: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Return the inclusive RNG-seed ranges used by one kinematic grid."""

    return tuple(
        (
            base_seed + KINEMATIC_MODE_SEED_STRIDE * mode_index,
            base_seed
            + KINEMATIC_MODE_SEED_STRIDE * mode_index
            + checks_per_mode
            - 1,
        )
        for mode_index, _pol_mode in enumerate(pol_modes)
    )


def _shared_kinematic_seed(
    zero_seed: int,
    zero_checks: int,
    validation_seed: int,
    validation_checks: int,
    pol_modes: Sequence[str],
) -> int | None:
    """Return one shared RNG seed, or ``None`` when both grids are disjoint."""

    zero_seed_ranges = _kinematic_seed_ranges(zero_seed, zero_checks, pol_modes)
    validation_seed_ranges = _kinematic_seed_ranges(
        validation_seed,
        validation_checks,
        pol_modes,
    )
    return next(
        (
            max(zero_start, validation_start)
            for zero_start, zero_end in zero_seed_ranges
            for validation_start, validation_end in validation_seed_ranges
            if max(zero_start, validation_start)
            <= min(zero_end, validation_end)
        ),
        None,
    )


def evaluate_expression(
    expression: str,
    points: Sequence[KinematicPoint],
) -> tuple[float, ...]:
    values: list[float] = []
    for point in points:
        value = eval_infix_numeric(
            expression,
            point.momenta,
            point.polarisations,
            strict=True,
        )
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {point.name}")
        values.append(value)
    return tuple(values)


def numerically_zero(
    expression: str,
    points: Sequence[KinematicPoint],
    *,
    tolerance: float,
    relative_tolerance: float = DEFAULT_ZERO_REL_TOL,
) -> bool:
    terms = split_top_level_sum(expression)
    values_by_term = [evaluate_expression(term, points) for term in terms]
    for point_index in range(len(points)):
        term_values = [values[point_index] for values in values_by_term]
        residual = abs(math.fsum(term_values))
        scale = math.fsum(abs(value) for value in term_values)
        if residual > tolerance + relative_tolerance * scale:
            return False
    return True


def numerically_equivalent(
    expression_a: str,
    expression_b: str,
    points: Sequence[KinematicPoint],
    *,
    tol_abs: float,
    tol_rel: float,
) -> bool:
    values_a = evaluate_expression(expression_a, points)
    values_b = evaluate_expression(expression_b, points)
    return all(
        numeric_values_close(a, b, tol_abs=tol_abs, tol_rel=tol_rel)
        for a, b in zip(values_a, values_b)
    )


def remove_numerically_zero_subsets(
    expression: str,
    points: Sequence[KinematicPoint],
    *,
    tolerance: float,
    max_subset_terms: int,
    relative_tolerance: float = DEFAULT_ZERO_REL_TOL,
) -> tuple[str | None, int]:
    """Remove the largest proper zero subsets; return ``None`` if all is zero."""

    if numerically_zero(
        expression,
        points,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    ):
        return None, len(split_top_level_sum(expression))

    terms = split_top_level_sum(expression)
    if len(terms) > max_subset_terms:
        return expression, 0

    removed = 0
    while len(terms) > 1:
        found: tuple[int, ...] | None = None
        for subset_size in range(len(terms) - 1, 0, -1):
            for indices in itertools.combinations(range(len(terms)), subset_size):
                subset = _join_signed_terms([terms[index] for index in indices])
                if numerically_zero(
                    subset,
                    points,
                    tolerance=tolerance,
                    relative_tolerance=relative_tolerance,
                ):
                    found = indices
                    break
            if found is not None:
                break
        if found is None:
            break
        found_set = set(found)
        removed += len(found)
        terms = [term for index, term in enumerate(terms) if index not in found_set]

    return _join_signed_terms(terms), removed


def prepare_pair(
    simple: str,
    scrambled: str,
    *,
    tokenizer: ScatteringAmplitudeTokenizer,
    zero_points: Sequence[KinematicPoint],
    validation_points: Sequence[KinematicPoint],
    zero_tolerance: float,
    tol_abs: float,
    tol_rel: float,
    max_subset_terms: int,
    max_tokens: int | None,
    zero_relative_tolerance: float = DEFAULT_ZERO_REL_TOL,
    n_particles: int = N_PARTICLES,
) -> PreparedPair | None:
    """Clean and independently validate one generated candidate pair."""

    canonical = canonicalize_simple_expression(
        simple,
        n_particles=n_particles,
    )
    if canonical.expression is None:
        return None

    pruned, numerical_terms_removed = remove_numerically_zero_subsets(
        canonical.expression,
        zero_points,
        tolerance=zero_tolerance,
        max_subset_terms=max_subset_terms,
        relative_tolerance=zero_relative_tolerance,
    )
    if pruned is None:
        return None

    recanonicalized = canonicalize_simple_expression(
        pruned,
        n_particles=n_particles,
    )
    if recanonicalized.expression is None:
        return None
    cleaned_simple = recanonicalized.expression

    # Use both independent grids for the final nonzero gate.  The second grid
    # prevents a subset selected on the first grid from passing accidentally.
    if numerically_zero(
        cleaned_simple,
        (*zero_points, *validation_points),
        tolerance=zero_tolerance,
        relative_tolerance=zero_relative_tolerance,
    ):
        return None
    if not numerically_equivalent(
        cleaned_simple,
        scrambled,
        validation_points,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
    ):
        raise ValueError("cleaned target is not equivalent to scrambled source")

    token_safe_simple = parenthesize_for_semantic_tokenization(cleaned_simple)
    token_safe_scrambled = parenthesize_for_semantic_tokenization(scrambled)
    try:
        simple_tokens = tokenizer.encode_infix(token_safe_simple)
        scrambled_tokens = tokenizer.encode_infix(token_safe_scrambled)
    except ValueError as exc:
        raise TokenizationError(str(exc)) from exc
    if not simple_tokens or not scrambled_tokens:
        raise TokenizationError("tokenizer produced an empty sequence")
    if 1 in simple_tokens or 1 in scrambled_tokens:
        raise TokenizationError("tokenizer produced UNK token id 1")
    if max_tokens is not None and (
        len(simple_tokens) > max_tokens or len(scrambled_tokens) > max_tokens
    ):
        raise OverflowError("pair exceeds the configured token budget")

    return PreparedPair(
        token_safe_simple,
        token_safe_scrambled,
        simple_tokens,
        scrambled_tokens,
        canonical.exact_zero_terms_removed,
        numerical_terms_removed,
        canonical.combined_terms_removed + recanonicalized.combined_terms_removed,
    )


def _open_text(path: Path, mode: str, *, compressed: bool) -> TextIO:
    if compressed:
        return gzip.open(
            path,
            mode + "t",
            newline="",
            encoding="utf-8",
            compresslevel=6,
        )
    return path.open(mode, newline="", encoding="utf-8")


def _temporary_sibling(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _lexists(path: Path) -> bool:
    """Return whether a directory entry exists, including dangling symlinks."""

    return os.path.lexists(path)


def _absolute_output_path(path: Path) -> Path:
    """Return a normalized absolute path without following an output symlink."""

    return Path(os.path.abspath(path))


def _publish(temp_path: Path, destination: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temp_path, destination)
    else:
        try:
            os.link(temp_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"destination appeared during generation: {destination}"
            ) from exc
        temp_path.unlink()
    os.chmod(destination, 0o644)


def _publish_all(
    temporary_paths: Sequence[Path],
    destinations: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    """Publish a related set of files, restoring prior outputs on failure."""

    temps = tuple(Path(path) for path in temporary_paths)
    targets = tuple(Path(path) for path in destinations)
    if not temps or len(temps) != len(targets):
        raise ValueError("temporary paths and destinations must have equal nonzero length")

    resolved_temps = tuple(path.resolve(strict=False) for path in temps)
    resolved_targets = tuple(path.resolve(strict=False) for path in targets)
    if len(set(resolved_temps)) != len(resolved_temps):
        raise ValueError("temporary paths must be distinct")
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("destinations must be distinct")
    if set(resolved_temps) & set(resolved_targets):
        raise ValueError("temporary paths and destinations must not overlap")

    missing = [path for path in temps if not _lexists(path)]
    if missing:
        raise FileNotFoundError(
            "temporary output does not exist: " + ", ".join(str(path) for path in missing)
        )
    if not overwrite:
        existing = [path for path in targets if _lexists(path)]
        if existing:
            raise FileExistsError(
                "output already exists (use --overwrite to replace): "
                + ", ".join(str(path) for path in existing)
            )

    backups: dict[int, Path] = {}
    attempted: list[int] = []
    try:
        if overwrite:
            for index, target in enumerate(targets):
                if not _lexists(target):
                    continue
                backup = _temporary_sibling(target)
                try:
                    os.replace(target, backup)
                except BaseException:
                    try:
                        backup.unlink()
                    except FileNotFoundError:
                        pass
                    raise
                backups[index] = backup

        for index, (temp_path, target) in enumerate(zip(temps, targets)):
            attempted.append(index)
            _publish(temp_path, target, overwrite=overwrite)
    except BaseException as publish_error:
        recovery_errors: list[str] = []

        # A backup can replace a partially published destination directly.  For
        # destinations that were new, remove only files demonstrably published
        # by this transaction (or whose source temp has already been consumed).
        for index in attempted:
            if index in backups:
                continue
            target = targets[index]
            should_remove = not _lexists(temps[index])
            if not should_remove and _lexists(target):
                try:
                    should_remove = os.path.samefile(temps[index], target)
                except OSError:
                    should_remove = False
            if should_remove:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    recovery_errors.append(f"remove {target}: {exc}")

        for index, backup in backups.items():
            try:
                os.replace(backup, targets[index])
            except OSError as exc:
                recovery_errors.append(
                    f"restore {targets[index]} from {backup}: {exc}"
                )

        if recovery_errors:
            raise RuntimeError(
                "publication failed and rollback was incomplete: "
                + "; ".join(recovery_errors)
            ) from publish_error
        raise

    for backup in backups.values():
        try:
            backup.unlink()
        except FileNotFoundError:
            pass


def _pair_digest(simple: str, scrambled: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(simple.encode("utf-8"))
    digest.update(b"\0")
    digest.update(scrambled.encode("utf-8"))
    return digest.digest()


def _sha256_uncompressed(path: Path, *, compressed: bool) -> str:
    digest = hashlib.sha256()
    if compressed:
        handle_context = gzip.open(path, "rb")
    else:
        handle_context = path.open("rb")
    with handle_context as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output_paths(
    raw_path: Path,
    token_path: Path | None,
    report_path: Path,
    *,
    overwrite: bool,
) -> None:
    paths = [raw_path, report_path, *(tuple() if token_path is None else (token_path,))]
    resolved = [path.expanduser().resolve(strict=False) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("raw, token, and report output paths must be distinct")
    if not overwrite:
        existing = [path for path in paths if _lexists(path)]
        if existing:
            raise FileExistsError(
                "output already exists (use --overwrite to replace): "
                + ", ".join(str(path) for path in existing)
            )


def generate_to_files(
    args: argparse.Namespace,
    *,
    n_particles: int = N_PARTICLES,
    generator_name: str | None = None,
) -> tuple[GenerationStats, dict[str, Any]]:
    raw_path = Path(args.raw_out).expanduser()
    token_path = None if args.no_tokenise else Path(args.tok_out).expanduser()
    report_path = Path(args.report_out).expanduser()
    _validate_output_paths(
        raw_path,
        token_path,
        report_path,
        overwrite=args.overwrite,
    )
    shared_seed = _shared_kinematic_seed(
        args.zero_seed,
        args.zero_checks,
        args.validation_seed,
        args.validation_checks,
        args.validation_pol_modes,
    )
    if shared_seed is not None:
        raise ValueError(
            "zero-check and validation kinematic seed grids must not overlap; "
            f"shared seed {shared_seed}"
        )

    tokenizer = ScatteringAmplitudeTokenizer(
        max_particles=args.tokenizer_max_particles,
        max_sequence_length=None,
    )
    zero_points = build_kinematic_points(
        base_seed=args.zero_seed,
        checks_per_mode=args.zero_checks,
        energy_scale=args.energy_scale,
        pol_modes=args.validation_pol_modes,
        n_particles=n_particles,
    )
    validation_points = build_kinematic_points(
        base_seed=args.validation_seed,
        checks_per_mode=args.validation_checks,
        energy_scale=args.energy_scale,
        pol_modes=args.validation_pol_modes,
        n_particles=n_particles,
    )

    stats = GenerationStats(requested=args.samples)
    max_candidates = max(
        args.candidate_batch_size,
        math.ceil(args.samples * args.max_candidates_factor),
    )
    max_tokens = None if args.max_tokens <= 0 else args.max_tokens
    seen: set[bytes] = set()
    raw_temp = _temporary_sibling(raw_path)
    token_temp = _temporary_sibling(token_path) if token_path is not None else None
    started = time.perf_counter()

    try:
        with ExitStack() as stack:
            raw_handle = stack.enter_context(
                _open_text(raw_temp, "w", compressed=raw_path.suffix == ".gz")
            )
            raw_writer = csv.writer(raw_handle)
            raw_writer.writerow(["simple", "scrambled"])

            token_writer: csv.DictWriter[str] | None = None
            if token_temp is not None and token_path is not None:
                token_handle = stack.enter_context(
                    _open_text(
                        token_temp,
                        "w",
                        compressed=token_path.suffix == ".gz",
                    )
                )
                token_writer = csv.DictWriter(
                    token_handle,
                    fieldnames=["simple", "scrambled"],
                )
                token_writer.writeheader()

            while stats.accepted < args.samples:
                if stats.candidates_generated >= max_candidates:
                    raise RuntimeError(
                        f"accepted only {stats.accepted}/{args.samples} rows after "
                        f"{stats.candidates_generated} candidates; increase "
                        "--max-candidates-factor"
                    )
                request_count = min(
                    args.candidate_batch_size,
                    max_candidates - stats.candidates_generated,
                )
                batch_seed = args.seed + 10_000_019 * stats.generation_batches
                candidates = build_dataset_batched(
                    n_particles,
                    request_count,
                    dataset_kind="oneshot",
                    max_scr=args.max_scr,
                    min_scr=args.min_scr,
                    seed=batch_seed,
                    unit_probability=args.unit_probability,
                    old_style_probability=args.old_style_probability,
                    denom_repeat_probability=args.denom_repeat_probability,
                    scalar_power_probability=args.scalar_power_probability,
                    use_denominators=DEFAULT_USE_DENOMINATORS,
                    # The legacy validator draws unseeded kinematics.  Disable
                    # it here and use the fixed independent grids in
                    # ``prepare_pair`` so a seed reproduces the same dataset.
                    validate=False,
                    M=args.energy_scale,
                    min_terms=args.min_terms,
                    max_terms=args.max_terms,
                    max_attempts_factor=args.max_attempts_factor,
                    full_expand_scrambled=not args.grouped_scrambled,
                    max_tokens=max_tokens,
                    tokenizer_max_particles=args.tokenizer_max_particles,
                    validation_pol_modes=tuple(args.validation_pol_modes),
                    scramble_names=args.scrambles,
                    batch_size=min(args.generator_batch_size, request_count),
                    jobs=_resolve_jobs(args.jobs),
                    progress=False,
                )
                stats.generation_batches += 1
                stats.candidates_generated += len(candidates)
                if not candidates:
                    raise RuntimeError(
                        "the underlying generator returned no candidates for a batch"
                    )

                for simple, scrambled in sorted(candidates):
                    if stats.accepted >= args.samples:
                        break
                    try:
                        prepared = prepare_pair(
                            simple,
                            scrambled,
                            tokenizer=tokenizer,
                            zero_points=zero_points,
                            validation_points=validation_points,
                            zero_tolerance=args.zero_tol,
                            tol_abs=args.tol_abs,
                            tol_rel=args.tol_rel,
                            max_subset_terms=args.max_zero_subset_terms,
                            max_tokens=max_tokens,
                            zero_relative_tolerance=args.zero_rel_tol,
                            n_particles=n_particles,
                        )
                    except (OverflowError, TokenizationError):
                        stats.token_rejections += 1
                        continue
                    except (ExpressionSyntaxError, SyntaxError):
                        stats.syntax_rejections += 1
                        continue
                    except (ArithmeticError, KeyError, ValueError):
                        stats.numerical_rejections += 1
                        continue

                    if prepared is None:
                        # Distinguish exact from broader numerical zeros cheaply by
                        # rerunning only the exact canonicalizer.
                        try:
                            exact = canonicalize_simple_expression(
                                simple,
                                n_particles=n_particles,
                            )
                        except (ExpressionSyntaxError, ValueError):
                            stats.syntax_rejections += 1
                            continue
                        if exact.expression is None:
                            stats.exact_zero_targets_rejected += 1
                        else:
                            stats.numerical_zero_targets_rejected += 1
                        continue

                    digest = _pair_digest(prepared.simple, prepared.scrambled)
                    if digest in seen:
                        stats.duplicate_rejections += 1
                        continue
                    seen.add(digest)

                    raw_writer.writerow([prepared.simple, prepared.scrambled])
                    if token_writer is not None:
                        token_writer.writerow(
                            {
                                "simple": json.dumps(prepared.simple_tokens),
                                "scrambled": json.dumps(prepared.scrambled_tokens),
                            }
                        )
                    stats.accepted += 1
                    stats.exact_zero_terms_removed += prepared.exact_zero_terms_removed
                    stats.numerical_zero_terms_removed += prepared.numerical_zero_terms_removed
                    stats.combined_terms_removed += prepared.combined_terms_removed

                    if args.progress_every and stats.accepted % args.progress_every == 0:
                        print(
                            f"accepted {stats.accepted:,}/{args.samples:,} from "
                            f"{stats.candidates_generated:,} generated candidates",
                            file=sys.stderr,
                        )

        raw_content_sha256 = _sha256_uncompressed(
            raw_temp,
            compressed=raw_path.suffix == ".gz",
        )
        token_content_sha256 = (
            _sha256_uncompressed(
                token_temp,
                compressed=token_path.suffix == ".gz",
            )
            if token_temp is not None and token_path is not None
            else None
        )

        report: dict[str, Any] = {
            "report_schema_version": 2,
            "generator": generator_name or f"clean_{n_particles}pt_yang_mills",
            "raw_output": str(_absolute_output_path(raw_path)),
            "token_output": (
                str(_absolute_output_path(token_path)) if token_path else None
            ),
            "outputs": {
                "raw": {
                    "path": str(_absolute_output_path(raw_path)),
                    "rows": stats.accepted,
                    "sha256_uncompressed": raw_content_sha256,
                },
                "tokenized": (
                    {
                        "path": str(_absolute_output_path(token_path)),
                        "rows": stats.accepted,
                        "sha256_uncompressed": token_content_sha256,
                    }
                    if token_path is not None
                    else None
                ),
            },
            "settings": {
                "particles": n_particles,
                "samples": args.samples,
                "seed": args.seed,
                "min_terms": args.min_terms,
                "max_terms": args.max_terms,
                "min_scrambles": args.min_scr,
                "max_scrambles": args.max_scr,
                "scramble_names": list(normalise_scramble_names(args.scrambles)),
                "grouped_scrambled": args.grouped_scrambled,
                "unit_probability": args.unit_probability,
                "old_style_probability": args.old_style_probability,
                "denominator_repeat_probability": args.denom_repeat_probability,
                "scalar_power_probability": args.scalar_power_probability,
                "candidate_batch_size": args.candidate_batch_size,
                "generator_batch_size": args.generator_batch_size,
                "max_candidates_factor": args.max_candidates_factor,
                "max_attempts_factor": args.max_attempts_factor,
                "jobs_requested": str(args.jobs),
                "jobs_resolved": _resolve_jobs(args.jobs),
                "zero_seed": args.zero_seed,
                "validation_seed": args.validation_seed,
                "zero_checks_per_mode": args.zero_checks,
                "validation_checks_per_mode": args.validation_checks,
                "polarisation_modes": list(args.validation_pol_modes),
                "zero_tolerance": args.zero_tol,
                "zero_relative_tolerance": args.zero_rel_tol,
                "max_zero_subset_terms": args.max_zero_subset_terms,
                "absolute_tolerance": args.tol_abs,
                "relative_tolerance": args.tol_rel,
                "energy_scale": args.energy_scale,
                "max_tokens": max_tokens,
                "tokenizer_max_particles": args.tokenizer_max_particles,
                "tokenized_output_enabled": token_path is not None,
                "tokenization_normalization": "fully_parenthesized_numeric_ast_v1",
            },
            "stats": asdict(stats),
            "wall_time_seconds": time.perf_counter() - started,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_temp = _temporary_sibling(report_path)
        try:
            with report_temp.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")

            temporary_paths = [raw_temp]
            destinations = [raw_path]
            if token_temp is not None and token_path is not None:
                temporary_paths.append(token_temp)
                destinations.append(token_path)
            temporary_paths.append(report_temp)
            destinations.append(report_path)
            _publish_all(
                temporary_paths,
                destinations,
                overwrite=args.overwrite,
            )
            raw_temp = None
            if token_temp is not None:
                token_temp = None
        finally:
            if report_temp.exists():
                report_temp.unlink()
        return stats, report
    finally:
        for temp_path in (raw_temp, token_temp):
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass


def _default_paths(
    samples: int,
    *,
    n_particles: int = N_PARTICLES,
) -> tuple[Path, Path, Path]:
    stem = f"ym_{n_particles}pt_{samples}_canonical_nonzero"
    directory = REPO_ROOT / "data" / "data_ym"
    return (
        directory / f"{stem}.csv.gz",
        directory / f"{stem}_tok.csv.gz",
        directory / f"{stem}.report.json",
    )


def build_parser(
    *,
    n_particles: int = N_PARTICLES,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Generate canonical, numerically nonzero {n_particles}-point Yang--Mills "
            "simple/scrambled training pairs."
        )
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-terms", type=int, default=DEFAULT_MIN_TERMS)
    parser.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS)
    parser.add_argument("--min-scr", type=int, default=1)
    parser.add_argument("--max-scr", type=int, default=4)
    parser.add_argument("--energy-scale", type=float, default=DEFAULT_ENERGY_SCALE)
    parser.add_argument("--unit-probability", type=float, default=UNIT_PROBABILITY)
    parser.add_argument("--old-style-probability", type=float, default=OLD_STYLE_PROBABILITY)
    parser.add_argument(
        "--denom-repeat-probability",
        type=float,
        default=DENOM_REPEAT_PROBABILITY,
    )
    parser.add_argument(
        "--scalar-power-probability",
        type=float,
        default=SCALAR_POWER_PROBABILITY,
    )
    parser.add_argument(
        "--scrambles",
        nargs="*",
        default=None,
        choices=["all", "none", *list(_SCRAMBLER_BY_NAME)],
    )
    parser.add_argument("--grouped-scrambled", action="store_true")
    parser.add_argument("--jobs", default="auto")
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=DEFAULT_CANDIDATE_BATCH_SIZE,
    )
    parser.add_argument(
        "--generator-batch-size",
        type=int,
        default=DEFAULT_GENERATOR_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-candidates-factor",
        type=float,
        default=DEFAULT_MAX_CANDIDATES_FACTOR,
    )
    parser.add_argument(
        "--max-attempts-factor",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_FACTOR,
    )
    default_max_tokens = (
        DEFAULT_5PT_MAX_TOKENS
        if n_particles >= 5
        else DEFAULT_MAX_TOKENS
    )
    parser.add_argument("--max-tokens", type=int, default=default_max_tokens)
    parser.add_argument("--tokenizer-max-particles", type=int, default=8)
    parser.add_argument("--zero-checks", type=int, default=DEFAULT_ZERO_CHECKS)
    parser.add_argument(
        "--validation-checks",
        type=int,
        default=DEFAULT_VALIDATION_CHECKS,
    )
    parser.add_argument("--zero-seed", type=int, default=DEFAULT_ZERO_SEED)
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=DEFAULT_VALIDATION_SEED,
    )
    parser.add_argument("--zero-tol", type=float, default=DEFAULT_ZERO_TOL)
    parser.add_argument(
        "--zero-rel-tol",
        type=float,
        default=DEFAULT_ZERO_REL_TOL,
        help="scale-aware relative tolerance for numerical zero rejection",
    )
    parser.add_argument("--tol-abs", type=float, default=DEFAULT_TOL_ABS)
    parser.add_argument("--tol-rel", type=float, default=DEFAULT_TOL_REL)
    parser.add_argument(
        "--validation-pol-modes",
        nargs="+",
        choices=["coulomb", "covariant"],
        default=list(DEFAULT_POL_MODES),
    )
    parser.add_argument(
        "--max-zero-subset-terms",
        type=int,
        default=6,
        help=(
            "maximum target term count for exhaustive numerical subset pruning; "
            "must be at least --max-terms"
        ),
    )
    parser.add_argument("--raw-out", type=str, default=None)
    parser.add_argument("--tok-out", type=str, default=None)
    parser.add_argument("--report-out", type=str, default=None)
    parser.add_argument("--no-tokenise", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    n_particles: int = N_PARTICLES,
) -> None:
    positive_integer_fields = (
        "samples",
        "candidate_batch_size",
        "generator_batch_size",
        "max_attempts_factor",
        "zero_checks",
        "validation_checks",
        "max_zero_subset_terms",
    )
    for field in positive_integer_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if args.min_terms < 1 or args.max_terms < args.min_terms:
        parser.error("term bounds must satisfy 1 <= --min-terms <= --max-terms")
    if args.max_zero_subset_terms < args.max_terms:
        parser.error(
            "--max-zero-subset-terms must be at least --max-terms so every "
            "accepted target is exhaustively checked"
        )
    if args.tokenizer_max_particles < n_particles:
        parser.error(
            f"--tokenizer-max-particles must be at least {n_particles}"
        )
    if args.min_scr < 0 or args.max_scr < args.min_scr:
        parser.error("scramble bounds must satisfy 0 <= --min-scr <= --max-scr")
    if args.energy_scale <= 0 or not math.isfinite(args.energy_scale):
        parser.error("--energy-scale must be finite and positive")
    if args.max_candidates_factor < 1 or not math.isfinite(args.max_candidates_factor):
        parser.error("--max-candidates-factor must be finite and at least 1")
    probability_fields = (
        "unit_probability",
        "old_style_probability",
        "denom_repeat_probability",
        "scalar_power_probability",
    )
    for field in probability_fields:
        value = getattr(args, field)
        if not 0 <= value <= 1:
            parser.error(f"--{field.replace('_', '-')} must lie in [0, 1]")
    for field in ("zero_tol", "zero_rel_tol", "tol_abs", "tol_rel"):
        value = getattr(args, field)
        if value < 0 or not math.isfinite(value):
            parser.error(f"--{field.replace('_', '-')} must be finite and non-negative")
    if args.zero_tol == 0:
        parser.error("--zero-tol must be positive")
    if args.tol_abs == 0 and args.tol_rel == 0:
        parser.error("at least one equivalence tolerance must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    shared_seed = _shared_kinematic_seed(
        args.zero_seed,
        args.zero_checks,
        args.validation_seed,
        args.validation_checks,
        args.validation_pol_modes,
    )
    if shared_seed is not None:
        parser.error(
            "zero-check and validation kinematic seed grids must not overlap; "
            f"shared seed {shared_seed}"
        )


def main_for_particles(
    n_particles: int,
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser(n_particles=n_particles)
    args = parser.parse_args(argv)
    _validate_args(args, parser, n_particles=n_particles)
    default_raw, default_tok, default_report = _default_paths(
        args.samples,
        n_particles=n_particles,
    )
    args.raw_out = args.raw_out or str(default_raw)
    args.tok_out = args.tok_out or str(default_tok)
    args.report_out = args.report_out or str(default_report)

    try:
        stats, report = generate_to_files(
            args,
            n_particles=n_particles,
        )
    except (ArithmeticError, csv.Error, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"accepted {stats.accepted:,} clean pairs from "
        f"{stats.candidates_generated:,} generated candidates"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_particles(N_PARTICLES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
