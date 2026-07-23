"""Gravity-aware identity scramblers for expanded dot-product expressions."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .. import gen_data as sqed
from .core import DOT, ProcessSpec, dot, e, p

SCRAMBLE_NAMES = (
    "multiply_one",
    "ward",
    "momentum",
    "commute_dot",
    "ratio",
    "on_shell_zero",
    "partial_fraction",
    "term_reorder",
)


def _call_legacy(function: Callable, rng: random.Random, *args) -> str:
    """Run a legacy scrambler reproducibly without leaking global RNG state."""
    state = random.getstate()
    random.seed(rng.randrange(2**63))
    try:
        return function(*args)
    finally:
        random.setstate(state)


def multiply_one(expr: str, spec: ProcessSpec, rng: random.Random) -> str:
    a, b = rng.sample(range(1, 6), 2)
    factor = dot(p(a), p(b))
    return f"({expr})*({factor})/({factor})"


def multiply_ratio(expr: str, spec: ProcessSpec, rng: random.Random) -> str:
    a, b, c, d = rng.sample(range(1, 6), 4)
    factor = f"({dot(p(a), p(b))} + {dot(p(c), p(d))})"
    return f"({expr})*{factor}/{factor}"


def ward_substitute(expr: str, spec: ProcessSpec, rng: random.Random) -> str:
    matches: list[tuple[re.Match, int, int]] = []
    for match in sqed._RE_DOT.finditer(expr):
        left, right = match.group(1), match.group(2)
        lm, rm = re.fullmatch(r"([pe])_(\d+)", left), re.fullmatch(r"([pe])_(\d+)", right)
        if not lm or not rm or lm.group(1) == rm.group(1):
            continue
        if lm.group(1) == "e":
            graviton, leg = int(lm.group(2)), int(rm.group(2))
        else:
            graviton, leg = int(rm.group(2)), int(lm.group(2))
        if graviton in spec.graviton_legs:
            matches.append((match, graviton, leg))
    if not matches:
        return expr
    match, graviton, leg = rng.choice(matches)
    replacement = "-(" + " + ".join(
        dot(e(graviton), p(k)) for k in range(1, 6) if k != leg
    ) + ")"
    return expr[: match.start()] + replacement + expr[match.end() :]


def momentum_substitute(expr: str, spec: ProcessSpec, rng: random.Random) -> str:
    matches = list(sqed._RE_pp.finditer(expr))
    if not matches:
        return expr
    match = rng.choice(matches)
    a, b = int(match.group(1)), int(match.group(2))
    replacement = "-(" + " + ".join(
        dot(p(k), p(b)) for k in range(1, 6) if k != a
    ) + ")"
    return expr[: match.start()] + replacement + expr[match.end() :]


def on_shell_zero(expr: str, spec: ProcessSpec, rng: random.Random) -> str:
    """Add a dimensionless multiple of ``p_i^2=0`` times the expression."""
    leg = rng.randint(1, 5)
    candidates = [
        match.group(0)
        for match in sqed._RE_pp.finditer(expr)
        if match.group(1) != match.group(2)
    ]
    if not candidates:
        a, b = rng.sample(range(1, 6), 2)
        denominator = dot(p(a), p(b))
    else:
        denominator = rng.choice(candidates)
    zero_ratio = f"({dot(p(leg), p(leg))})/({denominator})"
    sign = rng.choice(("+", "-"))
    return f"({expr}) {sign} ({expr})*{zero_ratio}"


def commute_dot(expr: str, spec: ProcessSpec, rng: random.Random) -> str:
    return _call_legacy(sqed.scr_commute_dot, rng, expr)


def partial_fraction(expr: str, spec: ProcessSpec, rng: random.Random) -> str:
    return _call_legacy(sqed.scr_partial_fraction, rng, expr)


def term_reorder(expr: str, spec: ProcessSpec, rng: random.Random) -> str:
    return _call_legacy(sqed.scr_term_reorder, rng, expr)


_SCRAMBLERS: dict[str, Callable[[str, ProcessSpec, random.Random], str]] = {
    "multiply_one": multiply_one,
    "ward": ward_substitute,
    "momentum": momentum_substitute,
    "commute_dot": commute_dot,
    "ratio": multiply_ratio,
    "on_shell_zero": on_shell_zero,
    "partial_fraction": partial_fraction,
    "term_reorder": term_reorder,
}


def normalise_names(names: Sequence[str] | None) -> tuple[str, ...]:
    if names is None:
        return SCRAMBLE_NAMES
    output: list[str] = []
    for item in names:
        for name in str(item).split(","):
            name = name.strip()
            if not name or name == "none":
                continue
            if name == "all":
                output.extend(SCRAMBLE_NAMES)
            elif name not in _SCRAMBLERS:
                raise ValueError(
                    f"Unknown gravity scrambler {name!r}; choose from {SCRAMBLE_NAMES}"
                )
            else:
                output.append(name)
    return tuple(dict.fromkeys(output))


@dataclass(frozen=True)
class ScrambleStep:
    depth: int
    label: str
    expression: str


def scramble_trajectory(
    expr: str,
    spec: ProcessSpec,
    *,
    rng: random.Random,
    depth: int,
    names: Sequence[str] | None = None,
    max_chars: int = 160_000,
) -> tuple[ScrambleStep, ...]:
    """Return effective, fully expanded scramble steps."""
    active = normalise_names(names)
    # Callers pass the already-expanded expression. Avoid reparsing/distributing
    # it once more before the first identity, which matters at 100k scale.
    output = expr
    steps: list[ScrambleStep] = []
    if not active:
        return tuple(steps)
    attempts = 0
    while len(steps) < depth and attempts < max(20, depth * 12):
        attempts += 1
        label = rng.choice(active)
        candidate = _SCRAMBLERS[label](output, spec, rng)
        try:
            candidate = sqed.full_expand_expression(candidate)
        except Exception:
            continue
        if candidate == output or len(candidate) > max_chars:
            continue
        output = candidate
        steps.append(ScrambleStep(len(steps) + 1, label, output))
    return tuple(steps)


def scramble(
    expr: str,
    spec: ProcessSpec,
    *,
    rng: random.Random,
    min_depth: int = 1,
    max_depth: int = 5,
    names: Sequence[str] | None = None,
    max_chars: int = 160_000,
) -> tuple[str, tuple[ScrambleStep, ...]]:
    depth = rng.randint(min_depth, max_depth)
    steps = scramble_trajectory(
        expr, spec, rng=rng, depth=depth, names=names, max_chars=max_chars
    )
    output = steps[-1].expression if steps else sqed.full_expand_expression(expr)
    return output, steps
