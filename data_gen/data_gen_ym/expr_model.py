"""expr_model — extracted from gen_data.py (scaffold, verbatim)."""

from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import Sequence

from notation import *
from algebra import *


class _AnsatzInfeasible(ValueError):
    """The randomly chosen block/pole ansatz can't be realised — caller should retry.

    Distinct from a plain ValueError (which signals a genuine bug / invariant
    violation that must propagate, not be silently retried).
    """


@dataclass(frozen=True)
class BlockSpec:
    kind: str
    photons: tuple[int, ...]
    left: int | None = None
    right: int | None = None


@dataclass(frozen=True)
class MonomialSpec:
    numerator: str
    blocks: tuple[BlockSpec, ...]
    scalar_pairs: int
    numerator_mass_dim: int


def _rw_pFchainp(*idxs: int) -> str:
    """Expand p_i · F_{j1} · … · F_{jn} · p_k into dot products."""
    i, *Fs, k = idxs
    terms: list[tuple[int, str]] = []
    for mask in range(1 << len(Fs)):
        sign = 1 if bin(mask).count("1") % 2 == 0 else -1
        factors: list[str] = []
        prev = ("p", i)
        for bit, j in enumerate(Fs):
            swap = (mask >> bit) & 1
            left = ("e", j) if swap else ("p", j)
            right = ("p", j) if swap else ("e", j)
            factors.append(f"({dot(_vec(*prev), _vec(*left))})")
            prev = right
        factors.append(f"({dot(_vec(*prev), p(k))})")
        terms.append((sign, "*".join(factors)))
    return _format_signed_sum(terms)


def _rw_TrN(*js: int) -> str:
    """Expand Tr(F_{j1} · … · F_{jn}) into dot products."""
    terms: list[tuple[int, str]] = []
    for mask in range(1 << len(js)):
        sign = 1 if bin(mask).count("1") % 2 == 0 else -1
        pairs: list[tuple[str, str]] = []
        for bit, j in enumerate(js):
            if (mask >> bit) & 1:
                pairs.append((e(j), p(j)))
            else:
                pairs.append((p(j), e(j)))
        factors = [
            f"({dot(pairs[i][1], pairs[(i + 1) % len(js)][0])})"
            for i in range(len(js))
        ]
        terms.append((sign, "*".join(factors)))
    return _format_signed_sum(terms)


def rewrite_gi(block: str) -> str:
    block = block.strip()

    m = _RE_TrN.fullmatch(block)
    if m:
        js = [int(x) for x in re.findall(r"F_(\d+)", m.group(1))]
        return _rw_TrN(*js)

    m = _RE_pFchainp.fullmatch(block)
    if m:
        i = int(m.group(1))
        Fs = [int(x) for x in re.findall(r"F_(\d+)", m.group(2))]
        k = int(m.group(3))
        return _rw_pFchainp(i, *Fs, k)

    if _RE_pp.fullmatch(block):
        return block
    return block


def expand_simple_term(simple_term: str) -> str:
    """Expand one GI term into e·p / e·e / p·p form."""
    simple_term = simple_term.strip()
    if "/" in simple_term:
        num, den = simple_term.split("/", 1)
        num = num.strip()
        den = den.strip()
        if num.startswith("(") and num.endswith(")"):
            num = num[1:-1]
        if den.startswith("(") and den.endswith(")"):
            den = den[1:-1]
        exp_num = "*".join(rewrite_gi(f) for f in num.split("*"))
        return f"({exp_num})/({den})"
    return "*".join(rewrite_gi(f) for f in simple_term.split("*"))


def _chain_endpoints(Fs: Sequence[int], N: int) -> tuple[int, int]:
    """Choose endpoints for p·F...F·p chains without over-restricting them.

    Gauge invariance only forces the immediately adjacent contractions
    p_j·F_j and F_j·p_j to vanish for a massless transverse photon j.  Older
    data sources also allow endpoints to be photons appearing elsewhere inside
    the chain, and allow the two endpoints to be the same momentum.  Therefore
    we exclude only the first photon from the left endpoint and only the last
    photon from the right endpoint.
    """
    if not Fs:
        raise ValueError("F-chain must contain at least one photon")
    first, last = Fs[0], Fs[-1]
    left_pool = [x for x in range(1, N + 1) if x != first]
    right_pool = [x for x in range(1, N + 1) if x != last]
    return random.choice(left_pool), random.choice(right_pool)


def _singleF_block(j: int, N: int) -> tuple[str, BlockSpec]:
    left, right = _chain_endpoints((j,), N)
    return (
        f"{p(left)} {DOT} {F(j)} {DOT} {p(right)}",
        BlockSpec("chain", (j,), left, right),
    )


def _doubleF_block(j: int, k: int, N: int) -> tuple[str, BlockSpec]:
    left, right = _chain_endpoints((j, k), N)
    return (
        f"{p(left)} {DOT} {F(j)} {DOT} {F(k)} {DOT} {p(right)}",
        BlockSpec("chain", (j, k), left, right),
    )


def _tripleF_block(j: int, k: int, l: int, N: int) -> tuple[str, BlockSpec]:
    left, right = _chain_endpoints((j, k, l), N)
    return (
        f"{p(left)} {DOT} {F(j)} {DOT} {F(k)} {DOT} {F(l)} {DOT} {p(right)}",
        BlockSpec("chain", (j, k, l), left, right),
    )


def _tr2_block(j: int, k: int) -> tuple[str, BlockSpec]:
    return Tr(F(j), F(k)), BlockSpec("trace", (j, k))


def _tr3_block(j: int, k: int, l: int) -> tuple[str, BlockSpec]:
    return Tr(F(j), F(k), F(l)), BlockSpec("trace", (j, k, l))


def _tr4_block(j: int, k: int, l: int, m: int) -> tuple[str, BlockSpec]:
    return Tr(F(j), F(k), F(l), F(m)), BlockSpec("trace", (j, k, l, m))


def _scalar_pp_factor(N: int) -> str:
    i, j = random.sample(range(1, N + 1), 2)
    return dot(p(i), p(j))


def _block_mass_dimension(block: BlockSpec) -> int:
    """Mass dimension with [p]=1 and [F]=1, matching the paper's counting."""
    if block.kind == "trace":
        return len(block.photons)
    if block.kind == "chain":
        return len(block.photons) + 2
    raise ValueError(f"Unknown block kind {block.kind}")


def _all_physical_poles(N: int) -> list[str]:
    """Colour-ordered planar poles: the N cyclically-adjacent 2-particle channels.

    For a colour-ordered amplitude A(1,…,N) the only allowed propagator poles are
    s_{i,i+1,…} for cyclically-consecutive leg sets.  For massless gluons the
    2-particle adjacent channel s_{i,i+1} = 2 p_i·p_{i+1}, i.e. the single dot
    p_i·p_{i+1}.  These are the only poles at 5 points (3-particle channels equal
    the complementary 2-particle ones by momentum conservation).

    NOTE: N>=6 also has genuine multi-particle poles (p_i+…+p_j)^2 that are *sums*
    of dots and cannot be represented as a single factor here — deferred.
    """
    pool: list[str] = []
    seen: set[str] = set()
    for i in range(1, N + 1):
        j = i % N + 1  # next leg cyclically (N -> 1)
        term = _canon_pp(dot(p(i), p(j)))
        if term not in seen:
            seen.add(term)
            pool.append(term)
    return pool


def _required_denominator_count(numerator_mass_dim: int, N: int) -> int | None:
    target_dim = 4 - N
    delta = numerator_mass_dim - target_dim
    if delta < 0 or delta % 2 != 0:
        return None
    return delta // 2


def _weighted_choice(weight_map: dict[str, int]) -> str:
    choices: list[str] = []
    for key, weight in weight_map.items():
        if weight > 0:
            choices.extend([key] * int(weight))
    if not choices:
        raise ValueError("At least one block-choice weight must be positive")
    return random.choice(choices)


def _block_choice_weights(N: int, remaining_count: int, *, old_style_blocks: bool) -> dict[str, int]:
    """Return editable block weights compatible with the remaining photons."""
    if N == 4:
        base = OLD_STYLE_N4_BLOCK_WEIGHTS if old_style_blocks else N4_BLOCK_WEIGHTS
    else:
        base = GENERAL_BLOCK_WEIGHTS

    allowed = {"singleF"}
    if remaining_count >= 2:
        allowed.update({"tr2", "doubleF"})
    if remaining_count >= 3:
        allowed.update({"tr3", "tripleF"})
    if remaining_count >= 4:
        allowed.add("tr4")
    return {kind: int(weight) for kind, weight in base.items() if kind in allowed and int(weight) > 0}


def _generate_gi_monomial_spec(
    N: int,
    *,
    old_style_blocks: bool = False,
    scalar_power_probability: float = SCALAR_POWER_PROBABILITY,
) -> MonomialSpec:
    remaining = gluon_legs(N)
    random.shuffle(remaining)
    factors: list[str] = []
    blocks: list[BlockSpec] = []

    while remaining:
        r = len(remaining)
        kind = _weighted_choice(_block_choice_weights(N, r, old_style_blocks=old_style_blocks))

        if kind == "tr4":
            chosen = random.sample(remaining, 4)
            block_str, spec = _tr4_block(*chosen)
        elif kind == "tr3":
            chosen = random.sample(remaining, 3)
            block_str, spec = _tr3_block(*chosen)
        elif kind == "tripleF":
            chosen = random.sample(remaining, 3)
            block_str, spec = _tripleF_block(*chosen, N)
        elif kind == "tr2":
            chosen = random.sample(remaining, 2)
            block_str, spec = _tr2_block(*chosen)
        elif kind == "doubleF":
            chosen = random.sample(remaining, 2)
            block_str, spec = _doubleF_block(*chosen, N)
        else:
            chosen = [remaining[-1]]
            block_str, spec = _singleF_block(chosen[0], N)

        factors.append(block_str)
        blocks.append(spec)
        remaining = [x for x in remaining if x not in chosen]

    base_mass_dim = sum(_block_mass_dimension(block) for block in blocks)
    max_scalar_pairs = 2 if N <= 5 else 4
    min_chain_poles = sum(len(block.photons) for block in blocks if block.kind == "chain")
    n_chains = sum(1 for block in blocks if block.kind == "chain")
    # Denominator capacity = Σ max_allowed = |physical| + Σ cancel_budget.
    # Each chain block contributes 2 endpoint poles (left + right), each worth one
    # pole-slot, so Σ cancel_budget = 2·n_chains exactly.
    capacity = len(_all_physical_poles(N)) + 2 * n_chains
    candidates: list[tuple[int, int]] = []
    for scalar_pairs in range(max_scalar_pairs + 1):
        numerator_mass_dim = base_mass_dim + 2 * scalar_pairs
        denom_count = _required_denominator_count(numerator_mass_dim, N)
        if denom_count is None:
            continue
        if denom_count < min_chain_poles:
            continue
        if denom_count > capacity:
            continue
        candidates.append((scalar_pairs, numerator_mass_dim))
    if not candidates:
        raise _AnsatzInfeasible(f"Could not realise manifest dimension 4-{N} with current ansatz")

    scalar_pairs, numerator_mass_dim = random.choice(candidates)

    # Add optional scalar p·p numerator factors.  With nonzero
    # scalar_power_probability, preferentially repeat an existing physical pole
    # factor.  After canonicalisation this deliberately creates numerator powers
    # such as (p_2 · p_4)^2, which can then support a spurious repeated
    # denominator factor without creating a physical double pole.
    scalar_power_probability = max(0.0, min(1.0, scalar_power_probability))
    scalar_factors: list[str] = []
    physical_scalar_pool = _all_physical_poles(N)
    for _ in range(scalar_pairs):
        repeatable = [term for term in scalar_factors if term in physical_scalar_pool]
        if repeatable and random.random() < scalar_power_probability:
            scalar_factors.append(random.choice(repeatable))
            continue
        if physical_scalar_pool and random.random() < 0.75:
            scalar_factors.append(random.choice(physical_scalar_pool))
        else:
            scalar_factors.append(_canon_pp(_scalar_pp_factor(N)))
    factors.extend(scalar_factors)

    random.shuffle(factors)
    return MonomialSpec(
        numerator=canonicalise_gi_product("*".join(factors)),
        blocks=tuple(blocks),
        scalar_pairs=scalar_pairs,
        numerator_mass_dim=numerator_mass_dim,
    )


def _dot_legs(term: str) -> list[int]:
    """The two leg indices in a canonical p_i·p_j pole string."""
    return [int(x) for x in re.findall(r"p_(\d+)", term)]


def _chain_endpoint_pole_budget(spec: MonomialSpec, N: int) -> dict[str, int]:
    """Cancellation budget per pole = number of F-chain endpoints that expose it.

    For a chain ``p_a · F_j · … · F_k · p_b`` the F-expansion yields the endpoint
    factors ``p_a · p_j`` (left, via ``p_a·F_j = (p_a·p_j) e_j − (p_a·e_j) p_j``) and
    ``p_k · p_b`` (right).  In the gauge ``p_a·e_j = 0`` such a factor is a clean common
    factor that cancels one power of that denominator pole.  Each endpoint cancels at
    most one power, so we accumulate a count.

    Unlike the sQED ``_chain_expansion_spurious_pp_counts`` there is NO physical-pool
    filter: surfacing the *non-adjacent* (unphysical-but-cancellable) endpoint poles is
    exactly the point.
    """
    budget: dict[str, int] = {}

    def add(a: int | None, b: int | None) -> None:
        if a is None or b is None or a == b:
            return
        term = _canon_pp(dot(p(a), p(b)))
        budget[term] = budget.get(term, 0) + 1

    for block in spec.blocks:
        if block.kind != "chain" or not block.photons:
            continue
        add(block.left, block.photons[0])
        add(block.right, block.photons[-1])
    return budget


def _physical_denominator_factors(
    spec: MonomialSpec,
    N: int,
    *,
    repeat_probability: float = DENOM_REPEAT_PROBABILITY,
) -> list[str]:
    """Build the colour-ordered denominator pole list.

    Allowed multiplicity of a pole ``D``::

        max_allowed(D) = (1 if D is an adjacent/physical channel else 0) + cancel_budget(D)

    where ``cancel_budget(D)`` counts the F-chain endpoints exposing ``D`` (each cancels
    one power).  Physical (adjacent ``p_i·p_{i+1}``) poles get one free genuine power;
    unphysical (non-adjacent) poles get zero free power and must be fully paid for by
    chain endpoints.  ``repeat_probability`` biases *optional* repeats only — it never
    blocks a repeat that is *required* to reach the target denominator count.
    """
    budget = _chain_endpoint_pole_budget(spec, N)
    physical = set(_all_physical_poles(N))
    selectable = sorted(physical | set(budget))

    counts: dict[str, int] = {}
    factors: list[str] = []

    def max_allowed(term: str) -> int:
        return (1 if term in physical else 0) + budget.get(term, 0)

    def can_add(term: str) -> bool:
        return counts.get(term, 0) < max_allowed(term)

    def add(term: str) -> None:
        if not can_add(term):
            raise _AnsatzInfeasible(f"pole {term} exceeds max_allowed {max_allowed(term)}")
        counts[term] = counts.get(term, 0) + 1
        factors.append(term)

    target = _required_denominator_count(spec.numerator_mass_dim, N)
    if target is None:
        raise _AnsatzInfeasible(
            f"numerator mass dimension {spec.numerator_mass_dim} cannot give 4-{N}"
        )

    # Mandatory support: each photon in a p·F…·p chain gets one pole on its momentum.
    for block in spec.blocks:
        if block.kind != "chain":
            continue
        for photon in block.photons:
            choices = [D for D in selectable if photon in _dot_legs(D) and can_add(D)]
            if not choices:
                raise _AnsatzInfeasible(f"no addable pole supporting photon {photon}")
            add(random.choice(choices))

    if len(factors) > target:
        raise _AnsatzInfeasible(
            f"mandatory chain poles ({len(factors)}) exceed target ({target})"
        )

    # Fill to target.  max_allowed is the sole authority: repeat when forced, and take an
    # *optional* repeat (a fresh pole still available) only with `repeat_probability`.
    repeat_probability = max(0.0, min(1.0, repeat_probability))
    while len(factors) < target:
        fresh = [D for D in selectable if can_add(D) and counts.get(D, 0) == 0]
        repeats = [D for D in selectable if can_add(D) and counts.get(D, 0) >= 1]
        if fresh and (not repeats or random.random() >= repeat_probability):
            add(random.choice(fresh))
        elif repeats:
            add(random.choice(repeats))
        elif fresh:
            add(random.choice(fresh))
        else:
            raise _AnsatzInfeasible(
                f"cannot reach {target} denominator factors (capacity exhausted)"
            )

    return factors


def _term_signature(spec: MonomialSpec, denom_factors: Sequence[str]) -> tuple:
    trace_lengths = sorted(len(block.photons) for block in spec.blocks if block.kind == "trace")
    chain_lengths = sorted(len(block.photons) for block in spec.blocks if block.kind == "chain")
    denom_support = sorted(
        tuple(sorted(map(int, re.findall(r"\d+", factor)))) for factor in denom_factors
    )
    return (
        tuple(trace_lengths),
        tuple(chain_lengths),
        spec.scalar_pairs,
        spec.numerator_mass_dim,
        tuple(denom_support),
    )


def _generate_term(
    N: int,
    *,
    use_denominators: bool,
    old_style_blocks: bool = False,
    denom_repeat_probability: float = DENOM_REPEAT_PROBABILITY,
    scalar_power_probability: float = SCALAR_POWER_PROBABILITY,
) -> tuple[str, str, tuple]:
    spec = _generate_gi_monomial_spec(
        N,
        old_style_blocks=old_style_blocks,
        scalar_power_probability=scalar_power_probability,
    )
    den_factors = (
        _physical_denominator_factors(
            spec,
            N,
            repeat_probability=denom_repeat_probability,
        )
        if use_denominators
        else []
    )

    denominator = canonicalise_denominator("*".join(den_factors))
    simple_num = spec.numerator
    expanded_num = "*".join(rewrite_gi(f) for f in simple_num.split("*"))

    if denominator:
        simple_term = f"({simple_num})/({denominator})"
        expanded_term = f"({expanded_num})/({denominator})"
    else:
        simple_term = simple_num
        expanded_term = expanded_num
    return simple_term, expanded_term, _term_signature(spec, den_factors)


def _has_supported_physical_poles(simple_term: str) -> bool:
    if "/" not in simple_term:
        return True
    num, den = simple_term.split("/", 1)
    den_factors = {
        _canon_pp(f.strip())
        for f in den.strip()[1:-1].split("*")
        if f.strip()
    }
    blocks = [f.strip() for f in num.strip()[1:-1].split("*") if f.strip()]
    for block in blocks:
        match = _RE_pFchainp.fullmatch(block)
        if not match:
            continue
        photons = [int(x) for x in re.findall(r"F_(\d+)", match.group(2))]
        for photon in photons:
            if not any(re.search(fr"\bp_{photon}\b", factor) for factor in den_factors):
                return False
    return True


def manifest_mass_dimension(simple_term: str) -> int:
    """Return the manifest mass dimension of a simple GI expression.

    Integer coefficients are treated as dimensionless. For sums, every term must
    have the same manifest dimension; otherwise a ValueError is raised.
    """

    def split_top_level_sum(expr: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and ch in "+-" and i > 0:
                prev = expr[i - 1]
                if prev not in "*/^(+":
                    parts.append("".join(current).strip())
                    current = [ch]
                    continue
            current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return [part for part in parts if part]

    def count_product_dim(expr: str) -> int:
        expr = _strip_matched_outer_parens(expr)
        factors = [f.strip() for f in _split_top_level(expr, "*") if f.strip()]
        total = 0
        for factor in factors:
            factor = _strip_matched_outer_parens(factor)
            
            # Handle powers: (base)^n
            power_match = re.fullmatch(r"\((.+)\)\^(\d+)", factor)
            if power_match:
                base = power_match.group(1)
                power = int(power_match.group(2))
                # Count dimension of base and multiply by power
                base_dim = count_product_dim(base)
                total += base_dim * power
                continue
            
            if re.fullmatch(r"-?\d+(?:\.\d+)?", factor):
                continue
            match = _RE_TrN.fullmatch(factor)
            if match:
                total += len(re.findall(r"F_\d+", match.group(1)))
                continue
            match = _RE_pFchainp.fullmatch(factor)
            if match:
                total += len(re.findall(r"F_\d+", match.group(2))) + 2
                continue
            if _RE_pp.fullmatch(factor):
                total += 2
                continue
            if "*" in factor:
                total += count_product_dim(factor)
                continue
            raise ValueError(f"Unrecognised factor in manifest dimension count: {factor}")
        return total

    term_dims: list[int] = []
    for term in split_top_level_sum(simple_term.strip()):
        expr = term.lstrip("+-").strip()
        if "/" in expr:
            num, den = expr.split("/", 1)
            num_dim = count_product_dim(num)
            den_body = _strip_matched_outer_parens(den)
            denom_dim = count_product_dim(den_body)
        else:
            num_dim = count_product_dim(expr)
            denom_dim = 0
        term_dims.append(num_dim - denom_dim)

    if not term_dims:
        raise ValueError("Empty expression")
    if len(set(term_dims)) != 1:
        raise ValueError(f"Inconsistent term dimensions: {term_dims}")
    return term_dims[0]


def _build_base_expression(
    N: int,
    *,
    unit_probability: float,
    old_style_probability: float,
    denom_repeat_probability: float,
    scalar_power_probability: float,
    use_denominators: bool,
    min_terms: int,
    max_terms: int,
) -> tuple[str, str] | None:
    use_old_style = random.random() < max(0.0, min(1.0, old_style_probability))
    if use_old_style:
        n_terms = 2
        use_unit_coeffs = True
    else:
        n_terms = random.randint(min_terms, max_terms)
        use_unit_coeffs = random.random() < max(0.0, min(1.0, unit_probability))

    first_simple = first_expanded = None
    for _attempt in range(80):
        try:
            cand_s, cand_e, _signature = _generate_term(
                N,
                use_denominators=use_denominators,
                old_style_blocks=use_old_style,
                denom_repeat_probability=denom_repeat_probability,
                scalar_power_probability=scalar_power_probability,
            )
        except _AnsatzInfeasible:
            # Infeasible partition (e.g. a chain-heavy block split needs more
            # poles than the budget allows). Retry. Genuine bugs (plain
            # ValueError) propagate instead of being silently swallowed.
            continue
        if not _has_supported_physical_poles(cand_s):
            continue
        if manifest_mass_dimension(cand_s) != 4 - N:
            continue
        first_simple, first_expanded = cand_s, cand_e
        break
    if first_simple is None:
        return None

    simple_terms = [first_simple]
    expanded_terms = [first_expanded]
    coeffs = [random.choice((-1, 1)) if use_unit_coeffs else random.choice(SCALAR_COEFF_POOL)]

    for _ in range(n_terms - 1):
        for _attempt in range(80):
            try:
                cand_simple, cand_expanded, _cand_signature = _generate_term(
                    N,
                    use_denominators=use_denominators,
                    old_style_blocks=use_old_style,
                    denom_repeat_probability=denom_repeat_probability,
                    scalar_power_probability=scalar_power_probability,
                )
            except _AnsatzInfeasible:
                continue  # infeasible ansatz; retry (genuine bugs propagate)
            if not _has_supported_physical_poles(cand_simple):
                continue
            if manifest_mass_dimension(cand_simple) != 4 - N:
                continue
            simple_terms.append(cand_simple)
            expanded_terms.append(cand_expanded)
            coeffs.append(random.choice((-1, 1)) if use_unit_coeffs else random.choice(TERM_COEFF_POOL))
            break
        else:
            return None

    return _format_poly(simple_terms, coeffs), _format_poly(expanded_terms, coeffs)

__all__ = [
    'BlockSpec',
    'MonomialSpec',
    '_rw_pFchainp',
    '_rw_TrN',
    'rewrite_gi',
    'expand_simple_term',
    '_chain_endpoints',
    '_singleF_block',
    '_doubleF_block',
    '_tripleF_block',
    '_tr2_block',
    '_tr3_block',
    '_tr4_block',
    '_scalar_pp_factor',
    '_block_mass_dimension',
    '_all_physical_poles',
    '_required_denominator_count',
    '_weighted_choice',
    '_block_choice_weights',
    '_generate_gi_monomial_spec',
    '_AnsatzInfeasible',
    '_dot_legs',
    '_chain_endpoint_pole_budget',
    '_physical_denominator_factors',
    '_term_signature',
    '_generate_term',
    '_has_supported_physical_poles',
    'manifest_mass_dimension',
    '_build_base_expression',
]
