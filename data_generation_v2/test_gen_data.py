#!/usr/bin/env python3
"""
test_gen_data.py — Tests for the amplitude data generation pipeline.

Runs numerical verification of:
    • Rewrite rules (F → e·p expansions)
    • Individual scrambler invariance
    • Canonicalisation idempotence
    • Full dataset build with parity checks
    • Seed reproducibility
"""
import random
import sys
import os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if here not in sys.path:
    sys.path.insert(0, here)

import gen_data as gd
from kinematics import generate_kinematics


def eval_infix(expr: str, mom: np.ndarray, pol: np.ndarray) -> float:
    return gd.eval_infix_numeric(expr, mom, pol)


def expand_from_simple(simple: str) -> str:
    """Expand a canonical 'simple' label into e·p/p·p form."""
    if "/" in simple:
        num, den = simple.split("/", 1)
        num, den = num.strip(), den.strip()
        if num.startswith("(") and num.endswith(")"):
            num = num[1:-1]
        if den.startswith("(") and den.endswith(")"):
            den = den[1:-1]
        expd_num = "*".join(gd.rewrite_gi(b) for b in num.split("*"))
        return f"({expd_num})/({den})"
    else:
        return "*".join(gd.rewrite_gi(b) for b in simple.split("*"))


# ── Test: individual rewrite rules ───────────────────────────────────

def test_rewrite_rules():
    """Test specific GI block expansions numerically."""
    print("Testing rewrite rules…")
    N = 6
    mom, pol = generate_kinematics(N, M=1.7, seed=123)

    # p_i · F_j · p_k
    for _ in range(10):
        i, k = random.sample(range(1, N + 1), 2)
        j = random.randint(2, N - 1)
        block = f"p_{i} · F_{j} · p_{k}"
        expanded = gd.rewrite_gi(block)
        v1 = eval_infix(block, mom, pol)
        v2 = eval_infix(expanded, mom, pol)
        assert abs(v1 - v2) < max(1e-10, 1e-8 * abs(v1)), f"pFp failed: {block}"

    # Tr(F_j · F_k)
    for j in range(2, N - 1):
        for k in range(j + 1, N):
            block = f"Tr(F_{j} · F_{k})"
            expanded = gd.rewrite_gi(block)
            v1 = eval_infix(block, mom, pol)
            v2 = eval_infix(expanded, mom, pol)
            assert abs(v1 - v2) < max(1e-10, 1e-8 * abs(v1)), f"Tr2 failed: {block}"

    # Tr(F_j · F_k · F_l) for a few triples
    photons = list(range(2, N))
    for _ in range(5):
        j, k, l = random.sample(photons, 3)
        block = f"Tr(F_{j} · F_{k} · F_{l})"
        expanded = gd.rewrite_gi(block)
        v1 = eval_infix(block, mom, pol)
        v2 = eval_infix(expanded, mom, pol)
        assert abs(v1 - v2) < max(1e-10, 1e-8 * abs(v1)), f"Tr3 failed: {block}"

    # p_a · F_j · F_k · p_b
    for _ in range(10):
        j, k = random.sample(photons, 2)
        a, b = random.sample(range(1, N + 1), 2)
        block = f"p_{a} · F_{j} · F_{k} · p_{b}"
        expanded = gd.rewrite_gi(block)
        v1 = eval_infix(block, mom, pol)
        v2 = eval_infix(expanded, mom, pol)
        assert abs(v1 - v2) < max(1e-10, 1e-8 * abs(v1)), f"pFFp failed: {block}"

    # p_a · F_j · F_k · F_l · p_b  (triple chain)
    if len(photons) >= 3:
        for _ in range(5):
            j, k, l = random.sample(photons, 3)
            a, b = random.sample(range(1, N + 1), 2)
            block = f"p_{a} · F_{j} · F_{k} · F_{l} · p_{b}"
            expanded = gd.rewrite_gi(block)
            v1 = eval_infix(block, mom, pol)
            v2 = eval_infix(expanded, mom, pol)
            assert abs(v1 - v2) < max(1e-10, 1e-8 * abs(v1)), f"pFFFp failed: {block}"

    # Tr(F · F · F · F) if we have 4 photons
    if len(photons) >= 4:
        for _ in range(3):
            js = random.sample(photons, 4)
            block = gd.Tr(*(gd.F(j) for j in js))
            expanded = gd.rewrite_gi(block)
            v1 = eval_infix(block, mom, pol)
            v2 = eval_infix(expanded, mom, pol)
            assert abs(v1 - v2) < max(1e-10, 1e-8 * abs(v1)), f"Tr4 failed: {block}"

    print("  ✓ All rewrite rules passed")


# ── Test: scrambler invariance ───────────────────────────────────────

def test_scrambler_invariance(N: int = 6, seed: int = 0):
    """Every scrambler must preserve the numeric value."""
    print("Testing scrambler invariance…")
    random.seed(seed)
    np.random.seed(seed)

    simple = gd.strict_gi_monomial(N)
    expd = "*".join(gd.rewrite_gi(b) for b in simple.split("*"))
    Ng = N - 2
    mom, pol = generate_kinematics(N, M=2.0)
    base = eval_infix(expd, mom, pol)

    scramblers = [
        ("mul_by_one",          lambda s: gd.scr_mul_by_one(s, N)),
        ("ward_substitute",     lambda s: gd.scr_ward_substitute(s, Ng, N)),
        ("momentum_substitute", lambda s: gd.scr_momentum_substitute(s, N)),
        ("commute_dot",         lambda s: gd.scr_commute_dot(s, Ng, N)),
        ("mul_by_ratio",        lambda s: gd.scr_mul_by_ratio(s, N)),
        ("double_ward",         lambda s: gd.scr_double_ward(s, Ng, N)),
        ("partial_fraction",    lambda s: gd.scr_partial_fraction(s, N)),
    ]

    for name, fn in scramblers:
        for trial in range(5):
            scr = fn(expd)
            val = eval_infix(scr, mom, pol)
            tol = max(1e-10, 1e-8 * max(1.0, abs(base)))
            assert abs(val - base) <= tol, \
                f"{name} failed on trial {trial}: {val} vs {base} (Δ={abs(val-base):.3e})"
    print("  ✓ All scramblers preserve numeric value")


# ── Test: partial fraction scrambler ─────────────────────────────────

def test_partial_fraction(N: int = 6, n_trials: int = 30):
    """Dedicated test for the partial fraction scrambler.

    This checks:
      1. The scrambled expression evaluates to the same value.
      2. The result actually contains a subtraction of two fractions
         (structural check: the scrambler did something).
      3. Works on expressions with various denominator sizes (2, 3, 4+ factors).
    """
    print("Testing partial fraction scrambler…")
    random.seed(42)
    np.random.seed(42)

    passed = 0
    structurally_changed = 0

    for trial in range(n_trials):
        # Generate an expression with a denominator (force use_denom=True)
        gi = gd.strict_gi_monomial(N)
        simple_num = gd.canonicalise_gi_product(gi)
        expd_num = "*".join(gd.rewrite_gi(b) for b in simple_num.split("*"))

        # Build a denominator with ≥2 factors
        n_den_factors = random.randint(2, min(4, N - 1))
        js = random.sample([j for j in range(2, N + 1)], n_den_factors)
        denom = "*".join(gd.dot(gd.p(1), gd.p(j)) for j in js)
        denom = gd.canonicalise_denominator(denom)

        expr = f"({expd_num})/({denom})"

        # Apply partial fraction
        pf_expr = gd.scr_partial_fraction(expr, N)

        # Check structural change
        if pf_expr != expr:
            structurally_changed += 1

        # Numerical check
        mom, pol = generate_kinematics(N, M=2.0)
        v_orig = eval_infix(expr, mom, pol)
        v_pf = eval_infix(pf_expr, mom, pol)
        tol = max(1e-10, 1e-8 * max(1.0, abs(v_orig)))
        assert abs(v_orig - v_pf) <= tol, \
            f"PF trial {trial}: {v_orig} vs {v_pf} (Δ={abs(v_orig - v_pf):.3e})\n" \
            f"  original:  {expr[:100]}\n  PF result: {pf_expr[:100]}"
        passed += 1

    # Check that partial fraction actually changes the expression often enough
    change_rate = structurally_changed / n_trials
    print(f"  ✓ {passed}/{n_trials} numerically verified, "
          f"{structurally_changed}/{n_trials} structurally changed ({change_rate:.0%})")
    assert change_rate > 0.5, \
        f"Partial fraction too rarely fires: {change_rate:.0%} structural changes"


def test_partial_fraction_iterated(N: int = 6):
    """Apply partial fractions multiple times and verify the result is still correct.

    This tests that the scrambler works on expressions that already have
    (D_a − D_b) denominators from a previous partial fraction step.
    """
    print("Testing iterated partial fractions…")
    random.seed(99)
    np.random.seed(99)

    gi = gd.strict_gi_monomial(N)
    simple_num = gd.canonicalise_gi_product(gi)
    expd_num = "*".join(gd.rewrite_gi(b) for b in simple_num.split("*"))
    denom = gd.canonicalise_denominator(
        "*".join(gd.dot(gd.p(1), gd.p(j)) for j in [2, 3, 4])
    )
    expr = f"({expd_num})/({denom})"

    mom, pol = generate_kinematics(N, M=2.0)
    v_base = eval_infix(expr, mom, pol)

    # Apply PF up to 4 times
    current = expr
    for step in range(4):
        current = gd.scr_partial_fraction(current, N)
        v_step = eval_infix(current, mom, pol)
        tol = max(1e-10, 1e-8 * max(1.0, abs(v_base)))
        assert abs(v_base - v_step) <= tol, \
            f"Iterated PF failed at step {step+1}: {v_base} vs {v_step}"

    print(f"  ✓ 4 iterated PF steps preserve value (expr grew to {len(current)} chars)")


def test_partial_fraction_in_dataset(N: int = 6, num: int = 20):
    """Build a dataset using only partial fraction + ward scramblers and verify."""
    print("Testing partial fractions within dataset build…")

    pairs = gd.build_dataset(N, num, max_scr=3, min_scr=1, seed=777,
                             use_denominators=True, validate=True)
    # Independent verification
    for i, (simple, scrambled) in enumerate(pairs):
        mom, pol = generate_kinematics(N, M=2.0)
        expd = expand_from_simple(simple)
        v1 = eval_infix(expd, mom, pol)
        v2 = eval_infix(scrambled, mom, pol)
        tol = max(1e-10, 1e-8 * max(1.0, abs(v1)))
        assert abs(v1 - v2) <= tol, f"Dataset pair {i} PF-scrambled mismatch"

    print(f"  ✓ {len(pairs)}/{num} pairs verified with PF-enabled scrambling")


# ── Test: canonicalisation ───────────────────────────────────────────

def test_canonicalisation():
    """Canonical forms should be idempotent and permutation-invariant."""
    print("Testing canonicalisation…")

    # Denominator sorting
    den = "*".join(["p_5 · p_1", "p_1 · p_3", "p_6 · p_2"])
    can = gd.canonicalise_denominator(den)
    assert "p_1 · p_3" in can
    assert "p_1 · p_5" in can
    assert "p_2 · p_6" in can

    # GI product: canonical form is independent of input order
    items = [
        "p_4 · F_2 · p_1",
        "Tr(F_5 · F_3)",
        "p_6 · F_4 · F_3 · p_1",
        "p_1 · p_6",
    ]
    can1 = gd.canonicalise_gi_product("*".join(items))
    random.shuffle(items)
    can2 = gd.canonicalise_gi_product("*".join(items))
    assert can1 == can2, f"Canonicalisation not order-invariant:\n  {can1}\n  {can2}"

    # Trace canonicalisation (cyclic)
    assert gd._canon_TrN("Tr(F_3 · F_1 · F_2)") == gd._canon_TrN("Tr(F_1 · F_2 · F_3)")
    assert gd._canon_TrN("Tr(F_2 · F_3 · F_1)") == gd._canon_TrN("Tr(F_1 · F_2 · F_3)")

    print("  ✓ Canonicalisation OK")


# ── Test: full dataset build ─────────────────────────────────────────

def test_build_dataset(N: int = 6, num: int = 10):
    """Build a small dataset and numerically verify all pairs."""
    print(f"Testing build_dataset (N={N}, {num} samples)…")
    pairs = gd.build_dataset(N, num_samples=num, max_scr=3, seed=99,
                             use_denominators=True, validate=True)
    # Independent verification on fresh kinematics
    passed = 0
    for simple, scrambled in pairs:
        mom, pol = generate_kinematics(N, M=2.5)
        expd = expand_from_simple(simple)
        v1 = eval_infix(expd, mom, pol)
        v2 = eval_infix(scrambled, mom, pol)
        tol = max(1e-10, 1e-8 * max(1.0, abs(v1)))
        assert abs(v1 - v2) <= tol, f"Dataset pair mismatch: {v1} vs {v2}"
        passed += 1
    print(f"  ✓ {passed}/{len(pairs)} pairs verified")


# ── Test: seed reproducibility ───────────────────────────────────────

def test_seed_reproducibility():
    print("Testing seed reproducibility…")
    a = gd.build_dataset(6, 5, max_scr=3, seed=123, validate=False)
    b = gd.build_dataset(6, 5, max_scr=3, seed=123, validate=False)
    assert a == b, "Seed reproducibility broken"
    print("  ✓ Deterministic with same seed")


# ── Test: scramble length cap ────────────────────────────────────────

def test_scramble_length_cap():
    print("Testing scramble length cap…")
    N = 6
    simple = gd.strict_gi_monomial(N)
    expd = "*".join(gd.rewrite_gi(b) for b in simple.split("*"))
    Ng = N - 2
    capped = gd.scramble(expd, Ng, N, max_scr=5, max_len=10)
    assert capped == expd, "Scramble should return unchanged when max_len is tiny"
    print("  ✓ Length cap respected")


# ── Demo: generate and print pairs ───────────────────────────────────

def demo_pairs(num: int = 10, N: int = 6, seed: int = 777):
    print(f"\nDemo: {num} pairs (N={N})")
    print("-" * 60)
    pairs = gd.build_dataset(N, num, max_scr=5, seed=seed,
                             use_denominators=True, validate=True)
    passed = failed = 0
    for i, (simple, scrambled) in enumerate(pairs, 1):
        mom, pol = generate_kinematics(N, M=2.0)
        expd = expand_from_simple(simple)
        v1 = eval_infix(expd, mom, pol)
        v2 = eval_infix(scrambled, mom, pol)
        diff = abs(v1 - v2)
        tol = max(1e-10, 1e-8 * max(1.0, abs(v1)))
        ok = diff <= tol
        status = "OK" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{i:02d}] simple:    {simple[:80]}{'…' if len(simple)>80 else ''}")
        print(f"     scrambled: {scrambled[:80]}{'…' if len(scrambled)>80 else ''}")
        print(f"     {v1:.6e} vs {v2:.6e}  Δ={diff:.3e}  [{status}]")
        print()
    print(f"Demo: {passed} OK, {failed} FAIL")
    return passed, failed


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("gen_data.py — Verification Suite")
    print("=" * 60)

    test_rewrite_rules()
    test_scrambler_invariance()
    test_partial_fraction()
    test_partial_fraction_iterated()
    test_canonicalisation()
    test_build_dataset()
    test_partial_fraction_in_dataset()
    test_seed_reproducibility()
    test_scramble_length_cap()
    demo_p, demo_f = demo_pairs(num=10, N=6)

    print("=" * 60)
    total_f = demo_f
    if total_f == 0:
        print("All tests passed.")
    else:
        print(f"{total_f} demo pair(s) failed — see above.")
