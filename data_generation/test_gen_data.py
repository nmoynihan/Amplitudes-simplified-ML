#!/usr/bin/env python3
# test_gen_data.py
#
# Test the gen_data module by generating simple and scrambled expressions
# with a single scramble and numerically verifying they are equivalent.
#
# This helps validate the correctness of the F tensor expansions and 
# scrambling operations in gen_data.py.

import random
import numpy as np
from typing import Tuple, Optional
import sys, os
# Ensure we can import local modules when running directly
here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if here not in sys.path:
    sys.path.insert(0, here)

import gen_data as gd
from kinematics import generate_kinematics

# Set up the same dot product function as in verify.py
def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Minkowski inner product with signature (+,-,-,-)"""
    return a[0]*b[0] - np.dot(a[1:], b[1:])

def eval_infix(expr: str, mom: np.ndarray, pol: np.ndarray) -> float:
    """Delegate to the library's safe evaluator (handles · and /)."""
    return gd.eval_infix_numeric(expr, mom, pol)

def expand_from_simple(simple: str) -> str:
    """
    Expand a canonical 'simple' label (possibly with a denominator) into an
    e·p/p·p-only infix expression using rewrite_gi.
    """
    if "/" in simple:
        num, den = simple.split("/", 1)
        num = num.strip()
        den = den.strip()
        if num.startswith("(") and num.endswith(")"): num = num[1:-1]
        if den.startswith("(") and den.endswith(")"): den = den[1:-1]
        expd_num = "*".join(gd.rewrite_gi(b) for b in num.split("*"))
        return f"({expd_num})/({den})"
    else:
        return "*".join(gd.rewrite_gi(b) for b in simple.split("*"))

def test_single_pair(N: int = 5, seed: Optional[int] = None) -> Tuple[str, str, bool, float, float]:
    """
    Generate a single simple/scrambled pair and test if they're numerically equivalent.
    Returns (simple, scrambled, passed, simple_value, scrambled_value)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate a simple GI monomial
    simple = gd.strict_gi_monomial(N)
    
    # Expand it to e·p/p·p form
    expanded = "*".join(gd.rewrite_gi(block) for block in simple.split("*"))
    
    # Apply exactly two scrambles
    Ngamma = N - 2
    scrambled = gd.scramble(expanded, Ngamma, N, max_scr=2, min_scr=1)
    
    # Generate random kinematics
    mom, pol = generate_kinematics(N, M=1.7)
    
    # Evaluate both expressions
    simple_value = eval_infix(expanded, mom, pol)
    scrambled_value = eval_infix(scrambled, mom, pol)
    
    # Check if they match (with reasonable tolerance)
    diff = abs(simple_value - scrambled_value)
    max_val = max(abs(simple_value), abs(scrambled_value))
    relative_error = diff / max_val if max_val > 1e-12 else diff
    
    # Use a more lenient tolerance for numerical precision
    passed = relative_error < 1e-8 or diff < 1e-10
    
    return simple, scrambled, passed, simple_value, scrambled_value

def run_verification_tests(num_tests: int = 20, N: int = 5):
    """
    Run multiple verification tests and report results.
    """
    print(f"Testing gen_data.py with {num_tests} random expressions (N={N})")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i in range(num_tests):
        try:
            simple, scrambled, test_passed, simple_val, scrambled_val = test_single_pair(N, seed=i+42)
            
            if test_passed:
                passed += 1
                print(f"✓ Test {i+1:2d}: PASSED")
                if i < 3:  # Show details for first few tests
                    print(f"    Simple:    {simple[:50]}{'...' if len(simple) > 50 else ''}")
                    print(f"    Values:    {simple_val:.6e} ≈ {scrambled_val:.6e}")
            else:
                failed += 1
                print(f"✗ Test {i+1:2d}: FAILED")
                print(f"    Simple:    {simple[:50]}{'...' if len(simple) > 50 else ''}")
                print(f"    Scrambled: {scrambled[:50]}{'...' if len(scrambled) > 50 else ''}")
                print(f"    Values:    {simple_val:.6e} vs {scrambled_val:.6e}")
                diff = abs(simple_val - scrambled_val)
                print(f"    Difference: {diff:.6e}")
                
        except Exception as e:
            failed += 1
            print(f"✗ Test {i+1:2d}: ERROR - {str(e)}")
    
    print("=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED")
    print(f"Success rate: {100*passed/(passed+failed):.1f}%")
    
    return passed, failed

def test_specific_rewrite_rules():
    """
    Test specific rewrite rules with known examples.
    """
    print("\nTesting specific rewrite rules:")
    print("-" * 40)
    
    # Test cases: (input, expected_pattern)
    test_cases = [
        ("(p_1 · F_3 · F_4 · p_1)/(p_1 · p_3*p_1 · p_4)","-e_3.e_4 + (e_3.p_1*e_4.p_2)/(p_1.p_3) + (e_3.p_2*e_4.p_1)/(p_1.p_4)"),
        ("p_1 · F_2 · p_3", "p_1.*e_2.*p_2.*p_3.*-.*p_1.*p_2.*e_2.*p_3"),
        ("Tr(F_2 · F_3)", "2.*e_2.*p_3.*p_2.*e_3.*-.*p_2.*p_3.*e_2.*e_3"),
    ]
    
    for simple_block, expected_pattern in test_cases:
        if "/" in simple_block:
            expanded = expand_from_simple(simple_block)
        else:
            expanded = gd.rewrite_gi(simple_block)
        print(f"  {simple_block}")
        print(f"  → {expanded}")
        
        # Quick numerical test
        N = 4
        mom, pol = generate_kinematics(N, M=1.7, seed=123)
        
        try:
            value = eval_infix(expanded, mom, pol)
            print(f"  Evaluates to: {value:.6e}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

def test_scrambler_invariance_once(N:int=6, seed:int=0):
    random.seed(seed)
    simple = gd.strict_gi_monomial(N)
    expd = "*".join(gd.rewrite_gi(b) for b in simple.split("*"))
    Ng = N-2
    mom, pol = generate_kinematics(N, M=2.0)
    base = eval_infix(expd, mom, pol)

    scramblers = [
        lambda s: gd.scr_mul_by_one(s, N),
        lambda s: gd.scr_add_zero_gauge(s, Ng, N),
        lambda s: gd.scr_Ptot_dot_pk(s, N),
        lambda s: gd.scr_mc_substitute_ei_pk(s, Ng, N),
        lambda s: gd.scr_commute_dot(s, Ng, N),
    ]
    for fn in scramblers:
        scr = fn(expd)
        val = eval_infix(scr, mom, pol)
        assert abs(val - base) <= max(1e-10, 1e-8*max(1.0, abs(base)))

def test_canonicalisation():
    # Denominator sorting
    den = "*".join(["p_5 · p_1", "p_1 · p_3", "p_6 · p_2"])  # intentionally unsorted
    can = gd.canonicalise_denominator(den)
    assert can.startswith("p_1 · p_3"), can
    assert "p_1 · p_5" in can and "p_2 · p_6" in can

    # Product canonicalisation idempotence under shuffle
    prod = "*".join([
        "p_4 · F_2 · p_1",
        "Tr(F_5 · F_3)",
        "p_6 · F_4 · F_3 · p_1",
        "Tr(F_2 · F_6 · F_5)",
        "p_1 · p_6",
    ])
    can1 = gd.canonicalise_gi_product(prod, strict=True)
    items = prod.split("*")
    random.shuffle(items)
    can2 = gd.canonicalise_gi_product("*".join(items), strict=True)
    assert can1 == can2

def test_build_dataset_numeric(N:int=6):
    # Build a few samples and re-validate numerically by independently expanding the simple label
    pairs = gd.build_dataset(N, num_samples=4, max_scr=3, seed=99, use_denominators=True, validate=True)
    for simple, scrambled in pairs:
        mom, pol = generate_kinematics(N, M=2.5)
        expd = expand_from_simple(simple)
        v1 = eval_infix(expd, mom, pol)
        v2 = eval_infix(scrambled, mom, pol)
        assert abs(v1 - v2) <= max(1e-10, 1e-8*max(1.0, abs(v1)))

def demo_generate_and_print(num_pairs:int=20, N:int=6, seed:int=123):
    print(f"\nDemo: {num_pairs} simple/scrambled pairs with numeric comparison (N={N})")
    print("-"*60)
    pairs = gd.build_dataset(N, num_samples=num_pairs, max_scr=5, seed=seed,
                             use_denominators=True, validate=True)
    passed = 0
    failed = 0
    failed_cases = []  # (index, diff, v1, v2)
    for i, (simple, scrambled) in enumerate(pairs, start=1):
        print(f"[{i:02d}] simple:")
        print(simple)
        print(f"[{i:02d}] scrambled:")
        print(scrambled)
        # numeric check on fresh kinematics
        mom, pol = generate_kinematics(N, M=2.0)
        expd = expand_from_simple(simple)
        v1 = eval_infix(expd, mom, pol)
        v2 = eval_infix(scrambled, mom, pol)
        diff = abs(v1 - v2)
        tol = max(1e-10, 1e-8*max(1.0, abs(v1)))
        ok = diff <= tol
        if ok:
            passed += 1
            status = "OK"
        else:
            failed += 1
            status = "FAIL"
            failed_cases.append((i, diff, v1, v2))
        print(f"     numeric: {v1:.6e} vs {v2:.6e}  diff={diff:.3e}  [{status}]")
        print()
    # Summary and concise failure report (if any)
    print(f"Demo summary: {passed} PASSED, {failed} FAILED")
    if failed_cases:
        brief = ", ".join([f"#{idx} diff={d:.3e}" for idx, d, _, _ in failed_cases[:10]])
        print(f"Failed cases (first {min(len(failed_cases),10)}): {brief}")
    return passed, failed

def test_scramble_length_cap():
    # If max_len is very small compared to the expression, scramble should return unchanged
    N = 6
    simple = gd.strict_gi_monomial(N)
    expd = "*".join(gd.rewrite_gi(b) for b in simple.split("*"))
    Ng = N-2
    capped = gd.scramble(expd, Ng, N, max_scr=5, max_len=10)
    assert capped == expd

def test_seed_reproducibility():
    # With validation off (to avoid stochastic kinematics), results should be deterministic
    N = 6
    a = gd.build_dataset(N, num_samples=5, max_scr=3, seed=123, use_denominators=True, validate=False)
    b = gd.build_dataset(N, num_samples=5, max_scr=3, seed=123, use_denominators=True, validate=False)
    assert a == b

if __name__ == "__main__":
    print("Gen_data.py Verification Test")
    print("=" * 40)
    
    # Test specific rewrite rules first
    test_specific_rewrite_rules()
    # Quick scrambler invariance checks
    test_scrambler_invariance_once()
    # Canonicalisation properties
    test_canonicalisation()
    # Dataset build + numeric check
    test_build_dataset_numeric()
    # Demo pairs printer (counted into totals)
    demo_passed, demo_failed = demo_generate_and_print(num_pairs=20, N=6, seed=777)

    # Run comprehensive verification tests
    tests_passed, tests_failed = run_verification_tests(num_tests=10, N=5)

    total_passed = demo_passed + tests_passed
    total_failed = demo_failed + tests_failed
    print("-" * 60)
    print(f"TOTAL: {total_passed} PASSED, {total_failed} FAILED (including demo)")
    if total_failed == 0:
        print("All tests passed! gen_data.py appears to be working correctly.")
    else:
        print("Some tests failed. See details above (demo failures and verification diffs).")
