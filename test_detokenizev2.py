from data_generation.Tokenizer import ScatteringAmplitudeTokenizer, numerically_equivalent
tok = ScatteringAmplitudeTokenizer(max_particles=8)
'''
Input:     [2, 7, 6, 24, 13, 4, 4, 22, 6, 22, 6, 22, 36, 37, 26, 28, 26, 29, 22, 6, 22, 6, 22, 36, 26, 37, 27, 26, 29, 22, 6, 22, 6, 22, 36, 27, 37, 26, 26, 28, 22, 6, 22, 26, 28, 26, 29, 3]
Best prediction: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 22, 6, 22, 6, 22, 26, 27, 26, 28, 26, 29, 3]
All 10 beam hypotheses:
  Beam 1: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 22, 6, 22, 26, 27, 26, 28, 3]
  Beam 2: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 22, 6, 22, 6, 22, 26, 28, 26, 29, 3]
  Beam 3: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 26, 29, 22, 6, 22, 26, 27, 26, 28, 3]
  Beam 4: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 22, 6, 22, 6, 22, 26, 27, 26, 28, 26, 29, 3]
  Beam 5: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 22, 6, 22, 6, 22, 26, 27, 26, 28, 26, 28, 3]
  Beam 6: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 22, 6, 22, 6, 22, 26, 28, 26, 28, 26, 29, 3]
  Beam 7: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 26, 29, 22, 6, 22, 6, 22, 26, 28, 26, 29, 3]
  Beam 8: [2, 7, 22, 6, 22, 6, 23, 22, 43, 44, 26, 27, 26, 29, 22, 6, 22, 26, 27, 26, 28, 3]
  Beam 9: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 26, 29, 22, 6, 22, 6, 22, 26, 28, 26, 28, 3]
  Beam 10: [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 26, 29, 22, 6, 22, 6, 22, 26, 27, 26, 28, 3]
'''

scrambled = [7, 6, 24, 13, 4, 4, 22, 6, 22, 6, 22, 36, 37, 26, 28, 26, 29, 22, 6, 22, 6, 22, 36, 26, 37, 27, 26, 29, 22, 6, 22, 6, 22, 36, 27, 37, 26, 26, 28, 22, 6, 22, 26, 28, 26, 29]

unscrambled = [2, 7, 22, 6, 23, 22, 43, 44, 26, 27, 22, 6, 22, 26, 27, 26, 28, 3]

inamp = tok.decode_infix(scrambled)
print(f"Scrambled: {inamp}")
polamp = tok.decode_prefix(unscrambled)
print(f"Polish: {polamp}")
amp = tok.decode_infix(unscrambled)
print(f"Result: {amp}")

ok, details = numerically_equivalent(
    tokenizer=tok,
    a_tokens=unscrambled,   # or unscrambled_expr
    b_tokens=scrambled,     # or scrambled_expr
    N=5,                        # total external legs
    samples=5,                  # number of random phase‑space points
    M=2.0,                      # scalar mass
    tol_abs=1e-12,
    tol_rel=1e-10,
    seed=123,
    return_details=True
)

if not ok:
    print("Mismatch detected:")
    for s in details['samples']:
        print(f" sample {s['index']}: a={s['value_a']:.6e}  b={s['value_b']:.6e}  diff={s['abs_diff']:.3e} rel={s['rel_diff']:.3e}")
else:
    print("Expressions are numerically equivalent.")