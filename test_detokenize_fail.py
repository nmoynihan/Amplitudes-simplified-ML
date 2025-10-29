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

scrambled = [5, 4, 5, 7, 22, 6, 22, 6, 22, 6, 13, 35, 28, 36, 27, 27, 28, 22, 26, 28, 7, 6, 22, 6, 13, 35, 36, 8, 22, 27, 28, 13, 22, 26, 28, 7, 22, 6, 22, 6, 22, 35, 28, 36, 27, 28, 29, 22, 26, 28, 7, 22, 6, 22, 6, 22, 35, 36, 27, 28, 28, 29, 22, 26, 28]
unscrambled = [7, 4, 22, 22, 22, 28, 43, 44, 29, 6, 22, 27, 28, 23, 22, 43, 44, 22, 26, 28]

#scrambled = [5, 7, 22, 6, 22, 6, 22, 6, 13, 35, 28, 36, 27, 26, 26, 8, 22, 28, 29, 13, 7, 22, 6, 22, 6, 22, 6, 13, 35, 36, 26, 26, 27, 28, 8, 22, 28, 29, 13]
#unscrambled = [7, 6, 22, 26, 26, 23, 22, 43, 44, 8, 22, 28, 29, 13]

#scrambled = [5, 4, 5, 7, 22, 6, 22, 6, 22, 6, 13, 35, 28, 36, 27, 27, 28, 22, 26, 28, 7, 6, 22, 6, 13, 35, 36, 8, 22, 27, 28, 13, 22, 26, 28, 7, 22, 6, 22, 6, 22, 35, 28, 36, 27, 28, 29, 22, 26, 28, 7, 22, 6, 22, 6, 22, 35, 36, 27, 28, 28, 29, 22, 26, 28]
#unscrambled = [7, 4, 6, 22, 27, 28, 23, 22, 43, 44, 22, 22, 22, 28, 43, 44, 29, 22, 26, 28]

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
    N=4,                        # total external legs
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