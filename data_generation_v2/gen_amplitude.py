#!/usr/bin/env python3
"""Quick demo: tokenise a single amplitude expression."""
from Tokenizer import ScatteringAmplitudeTokenizer

tok = ScatteringAmplitudeTokenizer(max_particles=8)
amp = "-2*(e_3·e_4*p_1·p_3*p_1·p_4 + e_3·p_1*e_4·p_2*p_1·p_4 + e_3·p_2*e_4·p_1*p_1·p_3)/(p_1·p_3*p_1·p_4)"
ids = tok.encode_infix(amp)
print(f"Infix:  {amp}")
print(f"Tokens: {ids}")
print(f"Prefix: {tok.decode_prefix(ids)}")
print(f"Back:   {tok.decode_infix(ids)}")
