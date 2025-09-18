from Tokenizer import ScatteringAmplitudeTokenizer
tok = ScatteringAmplitudeTokenizer(max_particles=8)
amp_infix = "-2*(e_3·e_4*p_1·p_3*p_1·p_4 + e_3·p_1*e_4·p_2*p_1·p_4 + e_3·p_2*e_4·p_1*p_1·p_3)/(p_1·p_3*p_1·p_4)"
amp_tok = tok.encode_infix(amp_infix)
print(f"Infix: {amp_infix}")
print(f"Tokens: {amp_tok}")