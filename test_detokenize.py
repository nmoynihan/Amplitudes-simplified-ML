from data_generation.Tokenizer import ScatteringAmplitudeTokenizer

tok = ScatteringAmplitudeTokenizer(max_particles=4)

tokamp = [2, 7, 23, 22, 43, 44, 22, 6, 22, 6, 22, 26, 29, 26, 29, 27, 29, 28, 29, 3]

polamp = tok.decode_prefix(tokamp)
# amp = tok.decode_infix(tokamp)

print(f"Polish: {polamp}")
# print(f"Result: {amp}")