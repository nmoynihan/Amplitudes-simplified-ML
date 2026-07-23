# Five-point gravity data

This package adapts the scalar-QED `simple → expand → scramble → validate →
tokenise` flow to:

- `3s2h`: scalar legs 1–3 and positive-helicity gravitons 4–5;
- `4s1h`: scalar legs 1–4 and positive-helicity graviton 5;
- `mixed`: exactly balanced sampling of the two processes.

Every graviton occurs twice in each compact monomial, always in two separate
`p_a · F_i · p_b` contractions. Compact terms have stripped dimension 0 for
`3s2h` and -2 for `4s1h`.

The paper fixtures from arXiv:2408.04720 are defined in `core.py`. The
one-graviton fixture is the integer-normalized `2 M`. At startup they are
checked directly against Eqs. (4.7) and (4.8) with complex spinor-helicity
kinematics. They expand to 32 and 12 dot-product terms.

## Commands

Run all physics and pipeline tests:

```bash
python3 -m unittest data_gen.data_gen_gravity.test_gravity -v
```

Generate the balanced 100k training set:

```bash
python3 -m data_gen.data_gen_gravity.generate \
  --samples 100000 --process mixed --kind mixed --jobs 8 \
  --raw-out data/gravity/gravity_5pt_100k_raw.csv.gz \
  --tok-out data/gravity/gravity_5pt_100k_tok.csv.gz \
  --metadata-out data/gravity/gravity_5pt_100k_metadata.csv.gz
```

Generate 100 held-out scrambles per paper amplitude (20 at each depth 1–5):

```bash
python3 -m data_gen.data_gen_gravity.generate \
  --benchmarks --benchmark-samples 100 \
  --raw-out data/gravity/benchmarks_raw.csv.gz \
  --tok-out data/gravity/benchmarks_tok.csv.gz \
  --metadata-out data/gravity/benchmarks_metadata.csv.gz
```

The top-level `run_gravity_100k.sh` script chains data generation, training
with dynamic padding/length bucketing and a 4096-token cap, and complex
gravity evaluation. `./run_gravity_100k.sh smoke` performs a tiny CPU run;
the full model is intended for CUDA.

As in the existing generators, the 4096 limit counts expression tokens.
Training reserves two additional sequence positions for BOS/EOS so a valid
4096-token row is never silently truncated.

Evaluation reports exact match, numerical equivalence, token reduction, and
breakdowns by process and scramble depth:

```bash
python3 -m data_gen.data_gen_gravity.evaluate \
  --model-path models/gravity_5pt_mixed_100k/best_model.pt
```
