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

The [fixed-order gravity specification](ORDERED_GRAVITY.md) defines scalar
flavour orderings for these benchmarks and explicit auxiliary seeds whose
permutation sums reconstruct them. It includes exact algebraic proofs, an
independent CHY check, normalization conventions, and the channel-preserving
permutations allowed for the four-scalar benchmark.

## Ordered-component train/test generation

The entry point `data_gen/ordered_gravity_gen.py` builds a synthetic training
set and a separate test set for the auxiliary `m3` and `m4` components defined
in [ORDERED_GRAVITY.md](ORDERED_GRAVITY.md). Run from the repository root:

```bash
# 100,000 training rows and 200 held-out test rows.
python -m data_gen.ordered_gravity_gen 5 --samples 100200 --test-size 200 --jobs 8 --output-dir data/gravity/ordered

# Small end-to-end generation check.
python -m data_gen.ordered_gravity_gen --samples 28 --test-size 8 --jobs 1 --max-scr 2 --output-dir /tmp/ordered-gravity-check
```

Direct execution of `data_gen/ordered_gravity_gen.py` is also supported.
As in the cyclic Yang–Mills CLI, **`--samples` includes the test rows**.
The default mixed test set has four distinct compact targets: the reference
and partner component of each of the two benchmarks. Its 200 rows are 50
scrambles per component, evenly distributed over depths 1–5; they are not 200
independent physical amplitudes. General test sizes are balanced as evenly as
possible across components and depths. `--process 3s2h` or `--process 4s1h`
selects one process and its two components.

Training targets use the existing synthetic, gauge-invariant gravity grammar
in reference role order `(1,2,3,4,5)`, with two occurrences of each graviton's
field strength per term. They are **synthetic expressions, not new physical
ordered amplitudes**. The single-field-strength contractions do not receive
Yang–Mills' cyclic-word constraint, and no cyclic adjacency restriction is
imposed on gravity poles. Metadata distinguishes these targets from the
held-out benchmark seeds. The physical double scalar-order coefficients in
the scientific note are a different object; this CLI targets auxiliary seeds.

The output directory contains:

| Split | Raw expressions | Prefix tokens | Metadata |
| --- | --- | --- | --- |
| Train | `ordered_gravity_train_raw.csv.gz` | `ordered_gravity_train_tok.csv.gz` | `ordered_gravity_train_metadata.csv.gz` |
| Test | `ordered_gravity_test_raw.csv.gz` | `ordered_gravity_test_tok.csv.gz` | `ordered_gravity_test_metadata.csv.gz` |

Raw and token files retain the standard `simple,scrambled` columns. Metadata
records the compact origin, object kind, role order, definition version and,
for benchmarks, the reconstruction component, partner order, coefficient and
parent normalization. Add the two component predictions with their recorded
coefficient to compare with the original full benchmark. Individual test rows
are evaluated against their own component target.

The generator reserves both full parent benchmarks and both component seeds,
including every species-preserving relabeling and nonzero overall rescaling.
It checks the compact origin even for staged training rows. Structural
signatures and four-point numerical fingerprints screen the reserved families;
both columns of the serialized train/test token files must be disjoint.
These checks do not prove inequivalence under every possible algebraic identity.
Numerical nonzero and equivalence checks, including decoded-token checks, are
mandatory. Rational coefficients such as the half in `m4` are represented
without changing the tokenizer vocabulary.

All six files are staged and audited before publication. The
`ordered_gravity_manifest.json` records counts, definitions, seeds, settings,
reconstruction checks and SHA256 hashes. Existing outputs require `--overwrite`;
publication restores the previous outputs if a file replacement fails.

Use `--kind oneshot`, `--kind staged`, or the default `--kind mixed`; the
training count is balanced over the selected process/kind combinations.
`--seed` controls training, and `--split-seed` controls test scrambling.
`--min-scr`, `--max-scr`, `--min-terms`, `--max-terms`, `--max-tokens`,
`--scrambles` and `--max-candidates-factor` configure the generation bounds.
`--split-output-dir` aliases `--output-dir`, and `--dataset-kind` aliases
`--kind`. Only five-point generation is defined.

Pass only `ordered_gravity_train_tok.csv.gz` to the trainer's `--data-files`
option. Its internal training/validation split must stay within that file;
keep `ordered_gravity_test_tok.csv.gz` reserved for final evaluation.

```bash
python -m unittest data_gen.test_ordered_gravity_gen -v
```

## Commands

The Python launcher `data_gen/gravity_generation.py` provides the `data`,
`train`, `eval`, `full`, and `smoke` modes of `run_gravity_100k.sh`, with an
explicit audit of the held-out benchmark test set. Its default mode is `data`.
For a fresh run alongside the existing datasets:

```bash
python3 -m data_gen.gravity_generation data --output-dir data/gravity/audited
python3 -m data_gen.gravity_generation train --output-dir data/gravity/audited
python3 -m data_gen.gravity_generation eval --output-dir data/gravity/audited
```

With no `--output-dir`, it uses the original six filenames under `data/gravity`.
Existing outputs require `--overwrite`. All six files are staged and checked
before publication; a `generation_manifest.json` records settings, split counts,
rejections, and file hashes. Publication restores previous outputs if a file
replacement fails. Evaluation output paths must be distinct from all datasets
and the manifest, including symbolic-link and hard-link aliases; `full` and
`eval` check this before starting work. Train/eval modes require the manifest and recheck
the hashes and actual tokenized train/test overlap before proceeding. Legacy
files without the manifest remain usable with the original shell launcher.

The default main corpus contains 100,000 pairs, exactly 25,000 for each
process/mode combination. The separate test set contains 200 pairs: 100
scrambles of each of the two paper amplitudes, evenly spread across depths
1–5. These test rows are **never passed to the trainer**. The trainer's default
90/10 training/validation split uses only the main corpus.

The launcher generates benchmarks first, excludes their compact amplitude
signatures and species-preserving particle relabelings from training, and
rejects any training input or target whose token sequence occurs on either
side of a benchmark pair. Rejected rows are replaced to preserve exact quotas.
New metadata includes `compact_origin`, so this family check also covers staged
rows with intermediate targets. These are structural and token-identity checks;
they do not prove inequivalence under every possible algebraic identity.

Useful options include `--samples`, `--benchmark-samples` (per amplitude,
divisible by five), `--seed`, `--benchmark-seed`, `--jobs`, and `--max-tokens`.
Training settings and each output path also have CLI options; the shell
launcher's environment settings such as `SAMPLES`, `JOBS`, and `RAW_OUT` are
supported. Subprocesses use the same Python interpreter as the launcher.
Relative custom paths resolve from the caller's working directory; default
paths resolve from the repository. Direct file execution is supported too.

```bash
# Small data-only check; does not train a model.
python3 -m data_gen.gravity_generation --samples 16 --benchmark-samples 5 \
  --jobs 2 --output-dir /tmp/gravity-check

# Small generation + one CPU training epoch, under data/gravity/smoke/.
python3 -m data_gen.gravity_generation smoke
```

The lower-level commands remain available:

Run all physics and pipeline tests:

```bash
python3 -m unittest data_gen.data_gen_gravity.test_gravity -v
python3 -m unittest data_gen.test_gravity_generation -v
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

The gravity run defaults to micro-batches of 2 with 12-step gradient
accumulation, giving an effective training batch of 24 without materializing
24 long attention matrices at once. Override these with `BATCH_SIZE` and
`GRAD_ACCUM_STEPS`.

Evaluation reports exact match, numerical equivalence, token reduction, and
breakdowns by process and scramble depth:

```bash
python3 -m data_gen.data_gen_gravity.evaluate \
  --model-path models/gravity_5pt_mixed_100k/best_model.pt
```
