# Fixed cyclic Yang–Mills generation

`yang_mills_cyclic_generation.py` produces `(simple, scrambled)` pairs using
the Yang–Mills pipeline in `generate.py` and `expr_model.py`. The original
generator still samples arbitrarily ordered numerator blocks by default.
The new entry point enables the explicit `cyclic_order=True` option throughout
generation and multiprocessing workers; it does not replace functions at runtime.

## Reference ordering defaults

`notation.default_cyclic_order` explicitly uses the external cycles named by:

- `data/data_ym/gluon4feyn1234.csv.gz`: **(1, 2, 3, 4)**;
- `data/data_ym/gluon5feyn12345.csv.gz`: **(1, 2, 3, 4, 5)**.

These match the existing natural-order defaults; no change to generated label
sequences is needed. Both numerator sampling and adjacent-channel construction
now use this shared definition. Other multiplicities retain `(1, ..., N)`.
Generation logs and split manifests record `particle_order` and
`ordering_reference`, and the CLI prints the selected cycle.

Each reference CSV contains one headerless `id,expression` row of expanded dot
products, with no compact `F_i` or `Tr` blocks. Its filename identifies the
intended external cycle. The literal denominator channels are `(1,2)` at four
points and `(1,2)` plus `(4,5)` at five points, consistent with those cycles but
insufficient on their own to infer a unique cycle. They do not restrict the
generator to that subset: the full adjacent-channel pool, including `(N,1)`,
is preserved. Scalar contractions such as `e_1 · e_3` do not specify or violate
the ordering of a compact field-strength word. Reference files are used by
regression tests and for provenance; generation does not need to load them.

## Exact numerator invariant

For each trace or open chain with field-strength word `(j0, j1, ..., jk)`, all
labels are distinct members of `1..N`, and the offsets
`((j - j0) % N)` are strictly increasing along the word. Equivalently, the word
is a subsequence of **one rotation** of `(1, 2, ..., N)`. It may omit labels and
cross `N -> 1`, but may not traverse the cycle twice or reverse its orientation.
At five points `(4, 5, 1, 3)` is valid; `(3, 1, 4)` and `(1, 3, 2)` are not.
A single label is valid, and either orientation of a two-label word is valid:
with only two labels, reversal is also a cyclic rotation. Distinct orientation
becomes meaningful for words of at least three labels.

Every external `F_i` occurs exactly once **across the product of blocks in each
numerator monomial**, not once in every block. Each block follows the same cycle
independently. Block label sets may interleave, for example traces on `(1, 3, 5)`
and `(2, 4)`; there is no extra contiguous-partition or noncrossing requirement.
Multiplying scalar traces/chains in a different textual order does not change
particle ordering. Concatenating their field-strength words is not an invariant.

Only the field-strength word is ordered in `p_a · F_j0 · ... · F_jk · p_b`.
Momentum endpoints retain the original choices: `a != j0` and `b != jk` to
avoid immediately vanishing contractions. Endpoints may coincide with each
other or with other labels inside the chain or in other blocks. They are not
extra field-strength occurrences and need not bracket the word cyclically.

The sampler preserves weighted block-family/arity selection and uniformly
samples a subset of remaining labels. Its first sampled label chooses a uniform
starting label within that subset; the rest are ordered by forward modular
distance. Removing that subset preserves exact field-strength coverage.

## Ordering audit and unchanged physics model

The relevant operations in the original pipeline are:

| Operation | Treatment |
| --- | --- |
| Remaining-leg shuffle and ordered `random.sample` in `_generate_gi_monomial_spec` | Original path unchanged; cyclic path samples a subset and orders it along the fixed cycle. |
| `_chain_endpoints` | Original momentum endpoint sampling retained. |
| `_scalar_pp_factor` and scalar-power sampling | Scalar momentum contractions remain unrestricted by the field-strength invariant. |
| Numerator-factor shuffle and `canonicalise_gi_product` | Only commuting factors are sorted/combined; traces are rotated to their smallest word, never reversed; chains are unchanged. |
| `_all_physical_poles`, denominator sampling and canonicalisation | Adjacent-channel pool, support choices, cancellation budgets, repeats and scalar sorting are unchanged. |
| `_term_signature` | Sorts lengths/support metadata, not field-strength words. |
| `rewrite_gi`, `_rw_TrN`, `_rw_pFchainp`, AST expansion | Expand the supplied field-strength word in order. |
| Scramblers and simplification | Apply the existing scalar algebra and on-shell identities after expansion; no particle relabelling or compact-block permutation is introduced. |
| Numerical evaluation | Momentum/polarisation arrays remain indexed by the original external labels. |

Trace rotation can hide a sampled wraparound by printing its smallest label
first. Symmetric momentum dots are printed with sorted labels, so the wraparound
channel `(N, 1)` appears as `p_1 · p_N`. Neither changes the invariant. The
separate `generate_clean_4pt.py`/`generate_clean_5pt.py` canonicalisers also use
reversal identities; this entry point does not use those canonicalisers.

The existing **physical-pole** restriction is distinct: only adjacent
two-particle channels `(1,2), (2,3), ..., (N,1)` receive one physical denominator
power. The original denominator multiplicity rule is preserved verbatim:

```text
max_allowed(D) = (1 if D is adjacent else 0) + chain_endpoint_budget(D)
```

Thus nonadjacent factors and repeated factors remain possible when supported by
the existing endpoint budget. That budget is the original model's treatment of
cancellable factors, including its gauge-motivated assumptions; this change adds
no new cancellation claim. Optional scalar numerator powers and mandatory
per-chain-field denominator support are also unchanged. Arbitrary `p_i·p_j`,
`e_i·p_j`, or `e_i·e_j` appearances in expanded/scrambled expressions are expected
and are not violations of compact-block ordering.

Coefficients, dimension `4-N`, retries, term sampling, expansion, all scramblers,
three numerical checks per polarisation mode at each validation stage, token
limits, oversampling, deduplication and CSV/tokenised CSV (including `.gz`)
formats are shared with the original pipeline. Validation defaults to both
`coulomb` and `covariant` polarisations. Both `--dataset-kind oneshot` and `step`
are accepted; as in the existing Yang–Mills generator, `step` is a compatibility
selector for the same pair builder, not a separate one-step trajectory generator.

For **N >= 6**, genuine multiparticle poles `(p_i + ... + p_j)^2` require sums of
scalar products and remain unimplemented. This is not a complete pole model at
those multiplicities. The current block menus have arities up to five; larger
numerators use products of those blocks. The default tokenizer supports labels
through eight (`--tokenizer-max-particles` changes this). These existing limits
are unaffected by numerator ordering.

## Commands and output

From the repository root, in an environment containing NumPy and SymPy:

```bash
python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 4 --samples 1000 --seed 42 --jobs 1
python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 5 --samples 1000 --seed 42 --jobs 1
```

All existing CLI options are available (`--help`), including batching/workers,
scrambler selection, grouped expansion, validation, token limits, and explicit
output paths. Direct script execution also works. From `data_gen/`, the module
name can instead be `data_gen_ym.yang_mills_cyclic_generation`.

Default filenames are `gi_cyclic_{N}pt_{NSAMPS}k.csv`,
`gi_cyclic_{N}pt_tok_{NSAMPS}k.csv`, and
`gen_data_cyclic_{N}pt_{NSAMPS}k.log`, where `NSAMPS = samples // 1000` as before.
They cannot collide with the original generator's **default** output names.
Explicit output paths retain their usual overwrite behaviour, and rerunning a
cyclic command can overwrite its previous cyclic outputs. Use explicit paths
to distinguish runs, especially runs with fewer than 1000 samples.

The module also exposes `build_dataset` and `build_dataset_batched` wrappers
that always enable cyclic ordering and otherwise accept the shared builder's
options. The original builders accept `cyclic_order=True` as an optional keyword.

## Training and held-out test sets

The cyclic entry point can generate one pool and split it into training and
test files in the same command. `--samples` is the total pool size, including
the held-out rows. For example, these commands request 1,000 pairs per particle
count, with 200 test rows and the remaining 800 rows in training:

```bash
python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 4 --samples 1000 --seed 42 --jobs 1 --test-size 200 --split-seed 20260401 --split-output-dir data/data_ym/cyclic_train_test
python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 5 --samples 1000 --seed 42 --jobs 1 --test-size 200 --split-seed 20260401 --split-output-dir data/data_ym/cyclic_train_test
```

The default `--test-size 0` retains the previous generation-only behaviour and
source filenames. A positive test size requires tokenised output and therefore
cannot be combined with `--no-tokenise`. `--split-seed` controls the reproducible
holdout independently of the generation `--seed`. Existing split outputs are
protected unless `--split-overwrite` is supplied; this does not change the
generator's existing overwrite behaviour for its source CSVs and log.

The splitter follows the repository's SQED release holdout approach: it hashes
the complete tokenised `simple` expression with SHA-256 and selects test rows
only when their target occurs exactly once in the source pool. It additionally
requires the tokenised `scrambled` expression to occur exactly once. Thus neither
a held-out target nor a held-out input can occur in training, even if repeated
source rows or several scrambled variants of one target are present. Repeated
targets and inputs stay in training; all source rows are retained across the two
partitions. If there are fewer eligible rows than requested, splitting fails
without publishing split outputs. Reduce the test size or generate a larger pool.

This guarantee concerns identical tokenised expressions. It does not identify
every algebraically equivalent expression. Splitting does not re-canonicalise
the expressions, reverse their field-strength words, or otherwise modify them.
It always verifies the raw/tokenised correspondence, row counts, and zero
train/test target and input overlap before publishing the four CSVs and manifest
transactionally. The source pool is unchanged by the splitter.

The output names are:

- `ym_cyclic_{N}pt_train_{count}.csv` and `ym_cyclic_{N}pt_train_{count}_tok.csv`;
- `ym_cyclic_{N}pt_test_{count}.csv` and `ym_cyclic_{N}pt_test_{count}_tok.csv`;
- `ym_cyclic_{N}pt_split_seed{seed}.json`, recording the split and validation.

To split an existing cyclic pool without regenerating it, use the standalone
module. These examples reuse the default source filenames from the commands
above and publish to a separate directory:

```bash
python -m data_gen.data_gen_ym.split_cyclic_train_test 4 --raw gi_cyclic_4pt_1k.csv --tokenised gi_cyclic_4pt_tok_1k.csv --test-size 200 --seed 20260401 --output-dir data/data_ym/cyclic_resplit --tokenizer-max-particles 8
python -m data_gen.data_gen_ym.split_cyclic_train_test 5 --raw gi_cyclic_5pt_1k.csv --tokenised gi_cyclic_5pt_tok_1k.csv --test-size 200 --seed 20260401 --output-dir data/data_ym/cyclic_resplit --tokenizer-max-particles 8
```

Set `--tokenizer-max-particles` to the value used to generate the pool, and add
`--overwrite` only when replacing existing standalone split outputs. The older
`split_clean_train_test.py` samples row indices and does not provide this
target/input exclusion guarantee for pools containing repeated expressions.

Run the focused tests from the repository root:

```bash
python -m unittest data_gen.data_gen_ym.test_yang_mills_cyclic_generation data_gen.data_gen_ym.test_yang_mills_cyclic_cli data_gen.data_gen_ym.test_split_cyclic_train_test data_gen.data_gen_ym.test_yang_mills_cyclic_split_cli data_gen.data_gen_ym.test_cyclic_reference_order
```
