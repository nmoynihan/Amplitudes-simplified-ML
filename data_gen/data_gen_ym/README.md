# Cyclic Yang–Mills datasets

Run from the `Amplitudes-simplified-ML` repository root, using a Python environment
with the repository dependencies installed. This requests exactly 450,000
accepted pairs and holds out 200, leaving 449,800 training rows:

```bash
python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 4 \
  --samples 450000 --test-size 200 \
  --seed 42 --split-seed 20260401 \
  --min-terms 1 --max-terms 3 --min-scr 1 --max-scr 4 \
  --max-tokens 2048 \
  --candidate-batch-size 4000 --generator-batch-size 500 --jobs auto \
  --raw-out data/data_ym/cyclic/ym_cyclic_4pt_pool.csv \
  --tok-out data/data_ym/cyclic/ym_cyclic_4pt_pool_tok.csv \
  --report-out data/data_ym/cyclic/ym_cyclic_4pt_pool.report.json \
  --split-output-dir data/data_ym/cyclic/train_test_split
```

Use `--samples 500000` for 499,800 training rows plus 200 test rows. `--samples`
always includes the test rows. Parent directories are created automatically.

The CLI applies the existing `canonical_nonzero` cleaning pipeline with a
cyclic-aware orientation rule: the F labels in each trace or open chain retain
a forward subsequence of `(1, …, N)`, including wraparound. Momentum endpoints
do not enter this ordering rule. Canonicalization cancels common factors,
combines matching terms, and removes exact and numerically detected zero terms
and subsets. The final target must pass a nonzero check and match the scrambled
source on independent deterministic kinematic grids. These numerical checks
are not symbolic proofs. Raw expressions are fully parenthesized for faithful
tokenization.

Deduplication happens after cleaning. Generation continues in batches until
exactly `--samples` distinct cleaned pairs are accepted. The default candidate
budget is five times the requested size; increase `--max-candidates-factor` if
the run exhausts that budget. A failed generation does not publish an undersized
pool or replace existing outputs. Use `--overwrite` to replace a completed pool
and its report deliberately. The JSON report records accepted/rejected counts,
settings, cyclic order, and output hashes.

The splitter writes four CSV files plus a manifest:

- `ym_cyclic_4pt_train_449800.csv` and `ym_cyclic_4pt_train_449800_tok.csv`
- `ym_cyclic_4pt_test_200.csv` and `ym_cyclic_4pt_test_200_tok.csv`
- `ym_cyclic_4pt_split_seed20260401.json`

Test targets and scrambled inputs must each occur once in the pool, and their
token sequences are checked to be disjoint from training. Repeated targets stay
in training. If there are fewer than 200 eligible rows, splitting fails and the
completed pool remains available; no partial split is published. This checks
token equality, not general algebraic equivalence. Use `--split-overwrite` to
replace an existing split. Omit `--test-size` to generate only the clean pool.

Previous command spellings remain supported: `--batch-size` means
`--generator-batch-size`, `--mass` means `--energy-scale`, and `--log-out` now
means `--report-out` and writes JSON. `--no-validate` is rejected because the
clean output requires validation. Both `--dataset-kind` values use the same
pair builder. Programmatic `build_dataset` and `build_dataset_batched` helpers
in the cyclic module remain low-level candidate builders; use the CLI or the
clean engine's `generate_to_files(..., cyclic_order=True)` for clean exact-count
outputs.

Five-point generation uses the same entry point with positional argument `5`.
The existing limitation for six or more particles remains: the denominator
model does not include genuine multiparticle poles.
