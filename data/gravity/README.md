# Generated gravity datasets

The three `gravity_5pt_100k_*` files are aligned views of one 100,000-pair
training corpus:

- `raw`: infix `simple,scrambled` expressions;
- `tok`: the same two columns encoded with the unchanged 58-token vocabulary;
- `metadata`: process, training mode, seed, scramble trajectory, stage, token
  lengths, and numerical validation error.

The realized process/mode counts are 25,004 `3s2h` one-shot, 25,004 `3s2h`
staged, 25,004 `4s1h` one-shot, and 24,988 `4s1h` staged. This 16-row
cross-worker deduplication skew was accepted for the materialized dataset; the
generator now enforces exact post-deduplication quotas on future runs.

The three `benchmarks_*` files contain 200 held-out examples derived from
arXiv:2408.04720: 100 unique scrambles for each amplitude, with 20 examples at
each depth from 1 through 5. They are never mixed into training.

The `smoke_*` files are the eight examples used for the successful local
one-epoch CPU training smoke test.
