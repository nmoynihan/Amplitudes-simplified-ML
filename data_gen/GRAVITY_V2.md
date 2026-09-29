# Gravity generator v2: implementation and future run

This assignment improves generator/evaluator code only. **No production dataset,
pilot corpus, bulk tokenization, or model training was run.** Generator tests use
bounded in-memory examples and temporary fixtures removed by their tests. Existing
datasets, diagnostic reports, and checkpoint weights are preserved.

## What the generator supports

`gravity_v2.py` reuses the existing exact v2 implementation. Its sparse scalar
contractions carry `Fraction` coefficients; the order inside a mixed field-strength
chain remains part of its atom. The six `3s2h` numerator templates are `XXXX`,
`TXX`, `TT`, `QXX`, `QQ`, and `TQ`, with denominator multiplicities 6, 4, 2, 5, 4,
and 3. `4s1h` keeps its two-single-F template and mass dimension -2. Independent
checks verify polarization degrees, dimensions, exact expansions, and nonzero
numerical values. Each pole may occur up to four times in a compact origin.

Default exact coefficients cover nonzero numerators from -12 through 12 and
denominators 1 through 12 or 24. They include both signs of 1/6, 1/3, 1/2, 2/3,
2, 3, 4, and 6. `--coefficient-numerator-max` and
`--coefficient-denominators` configure that support. Reduced Fractions are sampled
independently per monomial, with deliberate sampling of the named values.
Legacy/replay categories keep unit coefficients. Configurations with only unit
coefficients cannot satisfy the enriched categories and are rejected.

Source variants cover expansion, coefficient splitting/collection, removable
common factors, common denominators, monomial partial-fraction distribution,
momentum conservation, transversality, dot commutation, and term ordering.
These algebraic identities are checked exactly before numerical validation;
the generated pairs do not rely on a same-helicity reduction.

`--direct-fraction 3/4` is the default. A deterministic schedule gives
`ceil(N * fraction)` direct targets in each category's first N accepted rows.
Intermediate targets expand one field-strength atom per monomial. The recorded
stage is checked against the actual target, not inferred solely from the schedule.
Each accepted row receives one quota category, plus independently inferred,
potentially overlapping feature tags. Rejected attempts do not contribute to
accepted coverage statistics.

## Representation and holdouts

The tokenizer vocabulary and token IDs are unchanged. The safe renderer places
integer coefficient factors apart from other numerical leaves and expands powers
into repeated multiplication where appropriate. Each actual encode/decode result
must retain its exact scalar expression. This is a **restricted representation**:
for example, a standalone `2/3` is rejected because the legacy consecutive digit
format is ambiguous. Powers-to-products alone cannot solve every numeric boundary.
The strict parser rejects unsupported nodes and trailing text rather than mapping
them to zero. Existing trace rendering, signed reversal/cyclic signatures, and
integer-power degree checks are retained and covered by regression tests.

Compact-origin family assignment occurs before coefficient descendants,
relabelings, scrambles, and stages. The signature ignores independent coefficients
and canonically considers species-preserving relabelings. The supplied user
amplitude and both explicitly verified compact references are mandatory holdouts,
alongside the existing benchmark families. Missing required reference files cause
an error. Held-out test origins are added to the guard before any training rows.

Structural checks and independent numerical fingerprints both participate.
The guard conservatively reserves fingerprints that match only in the positive
helicity sector as well as matches across general transverse polarizations.
This closes the gap between the generator and its independent audit. It does not
turn a helicity-specific identity into a general-polarization identity. Finite
fingerprints cannot prove arbitrary inequivalence.

## Future 100,000/200 plan — commands not executed

Run from `/Users/kymani/Documents/New_Scattering_AI/Repo/Amplitudes-simplified-ML`.
Choose a new output directory; existing release CSVs are never overwritten.

```bash
/Users/kymani/miniforge3/envs/scattering/bin/python -m data_gen.gravity_v2 \
  --output-dir /Users/kymani/Documents/New_Scattering_AI/Repo/gravity-data-v2-future \
  --train-rows 100000 --test-rows 200 \
  --seed 20260923 --split-seed 730021 \
  --validation-seed 930103 --fingerprint-seed 510031 --audit-seed 830017 \
  --max-tokens 4096 --direct-fraction 3/4 \
  --coefficient-numerator-max 12 \
  --coefficient-denominators 1 2 3 4 5 6 7 8 9 10 11 12 24
```

`--train-rows` counts training rows only: this requests **100,200 total pairs**.
The planned quotas are:

| Assigned category | Training | Test |
|---|---:|---:|
| `3s2h`: legacy single-F | 10,000 | 20 |
| `3s2h`: coefficients, distinct target poles | 10,000 | 20 |
| `3s2h`: coefficients, repeated target poles | 10,000 | 20 |
| `3s2h`: trace/mixed-F | 10,000 | 20 |
| `3s2h`: mixed-family sums | 10,000 | 20 |
| `4s1h`: replay | 50,000 | 100 |
| **Total** | **100,000** | **200** |

These feature quotas are proposed starting values, not an experimentally established
optimum. The command's argument syntax and quota arithmetic were tested without
running it. Numerical validation is still required during the future run; quota
filling, runtime, and full accepted-row feature coverage have not been measured
for this revised implementation.

Generation preserves `simple,scrambled` raw/token schemas and writes aligned
metadata, rejection counts, origin diversity, accepted coverage, tokenizer/source
hashes, reference hashes, seeds, and settings. Before writing its manifest it
reopens all six CSVs and verifies row identity, content/token correspondence,
exact token round trips, capacity, and exact category/output counts. An interrupted
or rejected run is not an audited release; use a fresh directory for a retry.

After future generation, run the independent exported-file audit before using
those files. This command also **has not been executed in this assignment**:

```bash
/Users/kymani/miniforge3/envs/scattering/bin/python -m data_gen.gravity_v2_audit \
  --release-dir /Users/kymani/Documents/New_Scattering_AI/Repo/gravity-data-v2-future \
  --audit-seed 830017
```

The audit independently reparses each row, checks alignment and exact/numerical
identities, enforces accepted feature coverage and process/stage quotas, and checks
family exclusions at independent momenta. A fresh manifest can be audited without
an unrelated historical baseline directory; preservation checks cover the exact
files listed in its immutable reference manifest (or the additional historical
baseline files when such a baseline exists).

## Verification performed

```bash
/Users/kymani/miniforge3/envs/scattering/bin/python -m unittest \
  data_gen.test_gravity_v2 data_gen.test_gravity_v2_representation \
  data_gen.test_gravity_v2_validation data_gen.test_gravity_v2_publication -q

/Users/kymani/miniforge3/envs/scattering/bin/python -m unittest \
  data_gen.test_gravity_evaluator_contract -q

/Users/kymani/miniforge3/envs/scattering/bin/python -m unittest \
  data_gen.test_gravity_generation data_gen.test_ordered_gravity_gen \
  data_gen.data_gen_gravity.test_gravity -q

/Users/kymani/miniforge3/envs/scattering/bin/python -m data_gen.gravity_v2 --help
```

Results: **40 v2 tests, 5 generator/evaluator contract tests, and 49 legacy gravity
tests passed**. The 11 new generator tests
exercise configurable exact support and stage ratios, family assignment,
positive-helicity-only exclusion, required holdout files, two-row temporary
publication alignment/corruption/count checks, accepted feature counters, and
deterministic compressed fixture bytes. Existing tests independently cover all
six templates, generic trace/mixed-chain identities, transformations, degree and
pole accounting, numeric-boundary rejection, and tokenizer compatibility.
The contract tests pass emitted expressions and actual tokens through evaluator
input-integrity checks, numerical comparison at 12 independent momentum/reference
configurations, and full candidate records. They cover all six compact templates,
the `4s1h` template, independently weighted mixed-family sums, collection/cancellation
styles, strictly shorter correct outputs, and the user's frozen holdout references.
