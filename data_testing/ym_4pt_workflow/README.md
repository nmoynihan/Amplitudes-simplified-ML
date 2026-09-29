# Four-gluon audit and model workflow

This package reproduces the results behind the [scientific audit of
`gluon4feyn1234_model_ready.csv`](audit/reference/report.md): seed completion,
symbolic basis search, scalar input preparation, one actual checkpoint call,
and independent physics checks. Everything except the trained checkpoint is
in this repository. No original investigation directory is needed at runtime.

## Run the entire workflow

Run from the repository root with its Python environment. Install the usual
dependencies with `python -m pip install -r environment/requirements.txt`.
Use **SymPy 1.14.0** to reproduce the historical expression ordering exactly:
`python -m pip install 'sympy==1.14.0'`. Verification here used Python 3.12.13,
NumPy 2.5.1, and PyTorch 2.13.0.

```sh
python -m data_testing.ym_4pt_workflow run --derive \
    --checkpoint /path/to/4POINT_YM_Model_500k_Colour/best_model.pt \
    --output-dir /tmp/ym4-full-run
```

Use a new or empty output directory for every command. `--derive` repeats the
actual sparse basis search and feeds its result into preparation. Omitting it
uses the historical basis recorded in `prepare.py`, with fresh exact proofs.
Neither path reads a saved model prediction. Input files default to the tracked
`data/data_ym/gluon4feyn1234.csv.gz` and `gluon4feyn.csv.gz`.

The original checkpoint is epoch 44, SHA-256
`393a8a6c425bb96f0bcb8a2a1efca24641103ab92203083e8c5c62e5d66a2478`.
The runner loads weights with `weights_only=True`, overrides the saved device
with CPU, and uses four CPU threads and unmasked greedy decoding. The default
sequence limit is 160 tokens including BOS; `--threads` and `--max-length`
override those settings. The checkpoint stays external (about 322 MiB).
Different checkpoints must pass the same independent verification, and their
hashes are recorded without claiming to reproduce the original model result.

Expected successful result:

- Prepared input: 48 scalar terms, 1,247 content tokens, SHA-256
  `f73c1e6b8459876394697b26d4f2585b6f51ed0eee52fa8412dd5ba9fbc6ca0c`.
- Actual greedy prediction: 87 content tokens, with a genuinely emitted EOS.
- Exact prediction, exported expression, and tokenizer roundtrip residuals: zero.
- Numerical prediction checks: 40 passes (seeds 700–719, each in Coulomb and
  covariant polarization modes), atol 1e-10 and rtol 1e-8.
- All three independent scientific audits pass, including all 16 helicities.

## Run stages independently

Rebuild and audit the input without PyTorch or a checkpoint:

```sh
python -m data_testing.ym_4pt_workflow verify --derive \
    --output-dir /tmp/ym4-verification
```

This records zero model calls and writes no predicted amplitude. The non-model
stages need only SymPy and NumPy. For the audit of an existing scalar CSV alone,
only SymPy is required:

```sh
python -m data_testing.ym_4pt_workflow audit \
    --input data_testing/ym_4pt_workflow/inputs/gluon4feyn1234_model_ready.csv \
    --output-dir /tmp/ym4-audit
```

The checked-in input fixture is the exact audited CSV. Substitute another path
to audit a copy; the independent audits verify its physics rather than assume
correctness from the filename. They require one headerless `id,expression` row
containing scalar p/e contractions. See [audit details](audit/README.md).

Run the search and preparation separately:

```sh
python -m data_testing.ym_4pt_workflow derive --output-dir /tmp/ym4-basis
python -m data_testing.ym_4pt_workflow prepare \
    --derivation /tmp/ym4-basis/sparse_extended_result.json \
    --output-dir /tmp/ym4-prepared
```

`--seed` and `--reference` override the input locations for preparation/search.
Preparation still enforces the historical model-input hash: these options are
for copies or equivalent representations of this example, not arbitrary new
amplitudes. Standalone audit accepts equivalent scalar representations without
requiring that historical hash.

## What is reconstructed

1. **Complete the seed.** With simultaneous relabeling of momenta and
   polarizations, construct `A = (S1234 + S2341)/2`. This adds a missing cyclic
   contribution and changes the seed's function. Prove exact equality to the
   repository full amplitude and all four Ward identities. Export the grouped,
   expanded, and on-shell reduced variants (593, 519, and 424 content tokens).
2. **Derive the compact basis.** Enumerate the historical 666 gauge-invariant
   tensor candidates. Match the 43 polarization monomials; screen sparse
   supports using a generic kinematic point, then solve and prove accepted
   coefficients as exact rational functions of s and t. The first solution
   has three products of two field-strength chains. Numerical screening is
   a search aid; the full exact identity is the acceptance criterion. The
   procedure does not prove global minimality of every possible basis.
3. **Prepare the scalar model input.** Expand that derived representation with
   the training generator, retaining the historical denominator labels. An
   independent tensor expansion checks every component. Normalize and
   parenthesize with the repository routines, then prove the source and its
   token roundtrip equal the completed amplitude. Enforce the original hash.
4. **Run and verify the model.** Make one CPU greedy call, retain raw emitted
   IDs, expand the actual predicted tensors independently, and prove exact
   equality to the complete amplitude. Verify the exported token roundtrip
   and the original 40 fresh numerical points.
5. **Independently audit the scalar amplitude.** Three separate implementations
   check the normalized cubic/quartic vertices, Ward identities, cyclic and
   reflection symmetry, photon decoupling, BCJ, degree, physical poles and
   residues, 48 linear and 48 circular polarization evaluations, and all 16
   symbolic Parke–Taylor helicity assignments. The orchestrator runs this
   audit before spending a checkpoint call.

Exact proofs impose masslessness, momentum conservation, and transversality.
They use no dimension-specific Gram identities; the helicity checks are in
four dimensions. The amplitude has coupling, color factors, and overall i
stripped, as documented in the original report. Equalities hold away from
denominator poles. The scalar CSV has removable singularities term by term;
the independent audit checks their cancellation in the complete expression.

The compact three-term representation was derived symbolically **before** the
successful model call, then re-expanded into a training-like scalar input.
The checkpoint compresses that prepared input; this is not evidence that it
discovered the decomposition from the original Feynman expression. Historical
direct trials on the ordinary completed expression failed. This reproduction
covers the successful preparation and every result in the scientific audit;
it does not retrain the checkpoint or rerun the failed exploratory model trials.

## Outputs and evidence

| Artifact | Purpose |
| --- | --- |
| `workflow.json` | Mode, running/passed/failed state, model call count and failures |
| `derivation/` | Fresh search solution and manifest when `--derive` is used |
| `completion_manifest.json`, `gluon4feyn1234_completed*.csv` | Seed completion, variants and tokenized variants |
| `basis/` | Compact targets, expanded components and exact basis proof |
| `preparation_manifest.json` | Source/reference hashes, basis provenance and exact checks |
| `gluon4feyn1234_model_ready.csv`, `*_model_ready_tok.csv` | Exact scalar input and its content token IDs |
| `audit/*_results.json` | Fresh symbolic, vertex and helicity evidence |
| `inference_attempt.json` | Actual checkpoint identity and emitted IDs, including invalid/truncated output |
| `gluon4feyn1234_completed_simplified.csv` | Actual model prediction, exported only after verification |
| `results.json` | Successful run results; absent if any stage fails |

A failed run leaves diagnostic/partial stage files and marks `workflow.json`
as failed. It never fills in an answer from a reference prediction or labels a
truncated decode as success. Existing nonempty output directories are refused.

The original [report and audit evidence](audit/reference/report.md) and
[one-call result](reference/one_shot_results.json) are retained for comparison.
Their historical provenance is explicit. Fresh outputs are always separate.
The [29 September reproduction record](reference/reproduction_2026-09-29.json)
records the full `run --derive` validation and comparison with the historical
CSV, prediction token IDs, and every scientific audit field.
The workflow does not require or rebuild the pedagogical PDF to run the audit.

## Code and regression checks

- `algebra.py`: strict arithmetic parser, independent field-strength expansion,
  on-shell quotient and CSV handling.
- `derive.py`: candidate generation, support search and exact coefficient solve.
- `prepare.py`: completion, basis export and original scalar/token reconstruction.
- `inference.py`: checkpoint loading and actual greedy decoding.
- `verification.py`: independent prediction identity and numerical cross-checks.
- `audit/`: the three independent audits from the original investigation.
- `workflow.py`, `__main__.py`: orchestration and command-line entry point.

These port the original `build_completed.py`, `basis/derive_extended.py`,
`basis/export_three.py`, `simplify_completed.py`, `audit_symbolic.py`,
`check_vertices.py`, and `check_parke_taylor.py`. Machine-specific paths and
import-time calculations have been removed; the calculations are retained.

```sh
python -m unittest data_testing.ym_4pt_workflow.test_workflow \
    data_testing.ym_4pt_workflow.audit.test_audit
```

Tests need no checkpoint. They check byte-for-byte preparation, the seed versus
completion distinction, independent tensor algebra, wrong-normalization and
truncated-output rejection, portable imports, and failure evidence. Scientific
checks use explicit exceptions and remain enabled under `python -O`.
