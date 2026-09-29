# Model-assisted simplification of the five-point scalar-gravity expression

This package reproduces the selected two-call workflow used in the technical
note. It starts from the supplied scalar expression, derives two scalar cores,
asks the unchanged checkpoint to rewrite those cores in field-strength
notation, and reconstructs and verifies the complete result.

This is a workflow for the particular five-point expression studied here, with
three scalar legs and polarization vectors on legs 4 and 5. It is not a general
simplifier for arbitrary five-point gravity amplitudes. The model weights and
source data are external inputs and are not included in the repository.

## Run from the repository root

Use the same Python environment as the rest of this repository. The workflow
uses PyTorch, SymPy and NumPy, which are included in
[`environment/requirements.txt`](../../environment/requirements.txt). It imports
the repository's tokenizer, transformer and gravity evaluation code.

```sh
python -m pip install -r environment/requirements.txt

python -m data_testing.gravity_5pt_workflow \
    --source /path/to/gravity5unified12345_seed.csv \
    --checkpoint /path/to/best_model.pt \
    --output-dir /path/to/new-empty-run
```

For the original local files:

```sh
python -m data_testing.gravity_5pt_workflow \
    --source /Users/kymani/Documents/Big_YM_Data_Sets/Cyclic_Gravity/Test_Amplitude/gravity5unified12345_seed.csv \
    --checkpoint /Users/kymani/Documents/Big_YM_Data_Sets/Cyclic_Gravity/Model/best_model.pt \
    --output-dir /tmp/gravity5pt-reproduction-01
```

Supply the original single-row, headerless `id,expression` CSV. The selected
inference uses CPU execution, four threads by default and greedy decoding with
`--max-length 256`. Use `--threads` and `--max-length` to override those settings;
`--help` lists the complete interface.

The output directory must be new or empty. Give every rerun a fresh directory,
such as `/tmp/gravity5pt-reproduction-02`, including after an interrupted or
failed run. This prevents old predictions or verification records from being
mistaken for new evidence. The workflow does not modify the source CSV or
checkpoint and does not retrain the model.

To inspect only the analytic preparation, omit the checkpoint and use:

```sh
python -m data_testing.gravity_5pt_workflow \
    --source /path/to/gravity5unified12345_seed.csv \
    --output-dir /path/to/new-empty-preparation \
    --prepare-only
```

Preparation writes the scalar derivations and the two selected expanded inputs
without loading a checkpoint. It makes no neural calls and does not establish
that the model can predict the prepared cores. A complete run is required for
the recorded model-assisted results.

## What the workflow does

1. Parse the original scalar expression and derive an unrestricted
   decomposition by completing the square in `e4.e5` and separating the
   remaining polarization coefficient matrix into two rank-one blocks.
2. Derive the common-reference grouping for the positive-positive helicity
   sector, retaining the external rational coefficients.
3. Construct the two selected scalar-only inputs with the fixed dot-product
   orientation and term ordering used in the successful experiment. Each
   squared product is expanded without collecting its repeated terms.
4. Run one greedy prediction for each input. Check each prediction against the
   scalar expression that was actually submitted.
5. Reconstruct the amplitude from the actual outputs and external coefficients.
   Perform symbolic cleanup, exact identity and tokenizer checks, and numerical
   comparisons.

The selected serialization is fixed: this package does not run a search over
equivalent inputs. During the original exploration, two default SymPy
serializations failed before the two selected oriented inputs succeeded. Those
historical failed attempts are not extra calls in this reproduction.

## Model inputs, predictions and reconstruction

Write `dij = pi.pj`, `Fi = pi ∧ ei`, and

```text
D = d12*d13*d14*d15*d24*d35
X = (p1.F4.p5)*(p1.F5.p4)
Y = (p2.F4.p5)*(p3.F5.p4)
R = Tr(F4.F5)
```

With `Fi^(mu nu) = pi^mu*ei^nu - ei^mu*pi^nu` and mixed-index matrix
composition, `R = 2*((e4.p5)*(e5.p4) - d45*(e4.e5))`.

The two inputs are the uncollected scalar expansions of

```text
U_Y = (d24*(e4.p5) - (p2.e4)*d45)^2
      * (d35*(e5.p4) - (p3.e5)*d45)^2 / D

U_X = (d14*(e4.p5) - (p1.e4)*d45)^2
      * (d15*(e5.p4) - (p1.e5)*d45)^2 / D
```

These displayed squares explain the inputs; they are not the compact text fed
to the network. The actual expanded strings are saved in the run outputs.
For the original checkpoint, the actual greedy outputs are equivalent to

```text
M_Y = Y^2 / D
M_X = X^2 / D
```

For independent commuting dot products, the full reconstruction is

```text
A = R^2*(d23*d45 - 3*d14*d15)/(24*d23*d45^3)
    - D/(d24*d35*d45^4) * M_Y
    + D*(d45 - d23)/(2*d14*d15*d23*d45^4) * M_X.
```

The trace block and both external coefficients are analytic. Simplifying this
expression gives the unrestricted 99-token result. It has exact symbolic
residual zero against the supplied CSV without helicity, mass-shell or
momentum-conservation assumptions.

For four-dimensional massless positive-positive helicities, a common
nonsingular reference `r=p1` gives `e4.p1=e5.p1=e4.e5=0`. The reconstruction is

```text
A_pp = -D/(d24*d35*d45^4) * M_Y
       + D*(d45^2 - 3*d14*d15)/(6*d14^2*d15^2*d45^4) * M_X.
```

Symbolic cancellation gives the 67-token version. The additional
same-helicity identity `X/(d14*d15) = R/2` gives

```text
A_pp = [R^2*(d45^2 - 3*d14*d15)/24 - Y^2/(d24*d35)] / d45^4.
```

This last form has 53 tokens. The 67- and 53-token reconstructions are restricted
to the positive-positive sector; they are not identities for arbitrary
polarizations. Gauge invariance extends the common-reference derivation to
other nonsingular reference choices within that sector. Use the unrestricted
99-token result when no helicity restriction is intended. All rational
identities are understood away from their denominator zeros.

## Attribution and expected results

The neural model supplies the compact field-strength representation of two
cores. The analytic preparation already factors their scalar polynomials before
expanding them for inference. Grouping, normalization, coefficient extraction,
the trace block, reconstruction and final cleanup are analytic operations.
Consequently this is a hybrid reproduction, not evidence that the network
autonomously discovered the complete compact amplitude or is mathematically
indispensable to the reduction. No earlier compact answer file is required.

For the original source, checkpoint and current tokenizer:

| Object | Content tokens | Scope |
|---|---:|---|
| Original scalar CSV expression | 899 | Supplied rational function |
| Each expanded neural input | 895 | Scalar core |
| Each successful neural output | 47 | Same scalar core |
| Weighted two-call reconstruction | 205 | Positive-positive helicity |
| Reconstruction after algebraic cleanup | 67 | Positive-positive helicity |
| Reconstruction after the additional trace identity | 53 | Positive-positive helicity |
| Reconstruction with the analytic trace block | 99 | Unrestricted rational identity |

Counts exclude BOS/EOS and depend on the serialization. The final compact
expressions use powers and grouped products; expanding powers in the 67- and
53-token versions yields 107 and 78 tokens respectively. Two 895-token inputs
also require more total input tokens than one 899-token input. No claim of
optimal expression length or inference-cost savings is made.

## Evidence in each complete run

The output directory records the scalar inputs, actual decoded predictions,
external coefficients, reconstructed expressions and validation results. The
principal expression files are:

- `two_call_reconstruction.txt`: weighted positive-positive reconstruction.
- `model_plus_cleanup_67.txt`: positive-positive result after analytic cleanup.
- `model_plus_helicity_cleanup_53.txt`: result after the same-helicity identity.
- `model_plus_cleanup_general.txt`: unrestricted rational identity.
- Matching final `.csv` files: headerless `id,expression` exports.

`grouped_derivation.json` and `generic_derivation.json` record the analytic
decompositions, with `generic_scalar_identity.txt` preserving the unrestricted
scalar identity. `grouped_inputs.json` and `grouped_inputs.csv` record prepared
groups. `prepared_model_inputs.json` and `model_inputs.csv` preserve the two
selected oriented inputs, including in preparation-only runs.
`successful_model_inputs.csv` is written only after actual inference and
reconstruction verification; `selected_groups.json`, `serialization_results.json` and
`two_call_summary.json` record inference and reconstruction evidence.
`final_verification.json` records the final verification results. Consult
`run_manifest.json` for the source/checkpoint identity and run settings; the
token counts above describe the original experiment, not a guarantee for
different inputs or weights.

Verification includes exact equality of each decoded core to its scalar input,
the full unrestricted reconstruction to the original, the same-helicity
reconstructions in the common reference, and semantic tokenizer roundtrips.
Numerical checks use 20 momentum seeds (401–420) and four reference/gauge
choices, for 80 comparisons per final expression. All numerical checks use the
positive-positive sector, including the checks of the 99-token expression;
its unrestricted validity rests on the exact symbolic identity. The original
run's largest relative discrepancy was below `1.2e-11`.

Read verification records before treating a run as successful. The complete
pipeline exits unsuccessfully if either selected prediction fails the exact
identity check or does not terminate with EOS. The package keeps provenance
and derived results together and does not rely on the previous experiment's
output directories or the earlier nine-call reconstruction.

## Code map

| Module | Responsibility |
|---|---|
| `__main__.py` | CLI and stage orchestration |
| `common.py` | Input/output handling, hashes, numerical checks and validation |
| `derive_generic_groups.py` | Unrestricted scalar decomposition |
| `grouped_model.py` | Common-reference grouping and scalar preparation |
| `serialize_groups.py` | Fixed selected serialization and two greedy calls |
| `verify_final.py` | Reconstruction, final cleanup and verification |
| `symbolic_utils.py` | Symbolic handling of compact field-strength notation |

Run the small contract and failure-path tests without a model checkpoint:

```sh
python -m unittest data_testing.gravity_5pt_workflow.test_workflow
```
