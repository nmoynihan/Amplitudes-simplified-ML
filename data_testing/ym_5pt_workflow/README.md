# Five-point Yang–Mills simplification workflow

This package reproduces the 16-component workflow used for the five-point YM
technical note. It includes the symbolic preparation, actual checkpoint
inference, cyclic reconstruction, and independent exact and numerical checks.
It has no dependency on the original investigation folders or saved prediction
logs. The checkpoint remains external.

**The result is the complete cyclic amplitude.** The input
`data/data_ym/gluon5feyn12345.csv.gz` is one seed, S. The workflow first verifies

```text
A = sum(k=0,...,4) rho_k(S) = data/data_ym/gluon5feyn.csv.gz,
rho_k: p_i, e_i, F_i -> p_(1+(i-1+k) mod 5), e_(...), F_(...).
```

There is no factor of 1/5. This completion adds missing contributions and changes
the original seed into the full amplitude; the model's final output equals A.

## Run the simplification

Run commands from the repository root using its Python environment
(`pip install -r environment/requirements.txt`). The workflow uses PyTorch,
SymPy, NumPy and the repository tokenizer/decoder. It was checked with SymPy
1.14.0; changing symbolic rendering can change the model's input order, so the
original prepared inputs are included in `prepared_components.json`.

```sh
python -m data_testing.ym_5pt_workflow \
    --checkpoint /path/to/5POINT_YM_Model_500k_Colour/best_model.pt \
    --output-dir /tmp/ym5-run
```

Use a new or empty output directory. The original checkpoint SHA-256 is
`b0918473800c014699752f9a1d1a9e080c18c1e2580ba4e1c977e6ada5203179`
(epoch 52). A different checkpoint may be supplied, but every actual prediction
must pass the same verification. The runner loads weights with
`torch.load(..., weights_only=True)`, uses CPU with two threads, and performs
unmasked greedy decoding, with a maximum sequence length of 256 including BOS.
Use `--threads` or `--max-output-tokens` to change these settings.

The run writes artifacts only after every verification passes:

| File | Contents |
|---|---|
| `model_ready_components.csv` | The 16 headerless `id,expression` scalar model inputs |
| `component_predictions.csv` | Each actual prediction and its external coefficient |
| `gluon5feyn12345_gauge_invariant_cyclic_core.csv` | Weighted sum of the 16 actual predictions |
| `gluon5feyn12345_completed_simplified.csv` | Sum of the five cyclic images of that core: the final amplitude |
| `results.json` | Input/checkpoint hashes, emitted token IDs, predictions, token counts, exact residuals, numerical checks and timing |

The core alone is not the complete amplitude. Do not concatenate the 16 CSV rows
as a single model input. Each is decoded separately; its coefficient is applied
once, outside the model.

To verify the checked-in preparation without loading any checkpoint:

```sh
python -m data_testing.ym_5pt_workflow \
    --verify-only --output-dir /tmp/ym5-source-check
```

This checks the complete amplitude's Ward identities, all source token
roundtrips, nonzero sources, exact source reconstruction, and numerical
reconstruction. Its report records zero model calls and does not claim model
success or write predicted amplitudes.

## Rebuild the symbolic preparation

```sh
python -m data_testing.ym_5pt_workflow.prepare \
    --output-dir /tmp/ym5-prepared

python -m data_testing.ym_5pt_workflow \
    --components /tmp/ym5-prepared/prepared_components.json \
    --checkpoint /path/to/5POINT_YM_Model_500k_Colour/best_model.pt \
    --output-dir /tmp/ym5-regenerated-run
```

Preparation projects each polarization using q_i = p_(i+1), with labels modulo
five and F_i = p_i tensor e_i − e_i tensor p_i:

```text
e-hat_i = e_i - (q_i.e_i)/(q_i.p_i) p_i = q_i.F_i/(q_i.p_i).
```

Gauge invariance gives A(p,e-hat) = A(p,e). Substituting into the complete scalar
expression and removing on-shell zero contractions yields 80 nonzero tensor
monomials. They form 16 cyclic orbits. The fixed historical representative
from each orbit is expanded back into scalar p/e contractions, normalized with
the existing generator routines, and given to the model without a tensor
reference answer. The external coefficients are +1/2 or −1/2.

The preparation command writes `projected_components.json` (all 80 sources and
tensor references), `orbit_manifest.json` (the exact grouping and provenance),
`prepared_components.json` (only the 16 sources and coefficients), and
`model_inputs.csv`. The runner reads only the prepared inputs. Rebuilding with
the verified environment reproduces the checked-in prepared JSON byte for byte
(SHA-256 `ebebf438cae38961ac8c676a0f7063db013996d7a77bd2c4a5b33b86dc3be423`).

The compact decomposition was derived analytically before inference. The
representatives were selected using exploratory model predictions, choosing a
short successful input in each orbit. They are fixed in `prepare.py`; rebuilding
does not repeat that search. This is a reproduction of this worked example,
not an unbiased accuracy benchmark or a general automatic decomposition of
arbitrary amplitudes. Neither checkpoint training nor a successful single-call
simplification is part of this workflow.

## Verification and expected result

The independent parser expands every actual predicted F-chain/trace directly
from the tensor definition. It proves U_j = M_j for each scalar input U_j and
actual model output M_j, then proves

```text
A = sum(k=0,...,4) rho_k(sum(j=1,...,16) c_j M_j).
```

The exact quotient imposes p_i² = 0, sum_i p_i = 0 and e_i.p_i = 0. It uses no
dimension-specific Gram identities. Equalities of rational expressions are
understood away from denominator poles. Numerical verification uses 20 random
transverse-polarization points in each of Coulomb and covariant modes, with
independent polarization vectors and random gauge shifts in covariant mode.
The defaults are seed 310921, atol 1e-9 and rtol 1e-8.

The verified checkpoint gives 16/16 exact greedy matches, a zero exact
reconstruction residual, and 40/40 numerical matches. Inputs contain 9,730
content tokens in total; the largest call is 1,121 tokens including BOS/EOS,
below the checkpoint's 5,000-token capacity. The actual component predictions
contain 683 content tokens in total.

The exported full expression contains 4,014 content tokens. This is the same
amplitude as the original workflow's 3,854-token rendering, with external
divisors written as `-(-2)` instead of `2`. The repository tokenizer merges
adjacent digit tokens in some prefix trees; these unary operators separate
the digits. The runner proves the exported expression survives a complete
encode/decode roundtrip. The original full expression contains 10,759 tokens.

The run aborts on unfinished/invalid predictions, positional overflow, incorrect
source/output identities, failed token roundtrips, or failed numerical checks.
It never replaces a failed prediction with a reference expression.

## Code and tests

- `prepare.py`: projection, scalar preparation, and exact cyclic orbit grouping.
- `inference.py`: checkpoint loading and the 16 actual greedy calls.
- `workflow.py`: orchestration, reconstruction, verification and export.
- `algebra.py`: strict arithmetic parser, independent F expansion and exact on-shell identities.
- `verification.py`: numerical kinematics, polarizations and comparisons.
- `test_workflow.py`: regression checks without a checkpoint.

```sh
python -m unittest data_testing.ym_5pt_workflow.test_workflow
```
