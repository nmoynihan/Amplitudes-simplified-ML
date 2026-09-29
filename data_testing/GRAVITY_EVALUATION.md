# Gravity evaluation: input integrity and candidate evidence

The evaluator preserves the checkpoint, architecture and tokenizer IDs. It checks
the actual input representation before inference and records the model's actual
retained hypotheses before token cleanup. The generator changes and **unexecuted**
future 100,000-training/200-test command are in
[GRAVITY_V2.md](../data_gen/GRAVITY_V2.md).

## Input and reference contract

`evaluate_single_amplitude.py` recognizes paired raw expressions, paired token
lists, a named token column, a named expression column and headerless
`id,expression` rows. It uses the first amplitude and reports additional rows.
An expression-only input has no independent target. Its internally copied target
is marked as a placeholder: exact-target and target-equivalence metrics are
`null` in JSON and empty in CSV. Paired inputs supply independent references.
`--reference-provided` / `--no-reference-provided` can override inference in the
general evaluator when importing a legacy file with known provenance.

Before inference, content token IDs, full-prefix decoding, process labels, model
vocabulary and positional capacity are checked. Inputs are never silently
truncated. The original expression is compared with its actual decoded tokens
using exact gravity normalization where supported, followed by strict numerical
checks. Unsupported syntax and changes caused by ambiguous adjacent numeric leaves
fail explicitly. The checkpoint-compatible format remains restricted; the code
does not renumber IDs or repair arbitrary unrepresentable expressions.

The manifest contains the original input and checkpoint hashes, original
expressions/tokens, decoded expressions, input-integrity evidence, tokenizer map
and hash, code hashes, device, model capacity, effective decode and numerical
settings, seeds and the numerical sample schedule. A frontend's temporary row
retains the original input file identity via `--input-provenance-path`.

## Raw candidates and metrics

The sequence contract permits one initial BOS, a terminating EOS and trailing PAD
after EOS. Internal PAD/BOS/UNK, unmapped IDs and non-padding content after EOS
are invalid. A sequence at the length ceiling without EOS remains incomplete.
An incomplete sequence may have a meaningful numerical value, but cannot count
as a successful simplification or an exact completed target match.

Each mode writes `*_candidates.jsonl`, with one record per input containing every
retained raw hypothesis, duplicates included. Each candidate keeps its raw IDs,
model rank, available score, completion and stop status, decoded expression or
error, structured numerical evidence and length metrics. `*_manifest.json`
provides run-wide provenance. Existing detail, human and summary CSVs remain.
Failures write `*_failure.json`; an input-integrity failure retains its attempted
check. Terminal transcript PDFs retain the existing evaluator output location.

Counts distinguish generated retained hypotheses, unique raw sequences, checked
hypotheses, checked unique sequences, valid parses, complete sequences,
numerically equivalent candidates and shorter equivalent candidates. Pruned
search paths are not materialized as returned candidates; decoder metadata also
records terminal proposals and stopping information. A finite beam is not an
exhaustive algebraic search. Unchecked candidates retain token evidence and
lengths, with unavailable numerical/simplification results marked accordingly.

Token reduction is source content length minus candidate content length;
BOS/EOS/PAD are excluded. Successful simplification requires a valid, complete,
numerically equivalent candidate with **strictly fewer content tokens**. Copying
is recorded separately. The legacy CSV column `correct` means numerical
equivalence, while `shorter_equivalent` means successful simplification. Original
model top-1 metrics and the original candidate remain separate from selection.
Optional numerical reranking chooses the shortest complete equivalent generated
candidate; ties use model rank. A valid display fallback retains its actual
numerical verdict. The independently known answer is never injected into search.

`--mask-invalid-tokens` optionally prevents PAD, UNK, internal BOS and unmapped
model output IDs in greedy, beam and nucleus modes. Its default is off, preserving
the baseline policy. Mask and score/renormalization semantics are recorded. Wider
beams or masking do not guarantee a correct simplification.

The low-level decoder retains its two-return legacy API for existing callers;
`return_diagnostics=True` adds raw evidence as a third result. All modes in
`evaluate_model.py` use this evidence. The separate iterative
`evaluate_nucleus_search.py` retains its legacy output contract and does not yet
emit the new full-candidate artifact.

## Numerical domain and failures

`numerical_comparison(source, candidate, cache, gravity_process=...)` returns a
JSON-safe result. `numerically_equivalent_exprs(...)` remains its boolean wrapper.
Statuses distinguish equivalent expressions, numerical mismatch, invalid or
unsupported expressions, evaluation exceptions, nonfinite values and insufficient
valid samples. Raw token failures are reported before numerical evaluation.
Every valid mismatch retains both complex values, absolute error, relative error,
scaled error, sample index and tolerances.

The unchanged comparison rule is
`abs(source-candidate) <= max(atol, rtol*max(abs(source),abs(candidate)))`.
Relative error divides by the larger value magnitude; scaled error divides by the
allowed error. Gravity defaults are `atol=2e-9`, `rtol=2e-8`. Three momentum seeds
151–153, each with first/last/random references and a cyclic-reference gauge
shift, give 12 required checks per process. The manifest records exact reference
seeds and shifts. This oracle uses positive-helicity gravitons; passing it alone
does not prove an identity for arbitrary polarization.

The CLI requires every configured check and does not resample. A singular source
is inconclusive, not a pass. The backend-independent `compare_expressions` API
also supports an explicitly bounded, preselected replacement pool: only source
arithmetic/nonfinite failures can be replaced; candidate errors and any valid
mismatch can never be skipped. General generator identities additionally receive
exact symbolic and general-polarization checks.

## Reproduce the bounded checkpoint checks

From the project root, this runs only the supplied amplitude, the three saved
training controls, separate explicit oracle checks and report consistency checks:

```bash
/Users/kymani/miniforge3/envs/scattering/bin/python \
  ../gravity-evaluator-improvements/reproduce_evaluation.py
```

For a single beam-20 run through the public CLI:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/Users/kymani/miniforge3/envs/scattering/bin/python -m data_testing.evaluate_single_amplitude \
  /Users/kymani/Documents/Big_YM_Data_Sets/Cyclic_Gravity/Model/best_model.pt \
  /Users/kymani/Documents/Big_YM_Data_Sets/Cyclic_Gravity/Test_Amplitude/gravity5unified12345_seed.csv \
  --numeric-backend gravity --gravity-process 3s2h --device cpu \
  --decoding-method beam --beam-size 20 --max-decode-tokens 256 \
  --sampling-seed 0 --numeric-seed 151 --numeric-samples 3 \
  --no-mask-invalid-tokens --no-rerank-numerical --no-plots \
  --output-stem /Users/kymani/Documents/New_Scattering_AI/Repo/gravity-evaluator-improvements/after/user_beam20
```

Use `--decoding-method greedy` and a different output stem for greedy. The decode
limit includes BOS/EOS; the input remains 899 content tokens / 901 with BOS/EOS,
within the checkpoint's 5,000-position capacity. `--sampling-seed` and
`--numeric-seed` do not invoke data generation. The older `--seed` flag is a
synthetic-data-generation option and should not be used for this purpose.

The measured result remains zero equivalent candidates for greedy and beam-20.
Both compact reference expressions pass when explicitly checked as oracles, and
all three saved training controls still simplify correctly. These controls are
not a general accuracy estimate. Reports, full commands and test results are in
`../gravity-evaluator-improvements/`. Dataset production and model training were
not run.

The full 245-test command is in
`../gravity-evaluator-improvements/regression_command.txt`; its recorded result is
`regression_tests.log` in the same directory. It includes existing generator and
shared evaluator regressions plus the new bounded contract tests.
