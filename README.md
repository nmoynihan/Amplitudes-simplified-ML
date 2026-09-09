# Amplitudes-simplified-ML

**Teaching a Transformer to simplify scattering amplitudes**, framed as translation: given a long,
messy-but-correct form of an amplitude, predict a short equivalent one.

> ⚠️ Research code, under active development. Numbers below are preliminary, not a benchmark.

## Idea

Particle-physics calculations produce huge algebraic expressions. The same quantity can be written
many equivalent ways; finding the compact one by hand is slow expert work. We treat it as a
**sequence-to-sequence translation** task: `scrambled → simple` expressions. Crucially, "correct" is
checkable: plug in random kinematics and confirm two expressions evaluate to the same number.

## Pipeline

```
BUILD     a compact "simple" amplitude from gauge-invariant blocks (Tr(F···F), p·F···F·p + poles)
EXPAND    field-strength tensors into dot-products (e·p, e·e, p·p)
SCRAMBLE  apply 1–5 value-preserving identities (Ward, momentum conservation, partial fractions, …)
VALIDATE  evaluate simple vs scrambled at random kinematics in two gauges; reject mismatches
TOKENISE  infix → prefix → integer tokens; write CSV (+gzip)
TRAIN     encoder–decoder Transformer learns scrambled → simple
EVALUATE  decode held-out scrambled inputs; score exact-match, token accuracy, numerical equivalence
```

Each `simple ↔ scrambled` pair is a verified supervised example. The model
(`TransformerRegressor`, `transformer/transformer_functions.py`) is a standard PyTorch
encoder–decoder Transformer. Training uses cross-entropy + AdamW + warmup/cosine LR, early stopping,
length-bucketed batches, and checkpoints that store their own architecture args. Evaluation scores
exact-match, token accuracy, and numerical equivalence.

## Physics

Tree-level amplitudes for $N$ external legs, signature $(+,-,-,-)$, $\sum_i p_i = 0$. Built from
Lorentz invariants $p_i\cdot p_j$, $e_i\cdot p_j$, $e_i\cdot e_j$ and field-strength tensors
$F_i^{\mu\nu} = p_i^\mu e_i^\nu - e_i^\mu p_i^\nu$, with mass dimension $4-N$. The scramble identities
(Ward $\sum_s e_j\cdot p_s = 0$, momentum conservation $\sum_i p_i = 0$, partial fractions, …) change
the form but not the value.

Three generators share the machinery:

- **Scalar QED** (`data_gen/gen_data.py`): 2 massive scalars + $(N{-}2)$ massless photons; poles
  $p_i\cdot p_j$ allowed between any pair.
- **Colour-ordered Yang–Mills** (`data_gen/data_gen_ym/`): all $N$ legs massless gluons, one fixed
  colour ordering (no colour factors), so only **planar adjacent poles** $p_i\cdot p_{i+1}$ (cyclic)
  are physical.
- **Five-point gravity** (`data_gen/data_gen_gravity/`): either three scalars plus two
  same-helicity gravitons or four scalars plus one graviton. Each graviton is represented by two
  separate $p\cdot F_i\cdot p$ contractions and checked with complex spinor kinematics.

## Status / results

> ⚠️ Trained models are too large to be uploaded on GitHub. They will be available through other means once results are published.

Preliminary. The full generate → validate → train → check loop works end-to-end.

A 5pt sQED, 500k pairs, ~16M-param model trained on GPU cluster reached accuracy ≈ 93%.

A 4pt Yang-Mills, 50k pairs, ~4M-param model trained on an Apple M2 GPU reached accuracy ≈ 86%.

Open: scaling up, 6+ legs (multi-particle poles).

## Setup and usage

`pip install -r environment/requirements.txt`.

Example runs from the repo root:

```bash
# Generate — scalar QED (4-point, 50k) and Yang–Mills (4-gluon, 50k)
python3 data_gen/gen_data.py 4 --samples 50000 --seed 42 \
    --raw-out data/sqed_4pt_10k.csv --tok-out data/sqed_4pt_10k_tok.csv
SAMPLES=50000 SEED=7 ./data_gen/data_gen_ym/run_ym.sh 4 \
    --raw-out data/data_ym/ym_4pt_50k.csv.gz --tok-out data/data_ym/ym_4pt_50k_tok.csv.gz

# Generate/train/evaluate the balanced five-point gravity workflow
./run_gravity_100k.sh data
./run_gravity_100k.sh train
./run_gravity_100k.sh eval

# Train (--data-files resolves under ./data; transformer_trainer_paolo.py is the Apple-MPS variant)
python3 transformer/transformer_trainer.py --data-files "sqed_4pt_10k_tok.csv"        --run-name sqed_4pt_10k
python3 transformer/transformer_trainer.py --data-files "data_ym/ym_4pt_50k_tok.csv.gz" --run-name ym_4pt_50k

# Evaluate (configured in-file) / infer + numerically check a specific amplitude
python3 transformer/transformer_evaluator.py
python3 data_testing/evaluate_single_amplitude.py \
    models/ym_4pt_50k_mps/best_model.pt data/data_ym/gluon4feyn.csv.gz \
    --numeric-backend ym --n-particles 4 --decoding-method beam --beam-size 5
```

`evaluate_single_amplitude.py` accepts raw or tokenised `simple,scrambled` data, JSON token-list
CSVs, headered expression CSVs, and headerless `id,expression` Feynman CSVs (plain or
gzip-compressed).
It counts the amplitude rows and evaluates only the first one using the selected strict numerical
backend (`sqed`, `ym`, or `gravity`). Gravity inputs also need `--gravity-process 3s2h/4s1h`
unless that process is present in the selected row or its standard sibling metadata CSV.

`data_testing/evaluate_nucleus_search.py` repeatedly samples each test amplitude until a strict
numerical match is found or its decoding-call budget is exhausted. For a small run:

```bash
python3 data_testing/evaluate_nucleus_search.py \
    --checkpoint models/best_model.pt \
    --test-data data/sqed/sqed_4ptseed_oneshot.csv.gz --input-format raw \
    --max-rows 10 --max-attempts 20 --beam-size 4 \
    --p-nucleus 0.95 --temperature-nucleus 1.0 --max-length 512 \
    --sampling-seed 42 --numeric-seed 151 --numeric-samples 3 --device cpu \
    --output-dir data_testing/outputs/nucleus_search
```

Substitute an available checkpoint. Defaults are documented in `RunConfig`; `--help` lists all
overrides, including particle count, backend, mass/energy scale, tolerances, polarization modes,
gravity reference modes and process metadata. Relative paths resolve from the repository root.
`--max-length` includes BOS/EOS; both input and output lengths must fit the checkpoint.
The script supports the same CSV/CSV.GZ layouts as the single-amplitude evaluator; token lists
must contain content IDs only. Auto-detection recognizes JSON lists in scrambled pair cells.
Use `--tokens-column`, `--expression-column`, and `--id-column` for custom layouts.
Gravity requires `--numeric-backend gravity --n-particles 5`, plus an inline `process` column,
`--gravity-metadata-csv`, or `--gravity-process 3s2h/4s1h`. Metadata aligns by source row order;
a standard `_raw.csv[.gz]` input can discover its sibling `_metadata.csv[.gz]`.

Every attempt is one nucleus decode call; `--beam-size` controls hypotheses per call independently.
Candidates are checked top-1 first, then in the decoder's returned hypothesis order. Generation
and comparison always use the original scrambled input, even when the simple target differs.
There is no length requirement for a match. Duplicate/malformed candidates consume their attempts;
checks of repeated token candidates are cached separately for each amplitude. Sampling is seeded
once after setup, independently of the numerical seed; identical runs require the same software,
device, settings and dataset order. No full evaluation is launched by the example's row cap.

Each run creates a unique output subdirectory containing `config.json`, `results.csv` and
`summary.json`. Completed result rows flush immediately and retain source record ordinals and
IDs (including duplicate IDs); invalid inputs receive explicit diagnostics and zero attempts.
`candidates_checked` counts unique normalized candidates including malformed outputs;
`candidates_encountered` includes repeats, with `cache_hits` recording skipped repeated checks.
Success rate uses valid searched inputs (`matched + exhausted`) as its stated denominator;
the summary separately reports invalid inputs, dataset/selected counts, attempts to success and runtime.

Focused tests: `python3 -m unittest data_gen.test_evaluate_nucleus_search`.

Datasets are named `<theory>_<N>pt_<size>[_tok].csv[.gz]` (`ym_`/`sqed_`/`gi_`; `_tok` = tokenised).
`transformer_evaluator.py` and the Optuna hyperparameter-search scripts (`optuna_*.py`) are configured
in-file. The `run_*.sh` scripts chain generate → train → evaluate at fixed scales.

## Citation

```
raise NotImplementedError("Paper yet to be published.")
```
