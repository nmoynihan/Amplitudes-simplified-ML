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

Two generators share the machinery:

- **Scalar QED** (`data_gen/gen_data.py`): 2 massive scalars + $(N{-}2)$ massless photons; poles
  $p_i\cdot p_j$ allowed between any pair.
- **Colour-ordered Yang–Mills** (`data_gen/data_gen_ym/`): all $N$ legs massless gluons, one fixed
  colour ordering (no colour factors), so only **planar adjacent poles** $p_i\cdot p_{i+1}$ (cyclic)
  are physical.

## Status / results

> ⚠️ Trained models are too large to be uploaded on GitHub. They will be available through other means once results are published.

Preliminary. The full generate → validate → train → check loop works end-to-end.

A 5pt sQED, 500k pairs, ~16M-param model trained on GPU cluster reached token accuracy ≈ 93%.

A 4pt Yang-Mills, 50k pairs, ~4M-param model trained on an Apple M2 GPU reached token accuracy ≈ 86%.

Open: scaling up, 6+ legs (multi-particle poles).

## Setup & usage

Python 3.9. `conda create -n ml_amplitudes && conda activate ml_amplitudes && pip install -r environment/requirements.txt`.

Run from the repo root:

```bash
# Generate — scalar QED (4-point, 10k) and Yang–Mills (4-gluon, 50k, compressed)
python3 data_gen/gen_data.py 4 --samples 10000 --seed 42 \
    --raw-out data/sqed_4pt_10k.csv --tok-out data/sqed_4pt_10k_tok.csv
SAMPLES=50000 SEED=7 ./data_gen/data_gen_ym/run_ym.sh 4 \
    --raw-out data/data_ym/ym_4pt_50k.csv.gz --tok-out data/data_ym/ym_4pt_50k_tok.csv.gz

# Train (--data-files resolves under ./data; transformer_trainer_paolo.py is the Apple-MPS variant)
python3 transformer/transformer_trainer.py --data-files "sqed_4pt_10k_tok.csv"        --run-name sqed_4pt_10k
python3 transformer/transformer_trainer.py --data-files "data_ym/ym_4pt_50k_tok.csv.gz" --run-name ym_4pt_50k

# Evaluate (configured in-file) / infer + numerically check a specific amplitude
python3 transformer/transformer_evaluator.py
python3 data_gen/data_gen_ym/infer_amplitude.py \
    --model models/ym_4pt_50k_mps/best_model.pt --csv data/data_ym/gluon4feyn.csv --N 4
```

Datasets are named `<theory>_<N>pt_<size>[_tok].csv[.gz]` (`ym_`/`sqed_`/`gi_`; `_tok` = tokenised).
`transformer_evaluator.py` and the Optuna hyperparameter-search scripts (`optuna_*.py`) are configured
in-file. The `run_*.sh` scripts chain generate → train → evaluate at fixed scales.

## Citation

```
raise NotImplementedError("Paper yet to be published.")
```
