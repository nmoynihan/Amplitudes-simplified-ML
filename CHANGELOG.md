# Changelog

## Unreleased

### Data Generation

- Added three new scramble functions to `data_gen/gen_data.py`:
  - `scr_ward_substitute_all` (`ward_all`) — like `scr_ward_substitute` but
    replaces *every* occurrence of the chosen `e_j·p_k` in the expression
    rather than just one, producing longer Ward-sum expressions.
  - `scr_add_polarisation_zero` (`polarisation_zero`) — adds a multiple of the
    Ward-identity zero `Σ_s e_j·p_s = 0` (optionally scaled by a context dot
    product and divided by an existing denominator) with a random sign.
  - `scr_term_reorder` (`term_reorder`) — shuffles the top-level additive terms
    of a fully-expanded expression; purely positional diversity with no
    algebraic content.
  - All three are registered in `_SCRAMBLER_BY_NAME` and included in
    `DEFAULT_SCRAMBLES`. `ward_all` and `polarisation_zero` receive `Ngamma`
    and `N` the same way `scr_ward_substitute` does.

- Updated `gen_data.sh` with a broad stratified default profile:
  70% broad, 20% SQED-cover (Paolo's data), 10% hard/weird.
- Tuned default generator probabilities for the current SQED-focused training goal:
  `UNIT_PROBABILITY=0.6`, `OLD_STYLE_PROBABILITY=0.35`,
  `SQED_COVER_UNIT_PROBABILITY=0.9`, and
  `SQED_COVER_OLD_STYLE_PROBABILITY=0.55`.

### Training

- Added cosine LR schedule with linear warmup (5% of epochs) via `SequentialLR`.
  Warmup ramps LR from 10% to 100% of the peak value; cosine phase decays to 1% of
  peak. Scheduler state is saved and restored on resume.
- Changed default learning rate from `1e-4` to `3e-4` (peak LR for cosine schedule).
- Changed default `--amp-dtype` from `fp16` to `bf16` (recommended for RTX 40/50
  series GPUs; more numerically stable, no loss scaler required).
- Added gradient clipping via `--grad-clip` (default `1.0`). Uses
  `scaler.unscale_()` before clipping when AMP fp16 is active.
- Added label smoothing via `--label-smoothing` (default `0.1`).

- Reworked `transformer/transformer_trainer.py` to use CLI arguments instead of
  hardcoded training settings.
- Added CLI support for:
  `--data-files`, `--epochs`, `--batch-size`, `--max-length`, `--train-split`,
  optimizer settings, early stopping settings, and model-size hyperparameters.
- Added dynamic per-batch padding in `transformer/data_import.py`.
- Added length-bucketed batching to reduce padding waste on long sequence datasets.
- Added AMP mixed-precision training with `--amp/--no-amp` and
  `--amp-dtype {fp16,bf16}`.
- Enabled pinned memory and non-blocking tensor transfers for CUDA training.
- Set CUDA matmul precision to `high` during training.
- Added checkpoint resume support via `--resume-from`.
- Changed the default model in `run.sh` to a middle-size configuration:
  `EMBEDDING_DIM=384`, `N_HEADS=6`, `N_ENC_LAYERS=5`,
  `N_DEC_LAYERS=5`, `HEAD_FF_DIM=1024`, `DROPOUT=0.05`.
- Changed default training speed settings in `run.sh`:
  `MAX_TOKENS=1024`, `BATCH_SIZE=8`, `TRAIN_SPLIT=0.9`,
  `DYNAMIC_PADDING=1`, `BUCKETING=1`, `AMP=1`.
  Note: Setting max tokens to 1024 cuts of some particularly long (scrambled) outliars, but gives a good speedboost. Worth it for testing, but for a proper run maybe 2048 or 4096 is better. 

### Run Script

- Updated `run.sh` to use `gen_data.sh` for data generation, so training uses the
  same stratified generator profile by default.
- Added `SKIP_DATA_GEN`; when enabled, `run.sh` uses an existing tokenized CSV.
- Added deterministic training-row capping: if the tokenized CSV has more rows than
  `SAMPLES`, `run.sh` writes a sampled subset and trains on exactly `SAMPLES` rows.
- Added generated-data evaluation followed by external test CSV evaluation in the
  same run.
- Added `EVAL_TEST_CSV` and `EVAL_TEST_CSV_MAX_ROWS`.
- Added `RESUME=1`, which automatically resumes from the latest
  `models/$RUN_NAME/model_epoch_*.pt`, falling back to `best_model.pt`.
- Added `RESUME_FROM=...` for explicit checkpoint selection.

### Evaluation

- Added `eval_compare.sh`: a dedicated comparison script that evaluates multiple
  trained runs (e.g. 100k, 200k, 500k) across multiple test sets in one invocation.
  Options via environment variables: `RUN_NAMES`, `TEST_CSVS`, `SAMPLES`,
  `BEAM_SIZE`, `BEAM_SIZE_GENERATED`, `EVAL_GENERATED`, `SIMPLE_SUMMARY`.
- Added `--simple-summary` / `--no-simple-summary` flag to
  `evaluate_model_on_generated_data.py` (default: on). Prints a concise per-run
  block showing top-1 correct, any-beam correct, rerank gain, avg valid beams, and
  avg token counts (predicted vs scrambled) for correctly solved examples.
- Added `avg_pred_token_count_when_correct` and
  `avg_scrambled_token_count_when_correct` to the summary dict, computed over rows
  where `top1_num_eq_scrambled` is true.

- Expanded `data_testing/evaluate_model_on_generated_data.py` CLI support so it can
  be driven directly from `run.sh`.
- Added support for generated eval data and existing raw CSV eval data.
- Added options for model path, device, particle count, sample count, seed, scramble
  bounds, term bounds, token limits, decoding method, beam size, output stem, and
  existing CSV row caps.
- Added optional numerical-equivalence reranking over beam candidates. When enabled,
  the evaluator chooses the shortest decoded beam that is numerically equivalent to
  the scrambled input as the reported top-1 prediction.
- Enabled numerical beam reranking by default in `run.sh` via
  `EVAL_RERANK_NUMERICAL=1`.

### Notes

- The current fast training path is much faster primarily because dynamic padding
  and bucketing avoid padding most batches to the longest sequence in the dataset. 40x speedup on my 5070!
- `SKIP_DATA_GEN=1` does not apply generator-profile changes; it uses whatever
  already exists at `TOK_OUT`.
- `RESUME=1` defaults to skipping data generation unless `SKIP_DATA_GEN` is set
  explicitly.
