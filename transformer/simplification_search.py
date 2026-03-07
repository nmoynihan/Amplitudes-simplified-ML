"""
simplification_search.py

For each tokenized expression in a dataset, repeatedly asks the trained model
to produce a simpler form.  A candidate is accepted when it is:
  (a) numerically equivalent to the input, AND
  (b) strictly shorter (fewer content tokens) than the input.

Accepted simplifications and "no simplification found" notes are written to an
output CSV file as they are discovered.

Decoding notes
--------------
* greedy   – deterministic, so number_of_attempts is overridden to 1.
* beam     – also deterministic (temperature has no effect in the non-stochastic
             generate_beam path), so number_of_attempts is overridden to 1.
             Each attempt already checks all beam_size hypotheses.
* nucleus  – stochastic; multiple attempts are genuinely useful.  Temperature is
             varied linearly from temp_min (conservative) to temp_max (exploratory)
             across attempts, giving a principled annealing-style search.
"""

import os
import sys
import csv
import ast
import importlib
import importlib.util as _iutil

import torch
import pandas as pd

# Set CUDA memory allocator to reduce fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Detect and honour OMP_NUM_THREADS
n_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(n_threads)
print(f"Using {n_threads} CPU threads for PyTorch.")

from transformer_functions import TransformerRegressor, load_transformer_model, decode_with_model, clean_seq

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_generation'))
from Tokenizer import ScatteringAmplitudeTokenizer


# ============================================================
# HYPERPARAMETERS – edit these before running
# ============================================================

N_particles = 5  # Number of particles in the expressions

# Model
# Path is relative to the repo root (one level up from this script's directory)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(_repo_root, 'models', 'best_model.pt')

# Input data (relative to data/ directory)
csv_file = '_old/ampl00111_tok.csv'
input_column = 'scrambled'   # Column name containing the expressions to simplify

# Output file (relative to data/ directory)
output_file = 'simplification_results.csv'

# Decoding
decoding_method = 'nucleus'   # 'greedy', 'beam', or 'nucleus'
beam_size = 4                 # Number of beams (beam / nucleus)
p_nucleus = 0.99              # Nucleus cutoff probability
temperature_nucleus = 1.0     # Base temperature for nucleus sampling

# Search budget
# NOTE: automatically overridden to 1 for greedy and beam (both deterministic).
#       For nucleus, the temperature is varied linearly across attempts.
number_of_attempts = 10000

# Temperature schedule for nucleus across attempts
# Attempt 0 uses temp_min (focused), attempt number_of_attempts-1 uses temp_max (exploratory).
temp_min = 0.5
temp_max = 2.5

# Dataset limit
max_datasize = None # Set to an integer to cap the number of expressions processed

# Device
force_cpu = False  # True → always use CPU (useful for quick local tests)
use_mps   = False  # True → prefer MPS over CPU when CUDA is unavailable

# ============================================================
# END HYPERPARAMETERS
# ============================================================


def _make_temp_schedule(n: int, t_min: float, t_max: float) -> list:
    """Return *n* temperatures linearly spaced from *t_min* to *t_max*."""
    if n <= 1:
        return [t_min]
    step = (t_max - t_min) / (n - 1)
    return [t_min + i * step for i in range(n)]


def _local_import(mod_name: str, search_dir: str):
    """Import a module by name, falling back to a file in *search_dir*."""
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError:
        spec = _iutil.spec_from_file_location(
            mod_name, os.path.join(search_dir, f"{mod_name}.py")
        )
        mod = _iutil.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod


def main():
    # ------------------------------------------------------------------
    # 1.  Resolve device
    # ------------------------------------------------------------------
    print(f"PyTorch version  : {torch.__version__}", flush=True)
    print(f"CUDA compiled ver: {torch.version.cuda}", flush=True)
    print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)

    if force_cpu:
        device = 'cpu'
        print("Forcing CPU device.", flush=True)
    else:
        try:
            if torch.cuda.is_available():
                torch.zeros(1).cuda()   # smoke test
                device = 'cuda'
                props = torch.cuda.get_device_properties(0)
                print(f"Using CUDA: {props.name} ({props.total_memory / 1024**3:.1f} GB)", flush=True)
            else:
                raise RuntimeError("CUDA not available")
        except (RuntimeError, AssertionError) as e:
            print(f"CUDA not working: {e}", flush=True)
            try:
                if use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.zeros(1).to('mps')   # smoke test
                    device = 'mps'
                    print("Using MPS.", flush=True)
                else:
                    raise RuntimeError("MPS not available or disabled")
            except (RuntimeError, AssertionError):
                device = 'cpu'
                print("Using CPU.", flush=True)

    # ------------------------------------------------------------------
    # 2.  Load model
    # ------------------------------------------------------------------
    loaded = load_transformer_model(TransformerRegressor, model_path, device=device)
    model = loaded['model']
    model.to(device)
    model.device = device
    model.eval()
    print(f"Model loaded from {model_path} on {device}.", flush=True)

    # ------------------------------------------------------------------
    # 3.  Validate / override number_of_attempts for deterministic methods
    # ------------------------------------------------------------------
    effective_attempts = number_of_attempts
    if decoding_method in ('greedy', 'beam'):
        if number_of_attempts > 1:
            print(
                f"\nNote: decoding_method='{decoding_method}' is deterministic – "
                f"overriding number_of_attempts {number_of_attempts} → 1."
            )
        effective_attempts = 1

    # Build temperature schedule for nucleus sampling
    temp_schedule = _make_temp_schedule(effective_attempts, temp_min, temp_max)

    print(f"Decoding method  : {decoding_method}")
    print(f"Attempts/expr    : {effective_attempts}")
    if decoding_method in ('beam', 'nucleus'):
        print(f"Beam size        : {beam_size}")

    # ------------------------------------------------------------------
    # 4.  Pre-load kinematics modules and generate fixed phase-space points
    # ------------------------------------------------------------------
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8)

    _data_gen_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data_generation'
    )
    _km_mod = _local_import("kinematics", _data_gen_dir)
    _gd_mod = _local_import("gen_data",   _data_gen_dir)

    _NUM_EQUIV_SAMPLES = 3
    _NUM_EQUIV_SEED    = 42
    _NUM_EQUIV_M       = 2.0

    _precomputed_kinematics = [
        _km_mod.generate_kinematics(N_particles, M=_NUM_EQUIV_M, seed=_NUM_EQUIV_SEED + i)
        for i in range(_NUM_EQUIV_SAMPLES)
    ]
    print(
        f"\nPre-generated {_NUM_EQUIV_SAMPLES} kinematic phase-space points "
        f"(N={N_particles}, M={_NUM_EQUIV_M}, seed={_NUM_EQUIV_SEED})."
    )

    def _num_equiv(a_tokens, b_tokens, tol_abs=1e-12, tol_rel=1e-10) -> bool:
        """Check numerical equivalence of two token sequences."""
        expr_a = tokenizer.decode_infix(list(a_tokens))
        expr_b = tokenizer.decode_infix(list(b_tokens))
        for mom, pol in _precomputed_kinematics:
            val_a = _gd_mod.eval_infix_numeric(expr_a, mom, pol)
            val_b = _gd_mod.eval_infix_numeric(expr_b, mom, pol)
            diff  = abs(val_a - val_b)
            scale = max(abs(val_a), abs(val_b), 1.0)
            if not (diff <= tol_abs or diff / scale <= tol_rel):
                return False
        return True

    # ------------------------------------------------------------------
    # 5.  Load input CSV
    # ------------------------------------------------------------------
    data_dir = os.path.join(_repo_root, 'data')

    if not os.path.isabs(csv_file):
        csv_path = os.path.join(data_dir, csv_file)
    else:
        csv_path = csv_file

    df = pd.read_csv(csv_path)
    if input_column not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Column '{input_column}' not found in {csv_path}. "
            f"Available columns: {available}"
        )

    if max_datasize is not None:
        df = df.head(max_datasize)

    print(f"\nLoaded {len(df)} expressions from {csv_path} (column: '{input_column}').")

    # ------------------------------------------------------------------
    # 6.  Prepare output CSV
    # ------------------------------------------------------------------
    if not os.path.isabs(output_file):
        out_path = os.path.join(data_dir, output_file)
    else:
        out_path = output_file

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_fh   = open(out_path, 'w', newline='')
    writer   = csv.writer(out_fh)
    writer.writerow(['original', 'simplification'])

    # ------------------------------------------------------------------
    # 7.  Main simplification loop
    # ------------------------------------------------------------------
    BOS, EOS, PAD = 2, 3, 0

    simplified_count = 0
    failed_count     = 0

    for idx, row in df.iterrows():
        raw_input = ast.literal_eval(row[input_column])   # list of ints (no BOS/EOS)

        # Sequence with EOS appended – used for numerical equivalence comparisons
        input_seq_eos = raw_input + [EOS]
        input_content_len = len(raw_input)   # token count excluding BOS/EOS

        print(f"\n[{idx + 1}/{len(df)}] Original ({input_content_len} tokens): {raw_input}")

        # Build model input tensor: [BOS] + content + [EOS], padded to a batch of 1
        src_tokens = [BOS] + raw_input + [EOS]
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)

        simplification_found = None
        simplification_tokens = None

        # ------------------------------------------------------------------
        # Greedy pre-check (nucleus only): one deterministic decode first.
        # If it already yields a shorter equivalent, skip stochastic search.
        # ------------------------------------------------------------------
        if decoding_method == 'nucleus':
            with torch.no_grad():
                max_gen_len = min(len(src_tokens) * 2, model.max_seq_len - 1)
                gen_g, _ = decode_with_model(
                    model, src_tensor,
                    max_length          = max_gen_len,
                    decoding_method     = 'greedy',
                    beam_size           = beam_size,
                    p_nucleus           = p_nucleus,
                    temperature_nucleus = temperature_nucleus,
                    bos_token = BOS, eos_token = EOS, pad_token = PAD,
                )
            greedy_seq = clean_seq(gen_g.cpu().numpy()[0], pad_token=PAD, eos_token=EOS)
            greedy_content = greedy_seq[:-1] if greedy_seq and greedy_seq[-1] == EOS else greedy_seq
            if len(greedy_content) < input_content_len:
                try:
                    if _num_equiv(input_seq_eos, greedy_seq):
                        simplification_found  = greedy_content
                        simplification_tokens = greedy_seq
                        print(f"  ✓ Greedy pre-check succeeded ({len(greedy_content)} tokens).")
                except Exception:
                    pass

        for attempt in range(effective_attempts):
            if simplification_found is not None:
                break
            # Temperature for this attempt (only relevant for nucleus)
            t_this = temp_schedule[attempt] if decoding_method == 'nucleus' else temperature_nucleus

            if effective_attempts > 1:
                print(f"  Attempt {attempt + 1}/{effective_attempts}"
                      + (f" (temperature={t_this:.2f})" if decoding_method == 'nucleus' else ""))

            with torch.no_grad():
                max_gen_len = min(len(src_tokens) * 2, model.max_seq_len - 1)
                gen, beams = decode_with_model(
                    model, src_tensor,
                    max_length        = max_gen_len,
                    decoding_method   = decoding_method,
                    beam_size         = beam_size,
                    p_nucleus         = p_nucleus,
                    temperature_nucleus = t_this,
                    bos_token = BOS, eos_token = EOS, pad_token = PAD,
                )

            gen_np = gen.cpu().numpy()

            # Collect all candidate sequences for this attempt
            # (greedy → 1 candidate; beam/nucleus → best + all beams)
            best_seq   = clean_seq(gen_np[0], pad_token=PAD, eos_token=EOS)
            candidates = [best_seq]
            if beams is not None and len(beams) > 0:
                for hyp in beams[0]:
                    c = clean_seq(hyp, pad_token=PAD, eos_token=EOS)
                    if c not in candidates:
                        candidates.append(c)

            for cand_seq in candidates:
                # Content tokens = all except trailing EOS (if present)
                cand_content = cand_seq[:-1] if cand_seq and cand_seq[-1] == EOS else cand_seq
                cand_content_len = len(cand_content)

                if cand_content_len >= input_content_len:
                    continue   # not shorter – skip

                # Shorter candidate: verify numerical equivalence
                try:
                    if _num_equiv(input_seq_eos, cand_seq):
                        simplification_found  = cand_content
                        simplification_tokens = cand_seq
                        break
                except Exception as e:
                    # Malformed / unevaluable output – skip silently
                    pass

            if simplification_found is not None:
                break   # stop attempting for this expression

        # ------------------------------------------------------------------
        # Report and record result
        # ------------------------------------------------------------------
        if simplification_found is not None:
            simplified_count += 1
            print(f"  ✓ Simplified ({len(simplification_found)} tokens): {simplification_found}")
            writer.writerow([str(raw_input), str(simplification_found)])
        else:
            failed_count += 1
            print(f"  ✗ No simplification found.")
            writer.writerow([str(raw_input), 'no simplification found'])

        out_fh.flush()   # write each result immediately in case of early termination

    # ------------------------------------------------------------------
    # 8.  Summary
    # ------------------------------------------------------------------
    out_fh.close()
    total = simplified_count + failed_count
    print(f"\n{'='*60}")
    print(f"SIMPLIFICATION SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total expressions  : {total}")
    print(f"Simplified         : {simplified_count}  ({100.0 * simplified_count / total:.1f}%)" if total else "")
    print(f"Not simplified     : {failed_count}  ({100.0 * failed_count / total:.1f}%)" if total else "")
    print(f"Results saved to   : {out_path}")


if __name__ == "__main__":
    main()
