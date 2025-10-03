import os
import sys
import torch
import optuna
# Detect and set number of CPU threads for PyTorch
n_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(n_threads)
print(f"Using {n_threads} CPU threads for PyTorch.")
from transformer_functions import TransformerRegressor, load_transformer_model, decode_with_model, clean_seq

# Add data_generation to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_generation'))
from Tokenizer import ScatteringAmplitudeTokenizer, numerically_equivalent

def evaluate_decoding(model, device, data_loader, tokenizer, N_particles, 
                      decoding_method='greedy', beam_size=5, p_nucleus=0.9, 
                      temperature_nucleus=1.0, beam_match_any=True):
    """
    Evaluate model with given decoding hyperparameters.
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    total = 0
    exact_correct = 0
    numerical_correct = 0
    
    for batch in data_loader:
        src = batch['input'].to(device)
        tgt = batch['target'].to(device)
        
        # Generate predictions
        with torch.no_grad():
            decode_len = tgt.size(1) * 2  # Allow sequences to be up to 2x target length
            gen, beams = decode_with_model(
                model, src, max_length=decode_len, 
                decoding_method=decoding_method,
                beam_size=beam_size,
                p_nucleus=p_nucleus,
                temperature_nucleus=temperature_nucleus,
                bos_token=2, eos_token=3, pad_token=0
            )
        
        gen = gen.cpu().numpy()
        tgt = tgt.cpu().numpy()
        
        # Process each example in the batch
        for i in range(src.size(0)):
            tgt_seq = clean_seq(tgt[i], pad_token=0, eos_token=3)
            gen_seq = clean_seq(gen[i], pad_token=0, eos_token=3)
            
            # 1. Exact sequence match
            is_exact_match = False
            if beams is None:
                # Greedy: compare best sequence only
                is_exact_match = (tgt_seq == gen_seq)
            else:
                # Beam/nucleus: either any-beam match or best-only match
                if not beam_match_any:
                    is_exact_match = (tgt_seq == gen_seq)
                else:
                    beam_list = beams[i] if i < len(beams) else []
                    if len(beam_list) == 0:
                        is_exact_match = (tgt_seq == gen_seq)
                    else:
                        for hyp_seq in beam_list:
                            if clean_seq(hyp_seq, pad_token=0, eos_token=3) == tgt_seq:
                                is_exact_match = True
                                break
            
            if is_exact_match:
                exact_correct += 1
            
            # 2. Numerical equivalence check
            is_numerical_match = False
            
            # First, if sequences are exactly the same, they're numerically equivalent
            if is_exact_match:
                is_numerical_match = True
            else:
                # Check numerical equivalence
                if beams is None or not beam_match_any:
                    # Greedy or best-only beam: check only the best prediction
                    try:
                        is_numerical_match = numerically_equivalent(
                            tokenizer, tgt_seq, gen_seq, N_particles, 
                            samples=3, M=2.0, seed=42, return_details=False
                        )
                    except Exception:
                        # If evaluation fails, not numerically equivalent
                        pass
                else:
                    # Beam search with any-beam matching: check all beam hypotheses
                    beam_list = beams[i] if i < len(beams) else []
                    if len(beam_list) == 0:
                        try:
                            is_numerical_match = numerically_equivalent(
                                tokenizer, tgt_seq, gen_seq, N_particles, 
                                samples=3, M=2.0, seed=42, return_details=False
                            )
                        except Exception:
                            pass
                    else:
                        # Check each beam hypothesis for numerical equivalence
                        for hyp_seq in beam_list:
                            try:
                                clean_hyp = clean_seq(hyp_seq, pad_token=0, eos_token=3)
                                if numerically_equivalent(
                                    tokenizer, tgt_seq, clean_hyp, N_particles, 
                                    samples=3, M=2.0, seed=42, return_details=False
                                ):
                                    is_numerical_match = True
                                    break
                            except Exception:
                                continue
            
            if is_numerical_match:
                numerical_correct += 1
            
            total += 1
    
    exact_acc = 100.0 * exact_correct / total if total > 0 else 0.0
    numerical_acc = 100.0 * numerical_correct / total if total > 0 else 0.0
    
    return {
        'exact_accuracy': exact_acc,
        'numerical_accuracy': numerical_acc,
        'total': total
    }


def objective(trial, decoding_method, model, device, data_loader, tokenizer, N_particles, beam_match_any):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        decoding_method: 'greedy', 'beam', or 'nucleus'
        model: Pre-trained transformer model
        device: Torch device
        data_loader: DataLoader for evaluation
        tokenizer: ScatteringAmplitudeTokenizer instance
        N_particles: Number of particles
        beam_match_any: Whether to match any beam hypothesis (fixed parameter)
        
    Returns:
        float: Negative numerical accuracy (to minimize)
    """
    
    if decoding_method == 'greedy':
        # No hyperparameters to tune for greedy decoding
        results = evaluate_decoding(
            model, device, data_loader, tokenizer, N_particles,
            decoding_method='greedy'
        )
    elif decoding_method == 'beam':
        # Suggest beam search hyperparameters
        beam_size_exp = trial.suggest_int('beam_size_exp', 1, 4)  # 2^1 to 2^4: 2, 4, 8, 16
        beam_size = 2 ** beam_size_exp
        
        results = evaluate_decoding(
            model, device, data_loader, tokenizer, N_particles,
            decoding_method='beam',
            beam_size=beam_size,
            beam_match_any=beam_match_any
        )
    elif decoding_method == 'nucleus':
        # Suggest nucleus sampling hyperparameters
        beam_size_exp = trial.suggest_int('beam_size_exp', 1, 4)  # 2^1 to 2^4: 2, 4, 8, 16
        beam_size = 2 ** beam_size_exp
        p_nucleus = trial.suggest_float('p_nucleus', 0.8, 0.99)
        temperature_nucleus = trial.suggest_float('temperature_nucleus', 0.5, 2.0)
        
        results = evaluate_decoding(
            model, device, data_loader, tokenizer, N_particles,
            decoding_method='nucleus',
            beam_size=beam_size,
            p_nucleus=p_nucleus,
            temperature_nucleus=temperature_nucleus,
            beam_match_any=beam_match_any
        )
    else:
        raise ValueError(f"Unknown decoding method: {decoding_method}")
    
    # Report intermediate values to Optuna
    trial.set_user_attr('exact_accuracy', results['exact_accuracy'])
    trial.set_user_attr('numerical_accuracy', results['numerical_accuracy'])
    
    # Return negative accuracy to minimize (Optuna minimizes by default)
    # We optimize for exact accuracy as the primary metric
    return -results['exact_accuracy']


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python optuna_decoding_optim.py <decoding_method> [N_particles] [n_trials] [max_datasize] [beam_match_any] [n_startup_trials]")
        print("  decoding_method: 'greedy', 'beam', or 'nucleus'")
        print("  N_particles: Number of particles (default: 4)")
        print("  n_trials: Number of Optuna trials (default: 50)")
        print("  max_datasize: Max examples to evaluate per trial (default: 1000)")
        print("  beam_match_any: Whether to match any beam hypothesis - 'true' or 'false' (default: true)")
        print("  n_startup_trials: Number of random trials before TPE (default: 20)")
        sys.exit(1)
    
    decoding_method = sys.argv[1]
    N_particles = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    max_datasize = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    beam_match_any_str = sys.argv[5].lower() if len(sys.argv) > 5 else 'true'
    beam_match_any = beam_match_any_str in ['true', '1', 'yes', 't', 'y']
    n_startup_trials = int(sys.argv[6]) if len(sys.argv) > 6 else min(20, n_trials // 3)
    force_cpu = False  # Conservative default for compatibility
    
    if decoding_method not in ['greedy', 'beam', 'nucleus']:
        print(f"Error: decoding_method must be 'greedy', 'beam', or 'nucleus', got '{decoding_method}'")
        sys.exit(1)
    
    print(f"=== Optuna Decoding Hyperparameter Optimization ===")
    print(f"Decoding method: {decoding_method}")
    print(f"N_particles: {N_particles}")
    print(f"Number of trials: {n_trials}")
    print(f"Max examples per trial: {max_datasize}")
    print(f"Beam match any: {beam_match_any}")
    print()
    
    # Check if there's anything to optimize
    if decoding_method == 'greedy':
        print("No hyperparameters to optimise in mode 'greedy'.")
        sys.exit(0)
    
    # Ensure n_trials is greater than n_startup_trials
    assert n_trials > n_startup_trials, f"n_trials ({n_trials}) must be greater than n_startup_trials ({n_startup_trials})"
    
    # Load model
    model_path = os.path.join('models', f'model_{N_particles}pt.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Determine device
    if force_cpu:
        device = 'cpu'
        print("Using CPU device")
    else:
        try:
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1).cuda()
                device = 'cuda'
                print("Using CUDA device")
            else:
                device = 'cpu'
                print("Using CPU device")
        except (RuntimeError, AssertionError):
            device = 'cpu'
            print("Using CPU device")
    
    loaded = load_transformer_model(TransformerRegressor, model_path, device=device)
    model = loaded['model']
    model.to(device)
    model.device = device
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # Load test dataset
    csv_file = os.path.join('data/eval_tuning_set', f'gi_{N_particles}pt_tok.csv')
    if not os.path.exists(csv_file):
        print(f"Error: Test dataset not found at {csv_file}")
        sys.exit(1)
    
    from data_import import TransformerDataset
    from torch.utils.data import DataLoader, Subset
    
    dataset = TransformerDataset(csv_file, max_length=None)
    
    # Limit dataset size for faster optimization
    if max_datasize is not None and max_datasize < len(dataset):
        dataset = Subset(dataset, range(max_datasize))
        print(f"Limited dataset to {max_datasize} examples for optimization")
    else:
        print(f"Using {len(dataset)} examples for optimization")
    
    batch_size = min(64, len(dataset))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize tokenizer
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8)
    
    # Create Optuna study
    # TPE sampler with explicit n_startup_trials
    study_name = f"decoding_{decoding_method}_{N_particles}pt"
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=study_name
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, decoding_method, model, device, data_loader, tokenizer, N_particles, beam_match_any),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results in style similar to optuna_bayes_optim.py
    print('Best trial:')
    trial = study.best_trial
    print(f'  Trial Number: {trial.number}')
    print(f'  Exact Accuracy: {-trial.value:.2f}%')
    print(f'  Numerical Accuracy: {trial.user_attrs.get("numerical_accuracy", "N/A"):.2f}%')
    print('  Best Hyperparameters:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')


if __name__ == "__main__":
    main()
