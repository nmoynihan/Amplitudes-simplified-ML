import os
import sys
import torch

# Set CUDA memory allocator to reduce fragmentation (helps with OOM issues)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Detect and set number of CPU threads for PyTorch
n_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(n_threads)
print(f"Using {n_threads} CPU threads for PyTorch.")
from transformer_functions import TransformerRegressor, load_transformer_model, decode_with_model, clean_seq

# Add data_generation to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_generation'))
from Tokenizer import ScatteringAmplitudeTokenizer, numerically_equivalent

# Settings
N_particles = 5 
model_path = os.path.join('models', 'default_run/best_model.pt')  # Path to the trained model
csv_file = ['expanded_data/test_data/gi_5pt_tok_python.csv', 'expanded_data/test_data/gi_5pt_tok_mathematica.csv'] 
#csv_file = f'relabM_alt_{N_particles}pt_tok.csv'  # Paths are relative to data/ directory
#csv_file = f'test_set/gi_{N_particles}pt_tok.csv'  # Path to the test dataset
#csv_file = f'_old/ampl00111_tok.csv' # Path the Feynman rules data
batch_size = 16  # Will be auto-adjusted for smaller GPUs (< 60GB will use batch_size=32)
max_datasize = None  # Max number of examples to evaluate (None = use whole file)
num_print = 2  # Number of examples to print
inference_only = False  # Set to True for pure inference (ignore simple column), False for evaluation
force_cpu = False # Force CPU usage (set to True to avoid CUDA/MPS device issues, good for local testing)
use_mps = False # Device toggle: enable MPS explicitly (default False due to missing ops in PyTorch Transformer on MPS)

# Decoding hyperparameters (set here for evaluation)
decoding_method = 'greedy'  # Use greedy for deterministic, teacher-forcing-like behavior
max_length = None           # Length limit for generation (None = no limit)
beam_size = 4              # Number of beams for beam/nucleus search
p_nucleus = 0.99            # Nucleus cutoff probability (lower => more diversity)
temperature_nucleus = 1.96   # Lower temperature for more deterministic output
# For beam/nucleus evaluation: if True, count as correct if ANY beam hypothesis matches target; if False, only best hyp
beam_match_any = True

def main():
    # Resolve device and load model on it (prefer CUDA, then optional MPS, else CPU)
    use_data_parallel = False
    num_gpus = 0
    effective_batch_size = batch_size  # Local copy that we can modify
    
    if force_cpu:
        preferred_device = 'cpu'
        print("Forcing CPU device usage")
    else:
        try:
            if torch.cuda.is_available():
                # Test if CUDA actually works by creating a small tensor
                test_tensor = torch.zeros(1).cuda()
                preferred_device = 'cuda'
                num_gpus = torch.cuda.device_count()
                print(f"CUDA device available and working")
                print(f"Found {num_gpus} GPU(s):")
                
                # Get minimum GPU memory across all GPUs
                min_gpu_memory = float('inf')
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    gpu_memory_gb = props.total_memory / (1024**3)
                    min_gpu_memory = min(min_gpu_memory, gpu_memory_gb)
                    print(f"  GPU {i}: {props.name} ({gpu_memory_gb:.1f} GB)")
                    print(f"    - CUDA Capability: {props.major}.{props.minor}")
                    print(f"    - Multi-Processors: {props.multi_processor_count}")
                    print(f"    - Max threads per Multi-Processor: {props.max_threads_per_multi_processor}")
                    total_parallel = props.multi_processor_count * props.max_threads_per_multi_processor
                    print(f"    - Total parallel threads: {total_parallel:,}")
                
                # Auto-adjust batch size based on GPU memory and decoding method
                if min_gpu_memory < 60:  # Less than 60GB (e.g., A100 40GB)
                    if decoding_method in ['beam', 'nucleus']:
                        # Beam/nucleus uses beam_size * batch_size memory, so reduce more
                        effective_batch_size = 8  # Reduced from 16 for better memory safety
                        print(f"\n⚠️  Detected GPUs with {min_gpu_memory:.1f} GB memory (< 60 GB)")
                        print(f"   Using {decoding_method} decoding with beam_size={beam_size}")
                        print(f"   Auto-adjusting batch_size: {batch_size} → {effective_batch_size} (accounts for {beam_size}x beam expansion)")
                    else:
                        effective_batch_size = 16  # Reduced from 32 for greedy decoding
                        print(f"\n⚠️  Detected GPUs with {min_gpu_memory:.1f} GB memory (< 60 GB)")
                        print(f"   Auto-adjusting batch_size: {batch_size} → {effective_batch_size}")
                
                # Autoregressive generation is not thread-safe for multi-GPU parallel processing
                # We'll use sequential processing on multiple GPUs for all decoding methods
                if num_gpus > 1:
                    use_data_parallel = True
                    print(f"\nWill distribute batches across {num_gpus} GPUs (sequential processing per GPU)")
                    print(f"Each GPU will process ~{effective_batch_size // num_gpus} examples per batch")
                    if decoding_method in ['beam', 'nucleus']:
                        print(f"Note: Using {decoding_method} decoding with beam_size={beam_size}")
            else:
                raise RuntimeError("CUDA not available")
        except (RuntimeError, AssertionError) as e:
            print(f"CUDA not working: {e}")
            try:
                if use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Test if MPS actually works
                    test_tensor = torch.zeros(1).to('mps')
                    preferred_device = 'mps'
                    print("MPS device available and working")
                else:
                    raise RuntimeError("MPS not available or disabled")
            except (RuntimeError, AssertionError) as e:
                print(f"MPS not working: {e}")
                preferred_device = 'cpu'
                print("Using CPU device")
    
    loaded = load_transformer_model(TransformerRegressor, model_path, device=preferred_device)
    model = loaded['model']
    
    # For multi-GPU, we'll create separate model instances on each GPU
    # This is cleaner for autoregressive generation than DataParallel
    models_per_gpu = []
    if use_data_parallel:
        print(f"\nCreating model replicas for each GPU...")
        for gpu_id in range(num_gpus):
            # Clone model to each GPU
            if gpu_id == 0:
                # First GPU uses the already loaded model
                model.to(f'cuda:{gpu_id}')
                model.device = f'cuda:{gpu_id}'
                model.eval()
                models_per_gpu.append(model)
            else:
                # Load separate copy for other GPUs
                loaded_gpu = load_transformer_model(TransformerRegressor, model_path, device=f'cuda:{gpu_id}')
                model_gpu = loaded_gpu['model']
                model_gpu.to(f'cuda:{gpu_id}')
                model_gpu.device = f'cuda:{gpu_id}'
                model_gpu.eval()
                models_per_gpu.append(model_gpu)
        print(f"Model replicas created on GPUs: {list(range(num_gpus))}")
        print(f"Strategy: Each batch will be split across GPUs, processed sequentially")
        print(f"  Note: Autoregressive generation is not thread-safe, so we process")
        print(f"  chunks sequentially on different GPUs to utilize all hardware.")
        print(f"  Example: batch_size={effective_batch_size} with {num_gpus} GPUs")
        base = effective_batch_size // num_gpus
        rem = effective_batch_size % num_gpus
        splits = [base + (1 if i < rem else 0) for i in range(num_gpus)]
        for i, s in enumerate(splits):
            print(f"    GPU {i}: {s} examples per batch")
        device = 'cuda'
    else:
        model.to(preferred_device)
        model.device = preferred_device
        model.eval()
        device = preferred_device
    
    print(f"Running on device: {device}")
    print(f"Decoding method: {decoding_method}")
    
    if inference_only:
        print("=== INFERENCE MODE ===")
        if decoding_method in ['beam', 'nucleus']:
            print(f"Beam size: {beam_size}")
        print()
    else:
        print("=== EVALUATION MODE ===")
        if decoding_method in ['beam', 'nucleus']:
            print(f"Beam size: {beam_size}")
            print(f"Beam match any: {beam_match_any}")

    # Initialize tokenizer for numerical equivalence checking
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8)
    
    # Load full test dataset for both inference and evaluation modes
    from data_import import TransformerDataset
    from torch.utils.data import DataLoader, Subset
    
    # Normalize paths - handle both single file and list of files
    data_dir = os.path.join(os.getcwd(), 'data')
    
    if isinstance(csv_file, list):
        # Multiple files: normalize each path
        normalized_files = []
        for file in csv_file:
            if not os.path.isabs(file):
                full_path = os.path.join(data_dir, file)
            else:
                full_path = file
            normalized_files.append(full_path)
        csv_file_normalized = normalized_files
    else:
        # Single file: normalize path
        if not os.path.isabs(csv_file):
            csv_file_normalized = os.path.join(data_dir, csv_file)
        else:
            csv_file_normalized = csv_file
    
    dataset = TransformerDataset(csv_file_normalized, max_length=None)
    
    # Apply max_datasize limit if specified
    if max_datasize is not None and max_datasize < len(dataset):
        dataset = Subset(dataset, range(max_datasize))
        print(f"Limited dataset to {max_datasize} examples")
    
    data_loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False)
    print(f"Using full test dataset: {len(dataset)} examples")
    
    # Print memory info before starting
    if preferred_device == 'cuda':
        print(f"\n{'='*70}")
        print(f"INITIAL MEMORY STATUS")
        for i in range(num_gpus if num_gpus > 0 else 1):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024**3)
            free = total - allocated
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free, {total:.2f} GB total")
        print(f"{'='*70}")

    # Ensure all models are in evaluation mode
    if use_data_parallel:
        for m in models_per_gpu:
            m.eval()
    else:
        model.eval()

    # Tracking metrics for evaluation mode
    total = 0
    token_total = 0    # Total tokens
    exact_correct = 0  # Exact sequence matches
    token_correct = 0  # Total matching tokens
    numerical_correct = 0  # Numerically equivalent expressions
    malformed_count = 0  # Expressions that fail to detokenize properly
    printed = 0
    
    # Progress tracking
    total_batches = len(data_loader) if hasattr(data_loader, '__len__') else None
    batch_count = 0
    
    # Initial GPU memory check
    if preferred_device == 'cuda' and use_data_parallel:
        print(f"\n{'='*60}")
        print(f"Initial GPU Memory Status:")
        print(f"{'='*60}")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"  GPU {i}: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB")
        print(f"{'='*60}\n")
    
    for batch in data_loader:
        batch_count += 1
        
        # Progress reporting for evaluation mode
        if not inference_only:
            if total_batches:
                print(f"Processing batch {batch_count}/{total_batches} ({100.0 * batch_count / total_batches:.1f}%)")
            else:
                print(f"Processing batch {batch_count}...")
            sys.stdout.flush()  # Force output to display immediately
        
        # Monitor GPU usage every 10 batches
        if preferred_device == 'cuda' and use_data_parallel and batch_count % 10 == 0:
            print(f"\n  [Batch {batch_count}] GPU Memory Usage:")
            for i in range(num_gpus):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"    GPU {i}: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB")
            print()
            sys.stdout.flush()
        
        # Generate predictions
        with torch.no_grad():
            if use_data_parallel:
                # Multi-GPU: Split batch across GPUs and process SEQUENTIALLY
                # Note: Autoregressive generation is not thread-safe with CUDA due to
                # dynamic tensor creation in loops. We process chunks sequentially but
                # on different GPUs to utilize all available hardware.
                
                src_full = batch['input']
                tgt_full = batch['target']
                batch_size_actual = src_full.size(0)
                decode_len = tgt_full.size(1) * 2
                # Cap at model's max positional encoding length
                # Use -1 to leave room for generation loop edge cases
                decode_len = min(decode_len, model.max_seq_len - 1)
                
                # Calculate split sizes for each GPU
                base_size = batch_size_actual // num_gpus
                remainder = batch_size_actual % num_gpus
                split_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_gpus)]
                
                # Split batch
                src_splits = torch.split(src_full, split_sizes, dim=0)
                tgt_splits = torch.split(tgt_full, split_sizes, dim=0)
                
                # Storage for results from each GPU
                results = []
                beams_results = []
                
                # Process each chunk sequentially on its designated GPU
                for gpu_id in range(num_gpus):
                    if split_sizes[gpu_id] > 0:  # Only process if this GPU has data
                        src_chunk = src_splits[gpu_id]
                        src_gpu = src_chunk.to(f'cuda:{gpu_id}')
                        model_gpu = models_per_gpu[gpu_id]
                        
                        # Set CUDA device context for this GPU
                        with torch.cuda.device(f'cuda:{gpu_id}'):
                            gen_gpu, beams_gpu = decode_with_model(
                                model_gpu, src_gpu, max_length=decode_len,
                                decoding_method=decoding_method,
                                beam_size=beam_size,
                                p_nucleus=p_nucleus,
                                temperature_nucleus=temperature_nucleus,
                                bos_token=2, eos_token=3, pad_token=0
                            )
                        
                        results.append(gen_gpu.cpu())
                        beams_results.append(beams_gpu)
                        
                        # Clean up GPU tensors
                        del src_gpu, gen_gpu
                        torch.cuda.empty_cache()
                
                # Combine results from all GPUs
                if len(results) == 0:
                    raise RuntimeError("No valid results from any GPU")
                gen = torch.cat(results, dim=0)
                
                # Combine beams if they exist
                if all(b is not None for b in beams_results):
                    beams = []
                    for beam_list in beams_results:
                        if beam_list is not None:
                            beams.extend(beam_list)
                else:
                    beams = None
                
                src = src_full.to(device)
                tgt = tgt_full.to(device)
                
            else:
                # Single GPU: standard processing
                src = batch['input'].to(device)
                tgt = batch['target'].to(device)
                decode_len = tgt.size(1) * 2
                # Cap at model's max positional encoding length
                # Use -1 to leave room for generation loop edge cases
                decode_len = min(decode_len, model.max_seq_len - 1)
                
                gen, beams = decode_with_model(
                    model, src, max_length=decode_len,
                    decoding_method=decoding_method,
                    beam_size=beam_size,
                    p_nucleus=p_nucleus,
                    temperature_nucleus=temperature_nucleus,
                    bos_token=2, eos_token=3, pad_token=0
                )
        

        
        # Clear cache more aggressively to prevent fragmentation and OOM
        if preferred_device == 'cuda':
            # Clear every 5 batches instead of 10 for better memory management
            if batch_count % 5 == 0:
                if use_data_parallel:
                    for i in range(num_gpus):
                        torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
                
                # Print memory status periodically
                if batch_count % 20 == 0:
                    for i in range(num_gpus if num_gpus > 0 else 1):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        free = total - allocated
                        print(f"    GPU {i} memory: {allocated:.2f}/{total:.2f} GB ({free:.2f} GB free)")
        
        gen = gen.cpu().numpy()
        tgt = tgt.cpu().numpy()
        
        # Process each example in the batch
        for i in range(src.size(0)):
            if inference_only:
                # INFERENCE MODE: Just print predictions with numerical equivalence checks
                src_seq = src[i].cpu().numpy().tolist()
                gen_seq = clean_seq(gen[i], pad_token=0, eos_token=3)
                
                print(f"Input:     {src_seq}")
                if decoding_method == 'greedy':
                    # Check numerical equivalence between input and output
                    try:
                        is_num_equiv = numerically_equivalent(
                            tokenizer, src_seq, gen_seq, N_particles,
                            samples=3, M=2.0, seed=42, return_details=False
                        )
                        print(f"Predicted: {gen_seq}")
                        print(f"  Numerically equivalent to input: {is_num_equiv}")
                    except Exception as e:
                        print(f"Predicted: {gen_seq}")
                        print(f"  Numerically equivalent to input: Error - {e}")
                else:  # beam or nucleus
                    print(f"Best prediction: {gen_seq}")
                    # Check numerical equivalence for best prediction
                    try:
                        is_num_equiv = numerically_equivalent(
                            tokenizer, src_seq, gen_seq, N_particles,
                            samples=3, M=2.0, seed=42, return_details=False
                        )
                        print(f"  Numerically equivalent to input: {is_num_equiv}")
                    except Exception as e:
                        print(f"  Numerically equivalent to input: Error - {e}")
                    
                    if beams and i < len(beams):
                        print(f"All {beam_size} beam hypotheses:")
                        for j, hyp_seq in enumerate(beams[i][:beam_size]):
                            clean_hyp = clean_seq(hyp_seq, pad_token=0, eos_token=3)
                            # Check numerical equivalence for each beam
                            try:
                                is_num_equiv = numerically_equivalent(
                                    tokenizer, src_seq, clean_hyp, N_particles,
                                    samples=3, M=2.0, seed=42, return_details=False
                                )
                                print(f"  Beam {j+1}: {clean_hyp}")
                                print(f"    Numerically equivalent to input: {is_num_equiv}")
                            except Exception as e:
                                print(f"  Beam {j+1}: {clean_hyp}")
                                print(f"    Numerically equivalent to input: Error - {e}")
                print()
                total += 1
                
            else:
                # EVALUATION MODE: Compare with targets
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
                
                # 2. Token-level accuracy (for best prediction only)
                min_len = min(len(tgt_seq), len(gen_seq))
                matching_tokens = sum(1 for j in range(min_len) if tgt_seq[j] == gen_seq[j])
                token_correct += matching_tokens
                token_total += max(len(tgt_seq), len(gen_seq))
                
                # 3. Numerical equivalence check
                is_numerical_match = False
                is_malformed = False
                
                # First, if sequences are exactly the same, they're numerically equivalent
                if is_exact_match:
                    is_numerical_match = True
                else:
                    # Check numerical equivalence
                    if beams is None or not beam_match_any:
                        # Greedy or best-only beam: check only the best prediction
                        try:
                            # Try to decode both sequences to infix expressions
                            tgt_infix = tokenizer.decode_infix(tgt_seq)
                            gen_infix = tokenizer.decode_infix(gen_seq)
                            
                            # Check numerical equivalence                       
                            is_numerical_match = numerically_equivalent(
                                tokenizer, tgt_seq, gen_seq, N_particles, 
                                samples=3, M=2.0, seed=42, return_details=False
                            )
                            
                        except Exception as e:
                            # If detokenization or numerical evaluation fails, mark as malformed
                            is_malformed = True
                            if printed < num_print:
                                print(f"Warning: Failed to evaluate numerical equivalence: {e}")
                                print(f"  Target sequence: {tgt_seq}")
                                print(f"  Generated sequence: {gen_seq}")
                                try:
                                    tgt_infix = tokenizer.decode_infix(tgt_seq)
                                    print(f"  Target infix: {tgt_infix}")
                                except Exception as e2:
                                    print(f"  Target decode failed: {e2}")
                                try:
                                    gen_infix = tokenizer.decode_infix(gen_seq)
                                    print(f"  Generated infix: {gen_infix}")
                                except Exception as e3:
                                    print(f"  Generated decode failed: {e3}")
                                print()
                    else:
                        # Beam search with any-beam matching: check all beam hypotheses
                        beam_list = beams[i] if i < len(beams) else []
                        if len(beam_list) == 0:
                            # Fallback to best prediction if no beams available
                            try:
                                tgt_infix = tokenizer.decode_infix(tgt_seq)
                                gen_infix = tokenizer.decode_infix(gen_seq)
                                is_numerical_match = numerically_equivalent(
                                    tokenizer, tgt_seq, gen_seq, N_particles, 
                                    samples=3, M=2.0, seed=42, return_details=False
                                )
                            except Exception as e:
                                is_malformed = True
                        else:
                            # Check each beam hypothesis for numerical equivalence
                            for hyp_seq in beam_list:
                                try:
                                    clean_hyp = clean_seq(hyp_seq, pad_token=0, eos_token=3)
                                    tgt_infix = tokenizer.decode_infix(tgt_seq)
                                    hyp_infix = tokenizer.decode_infix(clean_hyp)
                                    
                                    if numerically_equivalent(
                                        tokenizer, tgt_seq, clean_hyp, N_particles, 
                                        samples=3, M=2.0, seed=42, return_details=False
                                    ):
                                        is_numerical_match = True
                                        break
                                except Exception as e:
                                    # Continue to next beam if this one fails
                                    continue
                            
                            # If no beam matched and we had decoding errors, mark as malformed
                            if not is_numerical_match:
                                try:
                                    # Test if we can at least decode the target and best prediction
                                    tgt_infix = tokenizer.decode_infix(tgt_seq)
                                    gen_infix = tokenizer.decode_infix(gen_seq)
                                except Exception as e:
                                    is_malformed = True
                                    if printed < num_print:
                                        print(f"Warning: Failed to evaluate numerical equivalence: {e}")
                                        print(f"  Target sequence: {tgt_seq}")
                                        print(f"  Generated sequence: {gen_seq}")
                                        print()
                
                if is_numerical_match:
                    numerical_correct += 1
                
                # Only count as malformed if we can't decode AND it's not an exact match
                # (exact matches can't be malformed since target sequences are valid)
                if is_malformed and not is_exact_match:
                    malformed_count += 1
                
                total += 1
                
                if printed < num_print:
                    print(f"Input:      {src[i].cpu().numpy().tolist()}")
                    print(f"Target:     {tgt_seq}")
                    print(f"Generated:  {gen_seq}")
                    print(f"Exact match: {is_exact_match}")
                    print(f"Numerical match: {is_numerical_match}")
                    print(f"Malformed: {is_malformed}")
                    print()
                    printed += 1

    # Final GPU memory report
    if preferred_device == 'cuda' and use_data_parallel:
        print(f"\n{'='*60}")
        print(f"Final GPU Memory Statistics:")
        print(f"{'='*60}")
        total_peak = 0
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
            total_peak += max_allocated
            print(f"  GPU {i}: Current={allocated:.2f} GB, Reserved={reserved:.2f} GB, Peak={max_allocated:.2f} GB")
        print(f"  Total Peak Across All GPUs: {total_peak:.2f} GB")
        print(f"{'='*60}\n")
    
    if inference_only:
        print(f"Processed {total} examples for inference.")
    else:
        # Calculate and display all accuracy metrics
        exact_acc = 100.0 * exact_correct / total if total > 0 else 0.0
        token_acc = 100.0 * token_correct / token_total if token_total > 0 else 0.0
        numerical_acc = 100.0 * numerical_correct / total if total > 0 else 0.0
        malformed_prop = 100.0 * malformed_count / total if total > 0 else 0.0
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Total examples: {total}")
        print(f"Malformed expressions: {malformed_prop:.2f}% ({malformed_count}/{total})")
        print(f"Numerical equivalence accuracy: {numerical_acc:.2f}% ({numerical_correct}/{total})")
        print(f"Token-level accuracy: {token_acc:.2f}% ({token_correct}/{token_total})")
        print(f"Exact sequence match accuracy: {exact_acc:.2f}% ({exact_correct}/{total})")


if __name__ == "__main__":
    main()