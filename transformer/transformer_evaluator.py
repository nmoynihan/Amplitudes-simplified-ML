import os
import torch
# Detect and set number of CPU threads for PyTorch
n_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(n_threads)
print(f"Using {n_threads} CPU threads for PyTorch.")
from transformer_functions import TransformerRegressor, load_transformer_model
from data_import import load_and_prepare_data

# Settings
model_path = os.path.join('models', '5pt_model.pt')  # Change as needed
csv_file = os.path.join('data', 'ampl00111_tok.csv')  # Change as needed
batch_size = 1
max_length = 100
num_print = 5  # Number of examples to print
inference_only = True  # Set to True for pure inference (ignore simple column), False for evaluation
force_cpu = True # Force CPU usage (set to True to avoid CUDA/MPS device issues)
use_mps = False # Device toggle: enable MPS explicitly (default False due to missing ops in PyTorch Transformer on MPS)

# Decoding hyperparameters (set here for evaluation)
decoding_method = 'beam'        # Try 'beam' instead of 'nucleus' to test non-stochastic
beam_size = 10                   # Used for beam/nucleus search
p_nucleus = 0.8                 # Nucleus cutoff probability (lower => more diversity)
temperature_nucleus = 2.0       # Temperature for nucleus sampling (increased from 1.0 for more diversity)
# For beam/nucleus evaluation: if True, count as correct if ANY beam hypothesis matches target; if False, only best hyp
beam_match_any = True

def decode_with_model(model, src, max_length, bos_token=2, eos_token=3, pad_token=0):
    """Decode and, for beam/nucleus, also return all beam hypotheses per example."""
    if decoding_method == 'greedy':
        out = model.generate(src, max_length=max_length, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token)
        return out, None
    elif decoding_method in ['beam', 'nucleus']:
        stochastic = (decoding_method == 'nucleus')
        decoded, tgt_len, generated_hyps = model.generate_beam(
            src,
            beam_size=beam_size,
            length_penalty=1.0,
            early_stopping=True,
            max_length=max_length,
            stochastic=stochastic,
            nucl_p=p_nucleus,
            temperature=temperature_nucleus,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token
        )
        # Prepare list of all hypotheses per sample (append EOS for fair comparison)
        all_beams = []
        for hyps in generated_hyps:
            hyps_for_sample = []
            for score, hyp in hyps.hyp:
                seq = hyp.tolist() + [eos_token]
                hyps_for_sample.append(seq)
            all_beams.append(hyps_for_sample)
        # Return in same shape as greedy for compatibility, plus beams
        return decoded.transpose(0, 1), all_beams
    else:
        raise ValueError(f"Unknown decoding method: {decoding_method}")

def _clean_seq(arr, pad_token=0, eos_token=3):
    """Remove PAD and truncate at first EOS (inclusive) for fair comparison/printing."""
    out = []
    for x in arr:
        if x == pad_token:
            continue
        out.append(int(x))
        if x == eos_token:
            break
    return out

def main():
    # Resolve device and load model on it (prefer CUDA, then optional MPS, else CPU)
    if force_cpu:
        preferred_device = 'cpu'
        print("Forcing CPU device usage")
    else:
        try:
            if torch.cuda.is_available():
                # Test if CUDA actually works by creating a small tensor
                test_tensor = torch.zeros(1).cuda()
                preferred_device = 'cuda'
                print("CUDA device available and working")
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
    model.to(preferred_device)
    model.device = preferred_device  
    model.eval()
    device = preferred_device
    print(f"Running on device: {device}")
    
    if inference_only:
        print("=== INFERENCE MODE ===")
        print(f"Decoding method: {decoding_method}")
        if decoding_method in ['beam', 'nucleus']:
            print(f"Beam size: {beam_size}")
        print()
    else:
        print("=== EVALUATION MODE ===")

    # Load data - use full dataset for inference, validation only for evaluation
    if inference_only:
        # Load full dataset without train/validation split
        from data_import import TransformerDataset
        from torch.utils.data import DataLoader
        dataset = TransformerDataset(csv_file, max_length=None)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        # Load validation set only for evaluation
        _, data_loader = load_and_prepare_data(csv_file, batch_size=batch_size, max_length=None, train_split=0.8)

    total, correct = 0, 0
    printed = 0
    
    for batch in data_loader:
        src = batch['input'].to(device)
        tgt = batch['target'].to(device)
        
        # Generate predictions
        with torch.no_grad():
            decode_len = min(max_length, tgt.size(1)) if max_length is not None else tgt.size(1)
            gen, beams = decode_with_model(model, src, max_length=decode_len, bos_token=2, eos_token=3, pad_token=0)
        
        gen = gen.cpu().numpy()
        tgt = tgt.cpu().numpy()
        
        # Process each example in the batch
        for i in range(src.size(0)):
            if inference_only:
                # INFERENCE MODE: Just print predictions
                src_seq = src[i].cpu().numpy().tolist()
                gen_seq = _clean_seq(gen[i], pad_token=0, eos_token=3)
                
                print(f"Input:     {src_seq}")
                if decoding_method == 'greedy':
                    print(f"Predicted: {gen_seq}")
                else:  # beam or nucleus
                    print(f"Best prediction: {gen_seq}")
                    if beams and i < len(beams):
                        print(f"All {beam_size} beam hypotheses:")
                        for j, hyp_seq in enumerate(beams[i][:beam_size]):
                            clean_hyp = _clean_seq(hyp_seq, pad_token=0, eos_token=3)
                            print(f"  Beam {j+1}: {clean_hyp}")
                print()
                total += 1
                
            else:
                # EVALUATION MODE: Compare with targets
                tgt_seq = _clean_seq(tgt[i], pad_token=0, eos_token=3)
                gen_seq = _clean_seq(gen[i], pad_token=0, eos_token=3)
                is_match = False
                
                if beams is None:
                    # Greedy: compare best sequence only
                    is_match = (tgt_seq == gen_seq)
                else:
                    # Beam/nucleus: either any-beam match or best-only match
                    if not beam_match_any:
                        is_match = (tgt_seq == gen_seq)
                    else:
                        beam_list = beams[i] if i < len(beams) else []
                        if len(beam_list) == 0:
                            is_match = (tgt_seq == gen_seq)
                        else:
                            for hyp_seq in beam_list:
                                if _clean_seq(hyp_seq, pad_token=0, eos_token=3) == tgt_seq:
                                    is_match = True
                                    break
                
                if is_match:
                    correct += 1
                total += 1
                
                if printed < num_print:
                    print(f"Input:    {src[i].cpu().numpy().tolist()}")
                    print(f"Target:   {tgt_seq}")
                    print(f"Generated:{gen_seq}\n")
                    printed += 1

    if inference_only:
        print(f"Processed {total} examples for inference.")
    else:
        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"\nExact match accuracy: {acc:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()