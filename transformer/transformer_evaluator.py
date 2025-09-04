import os
import torch
# Detect and set number of CPU threads for PyTorch
n_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(n_threads)
print(f"Using {n_threads} CPU threads for PyTorch.")
from transformer_functions import TransformerRegressor, load_transformer_model
from data_import import load_and_prepare_data

# Settings
model_path = os.path.join('models', 'transformer_e2.pt')  # Change as needed
csv_file = os.path.join('data', 'gi_4pt_tok.csv')           # Change as needed
batch_size = 16
max_length = 100
num_print = 5  # Number of examples to print

# Decoding hyperparameters (set here for evaluation)
decoding_method = 'nucleus'      # Options: 'greedy', 'beam', 'nucleus'
beam_size = 5                   # Used for beam/nucleus search
p_nucleus = 0.95                # Nucleus cutoff probability (for nucleus sampling)
temperature_nucleus = 1.0       # Temperature for nucleus sampling
# For beam/nucleus evaluation: if True, count as correct if ANY beam hypothesis matches target; if False, only best hyp
beam_match_any = True
# Device toggle: enable MPS explicitly (default False due to missing ops in PyTorch Transformer on MPS)
use_mps = False

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
    if torch.cuda.is_available():
        preferred_device = 'cuda'
    elif use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        preferred_device = 'mps'
    else:
        preferred_device = 'cpu'
    loaded = load_transformer_model(TransformerRegressor, model_path, device=preferred_device)
    model = loaded['model']
    model.to(preferred_device)
    model.device = preferred_device  
    model.eval()
    device = preferred_device
    print(f"Evaluating on device: {device}")

    # Load data (validation set only)
    _, val_loader = load_and_prepare_data(csv_file, batch_size=batch_size, max_length=None, train_split=0.8)

    total, correct = 0, 0
    printed = 0
    for batch in val_loader:
        src = batch['input'].to(device)
        tgt = batch['target'].to(device)
        # Generate predictions
        with torch.no_grad():
            decode_len = min(max_length, tgt.size(1)) if max_length is not None else tgt.size(1)
            gen, beams = decode_with_model(model, src, max_length=decode_len, bos_token=2, eos_token=3, pad_token=0)
        gen = gen.cpu().numpy()
        tgt = tgt.cpu().numpy()
        # Compare and print
        for i in range(src.size(0)):
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
    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"\nExact match accuracy: {acc:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()