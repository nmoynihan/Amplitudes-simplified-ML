import os
import torch
from transformer_functions import TransformerRegressor, load_transformer_model
from data_import import load_and_prepare_data

# Settings
model_path = os.path.join('models', 'transformer_e1.pt')  # Change as needed
csv_file = os.path.join('data', 'w1_short.csv')           # Change as needed
batch_size = 16
max_length = 100
num_print = 5  # Number of examples to print

# Decoding hyperparameters (set here for evaluation)
decoding_method = 'nucleus'      # Options: 'greedy', 'beam', 'nucleus'
beam_size = 5                   # Used for beam/nucleus search
p_nucleus = 0.95                # Nucleus cutoff probability (for nucleus sampling)
temperature_nucleus = 1.0       # Temperature for nucleus sampling

def decode_with_model(model, src, max_length, bos_token=2, eos_token=3, pad_token=0):
    if decoding_method == 'greedy':
        return model.generate(src, max_length=max_length, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token)
    elif decoding_method in ['beam', 'nucleus']:
        stochastic = (decoding_method == 'nucleus')
        decoded, tgt_len, _ = model.generate_beam(
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
        # Return in same shape as greedy for compatibility
        return decoded.transpose(0, 1)
    else:
        raise ValueError(f"Unknown decoding method: {decoding_method}")

def main():
    # Load model
    loaded = load_transformer_model(TransformerRegressor, model_path)
    model = loaded['model']
    model.eval()
    device = model.device

    # Load data (validation set only)
    _, val_loader = load_and_prepare_data(csv_file, batch_size=batch_size, max_length=None, train_split=0.8)

    total, correct = 0, 0
    printed = 0
    for batch in val_loader:
        src = batch['input'].to(device)
        tgt = batch['target'].to(device)
        tgt_nopad = tgt.cpu().numpy()
        # Generate predictions
        with torch.no_grad():
            gen = decode_with_model(model, src, max_length=tgt.size(1), bos_token=2, eos_token=3, pad_token=0)
        gen = gen.cpu().numpy()
        tgt = tgt.cpu().numpy()
        # Compare and print
        for i in range(src.size(0)):
            # Remove padding and special tokens for comparison
            tgt_seq = [x for x in tgt[i] if x != 0]
            gen_seq = [x for x in gen[i] if x != 0]
            if tgt_seq == gen_seq:
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