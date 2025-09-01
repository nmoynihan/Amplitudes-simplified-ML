import os
import torch
import torch.nn as nn
import optuna
from transformer_functions import create_model, train_model
from data_import import load_and_prepare_data

# Detect and set number of CPU threads for PyTorch
n_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(n_threads)
print(f"Using {n_threads} CPU threads for PyTorch.")

def objective(trial):
    # Fixed hyperparameters
    n_epochs = 20
    # Suggest hyperparameters
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_exp', 6, 9)  # 64, 128, 256, 512
    n_heads = 2 * trial.suggest_int('n_heads_exp', 1, 5)  # 2, 4, 6, 8, 10
    n_enc_layers = trial.suggest_int('n_enc_layers', 2, 8)
    n_dec_layers = trial.suggest_int('n_dec_layers', 2, 8)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    sinusoidal_embeddings = trial.suggest_categorical('sinusoidal_embeddings', [True, False])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = 2 ** trial.suggest_int('batch_size_exp', 3, 6)  # 8, 16, 32, 64
    head_ff_dim = embedding_dim * trial.suggest_categorical('head_ff_dim_mult', [1, 2, 4, 8])

    # Data
    csv_file = os.path.join('data', 'w1_short.csv')
    train_loader, val_loader = load_and_prepare_data(
        csv_file,
        batch_size=batch_size,
        max_length=None,
        train_split=0.8
    )

    # Model
    vocab_size = 58  # Update if needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_hyperparams = {
        'embedding_dim': embedding_dim,
        'n_heads': n_heads,
        'n_enc_layers': n_enc_layers,
        'n_dec_layers': n_dec_layers,
        'dropout': dropout,
        'sinusoidal_embeddings': sinusoidal_embeddings,
        'head_ff_dim': head_ff_dim,
        'device': device
    }
    model = create_model(vocab_size, **model_hyperparams)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train
    train_model(model, optimizer, criterion, train_loader, val_loader, epochs=n_epochs)

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch['input'].to(device)
            tgt = batch['target'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            val_loss += loss.item()
            n_batches += 1
    avg_val_loss = val_loss / n_batches if n_batches > 0 else float('inf')
    return avg_val_loss

def main():
    # Create an Optuna study to minimize validation loss
    study = optuna.create_study(direction='minimize')
    
    # Run optimization for a specified number of trials
    study.optimize(objective, n_trials=20)
    
    # Print the best trial's results
    print('Best trial:')
    trial = study.best_trial
    print(f'  Validation Loss: {trial.value}')
    print('  Best Hyperparameters:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == '__main__':
    main() 