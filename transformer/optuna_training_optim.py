import os
import sys
import torch
import torch.nn as nn
import optuna
from transformer_functions import create_model, train_model
from data_import import load_and_prepare_data

# Detect and set number of CPU threads for PyTorch
n_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(n_threads)
print(f"Using {n_threads} CPU threads for PyTorch.")

# Set number of particles N from command line argument or default to 4
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# Configuration: Save models during trials (default: False to avoid creating directories/files)
SAVE_MODELS = False

# Optuna hyperparameters
n_trials = 50
n_startup_trials = 20

def objective(trial):
    # Fixed hyperparameters
    n_epochs = 20
    # Early stopping parameters
    early_stopping_patience = 5  # Shorter patience for optimization
    early_stopping_min_delta = 1e-4
    
    # Suggest hyperparameters
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_exp', 6, 9)  # 64, 128, 256, 512
    n_heads = 2 ** trial.suggest_int('n_heads_exp', 1, 4)  # 2, 4, 8, 16
    n_enc_layers = trial.suggest_int('n_enc_layers', 2, 8)
    n_dec_layers = trial.suggest_int('n_dec_layers', 2, 8)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    sinusoidal_embeddings = trial.suggest_categorical('sinusoidal_embeddings', [True, False])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = 2 ** trial.suggest_int('batch_size_exp', 3, 6)  # 8, 16, 32, 64
    head_ff_dim = embedding_dim * trial.suggest_categorical('head_ff_dim_mult', [1, 2, 4, 8])

    # Data
    csv_file = f'gi_{N}pt_tok.csv'
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

    # Train with early stopping
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        val_loader, 
        epochs=n_epochs,
        run_name=f'optuna_trial_{trial.number}' if SAVE_MODELS else 'temp_trial',
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        save_models=SAVE_MODELS
    )

    # Return the best (minimum) validation loss achieved during training
    return min(val_losses)

def main():
    # Ensure n_trials is greater than n_startup_trials
    assert n_trials > n_startup_trials, f"n_trials ({n_trials}) must be greater than n_startup_trials ({n_startup_trials})"
    
    # Create an Optuna study to minimize validation loss
    # TPE sampler with explicit n_startup_trials
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # Run optimization for a specified number of trials
    study.optimize(objective, n_trials=n_trials)
    
    # Print the best trial's results
    print('Best trial:')
    trial = study.best_trial
    print(f'  Trial Number: {trial.number}')
    print(f'  Validation Loss: {trial.value}')
    print('  Best Hyperparameters:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == '__main__':
    main() 
