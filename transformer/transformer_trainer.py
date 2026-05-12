'''Script for training a transformer model on amplitude simplification'''
# Import libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"           # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"      # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"           # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"    # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"       # NumExpr
import multiprocessing
import torch
import torch.nn as nn
import argparse

# Import from other files
from data_import import load_and_prepare_data
from transformer_functions import create_model, train_model

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cpu':
    n_threads = multiprocessing.cpu_count()
    torch.set_num_threads(n_threads)
    print(f"Using {n_threads} CPU threads for PyTorch.")

# Specify the run hyperparameters
filename = ['gi_4pt_os_tok.csv'] #'gi_4pt_tok.csv'
training_hyperparams = {
    'n_epochs': 30,
    'batch_size': 16, # prev 16. Setting this to 4 makes this run on a 16gb GPU.
    'train_split': 0.8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 8,  # Stop if no improvement for 10 epochs
    'early_stopping_min_delta': 1e-4  # Minimum change to qualify as improvement
}
model_hyperparams = {
    'embedding_dim': 256, # prev 512
    'n_heads': 4, # prev 4
    'n_enc_layers': 4, # prev 5
    'n_dec_layers': 4, # prev 5
    'dropout': 0.025,
    'sinusoidal_embeddings': True,
    'head_ff_dim': 512,  # 4 * embedding_dim (4 * 64 = 256)
    'device': device
}

# Hardcode the vocab size (use what specified in data_generation/ scripts)
vocab_size = 58 

# Load data
# Set up the dataloaders
train_loader, val_loader = load_and_prepare_data(
    filename, 
    batch_size=training_hyperparams['batch_size'], 
    max_length=None, 
    train_split=training_hyperparams['train_split']
 )

# Create model (automatically moves to device)
model = create_model(vocab_size, **model_hyperparams)

# Loss function (ignore padding tokens)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad token

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=training_hyperparams['learning_rate'], weight_decay=training_hyperparams['weight_decay'])

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")
print(f"Hyperparameters:\nModel:{model_hyperparams}\nTraining:{training_hyperparams}")

# Run training
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='default_run', help='Unique name for this training run')
args = parser.parse_args()
run_name = args.run_name
train_losses, val_losses, train_accuracies, val_accuracies= train_model(
    model, 
    optimizer, 
    criterion, 
    train_loader, 
    val_loader, 
    epochs=training_hyperparams['n_epochs'],
    run_name=run_name,
    early_stopping_patience=training_hyperparams['early_stopping_patience'],
    early_stopping_min_delta=training_hyperparams['early_stopping_min_delta']
)

'''
# Code to import a pre-trained transformer model
from transformer_functions import TransformerRegressor, load_transformer_model

# Load the model
loaded_model_path = '../models/transformer_e2.pt'
loaded_data = load_transformer_model(TransformerRegressor, loaded_model_path)
loaded_model = loaded_data['model']
print(f"Loaded model from epoch {loaded_data['epoch']}")
'''
