'''Script for training a transformer model on amplitude simplification'''
# Import libraries
import os
import torch
import torch.nn as nn

# Import from other files
from data_import import load_and_prepare_data
from transformer_functions import create_model, train_model

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the run hyperparameters
filename = 'w1.csv'
hyperparams = {
    'embedding_dim': 256,
    'n_heads': 8,
    'n_enc_layers': 6,
    'n_dec_layers': 6,
    'dropout': 0.1,
    'sinusoidal_embeddings': True,
    'device': device
}

# Define the vocabulary
vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "+": 4,
    "-": 5,
    "*": 6,
    "/": 7,
    "^": 8,
    "(": 9,
    ")": 10,
    "0:": 11,
    "1:": 12,
    "2:": 13,
    "3:": 14,
    "4:": 15,
    "5:": 16,
    "6:": 17,
    "7:": 18,
    "8:": 19,
    "9:": 20,
    "10:": 21,
    "·": 22, # dot operator.
    "Tr": 23,
    "u-": 24,  # unary minus
    "M": 25,  # mass. Probably best not used!
}
vocab_size = len(vocab.keys())

# Load data
csv_file = os.getcwd()+'/Data/'+filename

# Set up the dataloaders
train_loader, val_loader = load_and_prepare_data(
    csv_file, 
    batch_size=16, 
    max_length=None, 
    train_split=0.8
 )

# Create model (automatically moves to device)
model = create_model(vocab_size, **hyperparams)

# Loss function (ignore padding tokens)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad token

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Hyperparameters: {hyperparams}")

####

print("Embedding layer vocab size:", model.src_embedding.num_embeddings)

def get_max_token_id(loader):
    max_token = 0
    for batch in loader:
        current_max = batch['input'].max().item()  # Get max token ID in batch
        if current_max > max_token:
            max_token = current_max
    return max_token

# Check max token IDs in both loaders
train_max_token = get_max_token_id(train_loader)
val_max_token = get_max_token_id(val_loader)

global_max_token = max(train_max_token, val_max_token)
vocab_size = model.src_embedding.num_embeddings

print(f"Max token ID in train: {train_max_token}")
print(f"Max token ID in val: {val_max_token}")
print(f"Global max token ID: {global_max_token}")
print(f"Embedding layer vocab size: {vocab_size}")

#####


# Run training
train_losses, val_losses = train_model(
    model, 
    optimizer, 
    criterion, 
    train_loader, 
    val_loader, 
    epochs=10
)
