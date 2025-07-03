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

# Detect and set number of CPU threads for PyTorch
n_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(n_threads)
print(f"Using {n_threads} CPU threads for PyTorch.")

# Specify the run hyperparameters
filename = 'w1_short.csv'
training_hyperparams = {
    'n_epochs': 1,
    'batch_size': 16,
    'train_split': 0.8,
    'learning_rate': 1e-4
}
model_hyperparams = {
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
vocab_size = 58 ###len(vocab.keys())

# Load data
csv_file = os.getcwd()+'/data/'+filename

# Set up the dataloaders
train_loader, val_loader = load_and_prepare_data(
    csv_file, 
    batch_size=training_hyperparams['batch_size'], 
    max_length=None, 
    train_split=training_hyperparams['train_split']
 )

# Create model (automatically moves to device)
model = create_model(vocab_size, **model_hyperparams)

# Loss function (ignore padding tokens)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad token

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=training_hyperparams['learning_rate'])

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")
print(f"Hyperparameters:\nModel:{model_hyperparams}\nTraining:{training_hyperparams}")

# Run training
train_losses, val_losses = train_model(
    model, 
    optimizer, 
    criterion, 
    train_loader, 
    val_loader, 
    epochs=training_hyperparams['n_epochs']
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
