import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Dataset, scrambled amplitudes
class ExpressionDataset(Dataset):
    def __init__(self, num_samples=100000):
        self.samples = [self.generate_scramble() for _ in range(num_samples)]
        # Print example using self.multi_scramble (this runs when an instance is created)
        simple_expr = "e1.p2 + e3.p4"
        complex_expr = self.multi_scramble(simple_expr, times=3, total_particles=5, include_mass=True)
        print("Simple expr: e1.p2 + e3.p4. Scrambled", complex_expr)

    # Enacts momentum conservation, i.e. it enacts p_index = -(sum_{k != index} p_k)    
    def momentum_substitution(self, momentum_index, total_particles=5):
        others = [f"p{k}" for k in range(1, total_particles+1) if k != momentum_index]
        return "(-" + "+".join(others) + ")"
    
    # Scramble an expression
    def scramble_expression(self, expr, total_particles=5, include_mass=False):
        """
        Scrambles a simple amplitude expression (given as a string) by applying one of three operations.
        
        Parameters:
        expr            : The simple amplitude expression (e.g. "e1.p2 + e3.p4").
        total_particles : Total number of particles (and hence available indices).
        include_mass    : If True, also allow scramble operations involving masses.
        
        Operations:
        0. Multiply-by-1 using an ε·p fraction. Optionally, the numerator can have momentum conservation applied.
        1. Add zero using the transversality condition (e_i.p_i = 0), 
            i.e. add (e_i.p_i)/(e_i.p_j) with j ≠ i.
        2. [Optional] Multiply-by-1 using masses: (m_i²)/(m_i²).
        """
        # Choose which operation to perform.
        ops = [0, 1]
        if include_mass:
            ops.append(2)
        op = random.choice(ops)
        
        if op == 0:
            # Multiply-by-1 using an ε·p fraction or a p.p function 
            i = random.randint(1, total_particles)
            j = random.randint(1, total_particles)
            # Substitute momentum conservation in the numerator.
            substituted = self.momentum_substitution(j, total_particles)
            if random.choice([True, False]):
                numerator = f"e{i}.{substituted}"
                denominator = f"e{i}.p{j}"
            else:
                numerator = f"p{i}.{substituted}"
                denominator = f"p{i}.p{j}"
            fraction = f"({numerator})/({denominator})"
            scrambled = f"({expr})*{fraction}"
        
        elif op == 1:
            # --- Add Zero using transversality: e_i.p_i = 0 ---
            i = random.randint(1, total_particles)
            # Choose a denominator index j such that j != i
            j_choices = [k for k in range(1, total_particles+1) if k != i]
            j = random.choice(j_choices)
            # Since e_i.p_i = 0, this fraction is identically zero.
            addition = f"(e{i}.p{i})/(e{i}.p{j})"
            scrambled = f"({expr})+{addition}"
        
        elif op == 2:
            # --- Multiply-by-1 using masses ---
            i = random.randint(1, total_particles)
            fraction = f"(m{i}^2)/(m{i}^2)"
            scrambled = f"({expr})*{fraction}"
        
        return scrambled
    
    def multi_scramble(self, expr, times=3, total_particles=5, include_mass=False):
        scrambled_expr = expr
        for _ in range(times):
            scrambled_expr = self.scramble_expression(scrambled_expr, total_particles, include_mass)
        return scrambled_expr
    
    def generate_scramble(self):
        # For illustration, scramble a fixed simple expression.
        simple_expr = "e1.p2 + e3.p4"
        complex_expr = self.multi_scramble(simple_expr, times=3, total_particles=5, include_mass=True)
        # Return a tuple (complex expression, simple expression)
        return complex_expr, simple_expr
    
    def generate_sample(self):
        # Use the scrambler above to generate a bunch of data
        return complex_expr, simple_expr

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Return source (complex) and target (simple) expressions.
        return self.samples[idx]
# Tokenizer - convert amplitude bits into tokens
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(vocab)}
    
    def encode(self, text):
        # A simple whitespace-based tokenizer.
        tokens = text.strip().split()
        return [self.token_to_id[t] for t in tokens]
    
    def decode(self, ids):
        return " ".join(self.id_to_token[i] for i in ids)

vocab = [
        "<pad>", "<s>", "</s>",
        "eps",    # polarization vector symbol
        "p",      # momentum vector symbol
        "dot",    # dot product operator
        "(", ")",
        "^", # powers
        "+", "-", "*",
        "1", "2", "3", "4", "5", "6", "7", "8"  # particle labels
]
tokenizer = SimpleTokenizer(vocab)
vocab_size = len(vocab)

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If odd dimension, pad one column of zeros for cosine
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# === Transformer Model ===
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model,
                                          nhead,
                                          num_encoder_layers,
                                          num_decoder_layers,
                                          dim_feedforward,
                                          dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src, tgt: (seq_len, batch_size)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb.transpose(0,1)).transpose(0,1)  # back to (seq_len, batch_size, d_model)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0,1)).transpose(0,1)
        
        # Transformer expects (seq_len, batch_size, d_model)
        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(output)

# === Data Preparation Helpers ===
def collate_fn(batch):
    # For each batch element, tokenize and pad both source (complex_expr) and target (simple_expr)
    src_sequences, tgt_sequences = zip(*batch)
    src_tokenized = [tokenizer.encode(seq) for seq in src_sequences]
    tgt_tokenized = [tokenizer.encode("<s> " + seq + " </s>") for seq in tgt_sequences]
    
    # Determine max lengths
    src_max = max(len(seq) for seq in src_tokenized)
    tgt_max = max(len(seq) for seq in tgt_tokenized)
    
    # Pad sequences (using 0 as pad token; you may define a specific token)
    src_padded = [seq + [0]*(src_max-len(seq)) for seq in src_tokenized]
    tgt_padded = [seq + [0]*(tgt_max-len(seq)) for seq in tgt_tokenized]
    
    # Convert to tensors and transpose to shape (seq_len, batch_size)
    src_tensor = torch.tensor(src_padded, dtype=torch.long).transpose(0,1)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long).transpose(0,1)
    return src_tensor, tgt_tensor

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ExpressionDataset(num_samples=100000)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

model = TransformerModel(vocab_size, d_model=128, nhead=8,
                         num_encoder_layers=3, num_decoder_layers=3,
                         dim_feedforward=256, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padded tokens

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src = src.to(device)  # shape: (src_seq_len, batch_size)
        tgt = tgt.to(device)  # shape: (tgt_seq_len, batch_size)
        
        optimizer.zero_grad()
        tgt_input = tgt[:-1, :]  # all tokens except last as input to decoder
        tgt_out = tgt[1:, :]     # target output
        
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        
        # Forward pass
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        # output shape: (tgt_seq_len, batch_size, vocab_size)
        loss = criterion(output.reshape(-1, vocab_size), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
