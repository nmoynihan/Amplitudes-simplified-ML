'''Script to define the functions used in initialising and training the transformer model'''
import os
import torch
import torch.nn as nn
import math
from tqdm import tqdm

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used in 'Attention Is All You Need'"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.pe(positions)


class TransformerRegressor(nn.Module):
    """
    Transformer model for sequence-to-sequence regression
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_heads: int,
        n_enc_layers: int,
        n_dec_layers: int,
        dropout: float = 0.1,
        sinusoidal_embeddings: bool = True,
        max_seq_len: int = 5000,
        pad_token_id: int = 0,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.device = device
        
        # Token embeddings
        self.src_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.tgt_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        
        # Positional encodings
        if sinusoidal_embeddings:
            self.src_pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_len)
            self.tgt_pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_len)
        else:
            self.src_pos_encoding = LearnedPositionalEncoding(embedding_dim, max_seq_len)
            self.tgt_pos_encoding = LearnedPositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=n_heads,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dropout=dropout,
            batch_first=True  # batch_size, seq_len, embedding_dim
        )
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device
        self.to(device)
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, x, pad_token_id):
        """Create padding mask for attention"""
        return (x == pad_token_id)
    
    def create_causal_mask(self, size):
        """Create causal (triangular) mask for decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass
        
        Args:
            src: Source sequences [batch_size, src_seq_len]
            tgt: Target sequences [batch_size, tgt_seq_len]
            src_key_padding_mask: Padding mask for source [batch_size, src_seq_len]
            tgt_key_padding_mask: Padding mask for target [batch_size, tgt_seq_len]
        
        Returns:
            Output logits [batch_size, tgt_seq_len, vocab_size]
        """
        # Create masks if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = self.create_padding_mask(src, self.pad_token_id)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.create_padding_mask(tgt, self.pad_token_id)
        
        # Create causal mask for decoder
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.create_causal_mask(tgt_seq_len).to(tgt.device)
        
        # Embeddings
        src_emb = self.src_embedding(src) * math.sqrt(self.embedding_dim)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.embedding_dim)
        
        # Positional encoding
        src_emb = self.src_pos_encoding(src_emb)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        
        # Dropout
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Transformer
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Output projection
        output = self.output_projection(output)
        
        return output
    
    def generate(self, src, max_length=100, bos_token=2, eos_token=3, pad_token=0):
        """
        Generate sequences using the trained model
        
        Args:
            src: Source sequence [batch_size, src_seq_len]
            max_length: Maximum generation length
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            pad_token: Padding token
        
        Returns:
            Generated sequences [batch_size, generated_seq_len]
        """
        self.eval()
        batch_size = src.size(0)
        
        # Ensure source is on correct device
        src = src.to(self.device)
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_token, device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Forward pass
                output = self.forward(src, generated)
                
                # Get next token predictions (last position)
                next_token_logits = output[:, -1, :]  # [batch_size, vocab_size]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=1)
                
                # Check if all sequences have generated EOS token
                if (next_tokens.squeeze() == eos_token).all():
                    break
        
        return generated


def create_model(vocab_size, **hyperparams):
    """
    Factory function to create transformer model with hyperparameters
    
    Args:
        vocab_size: Vocabulary size
        **hyperparams: Model hyperparameters
            - embedding_dim: Embedding dimension
            - n_heads: Number of attention heads
            - n_enc_layers: Number of encoder layers
            - n_dec_layers: Number of decoder layers
            - dropout: Dropout rate
            - sinusoidal_embeddings: Use sinusoidal vs learned positional encodings
    
    Returns:
        TransformerRegressor model
    """
    return TransformerRegressor(vocab_size=vocab_size, **hyperparams)


def train_step(model, batch, criterion, optimizer):
    """Single training step"""
    model.train()
    
    # Move batch to model's device
    src = batch['input'].to(model.device)  # scrambled sequences
    tgt = batch['target'].to(model.device)  # simple sequences
    
    # Prepare target input and output
    tgt_input = tgt[:, :-1]  # All but last token
    tgt_output = tgt[:, 1:]  # All but first token (shifted)
    
    # Forward pass
    optimizer.zero_grad()
    output = model(src, tgt_input)
    
    # Reshape for loss calculation
    output = output.reshape(-1, output.size(-1))
    tgt_output = tgt_output.reshape(-1)
    
    # Calculate loss (ignore padding tokens)
    loss = criterion(output, tgt_output)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def validate_step(model, batch, criterion):
    """Single validation step"""
    model.eval()
    
    with torch.no_grad():
        # Move batch to model's device
        src = batch['input'].to(model.device)
        tgt = batch['target'].to(model.device)
        
        # Prepare target input and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Forward pass
        output = model(src, tgt_input)
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)
    
    return loss.item()


def train_model(model, optimizer, criterion, train_loader, val_loader, epochs):
    train_losses, val_losses = [], []
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            loss = train_step(model, batch, criterion, optimizer)
            train_loss += loss
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                loss = validate_step(model, batch, criterion)
                val_loss += loss
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        # Save model after each epoch
        model_path = os.path.join('models', f'transformer_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
        }, model_path)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f} | Model saved to {model_path}")
    
    return train_losses, val_losses
