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
        head_ff_dim: int = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.dropout = dropout
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.head_ff_dim = head_ff_dim if head_ff_dim is not None else 4 * embedding_dim
        self.device = device
        self.model_hyperparams = {'vocab_size': self.vocab_size, 
                                 'embedding_dim': self.embedding_dim,
                                 'n_heads': self.n_heads,
                                 'n_enc_layers': self.n_enc_layers,
                                 'n_dec_layers': self.n_dec_layers,
                                 'dropout': self.dropout,
                                 'sinusoidal_embeddings': self.sinusoidal_embeddings,
                                 'max_seq_len': self.max_seq_len,
                                 'pad_token_id': self.pad_token_id,
                                 'head_ff_dim': self.head_ff_dim,
                                 'device': self.device,
                                 }
        
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
        # enable_nested_tensor=False suppresses the prototype nested-tensor warning
        # triggered when src_key_padding_mask is passed to the encoder.
        # The kwarg was added in PyTorch 1.13; skip it on older versions.
        _transformer_kwargs = dict(
            d_model=embedding_dim,
            nhead=n_heads,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=self.head_ff_dim,
            dropout=dropout,
            batch_first=True,  # batch_size, seq_len, embedding_dim
        )
        import inspect
        if "enable_nested_tensor" in inspect.signature(nn.Transformer.__init__).parameters:
            _transformer_kwargs["enable_nested_tensor"] = False
        self.transformer = nn.Transformer(**_transformer_kwargs)
        
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
        Generate sequences using the trained model.
        Encoder output is cached once and reused every decoding step (Fix 1).
        Per-sequence early stopping emits pad tokens for finished sequences (Fix 4).
        
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
        
        with torch.no_grad():
            # --- Fix 1: Run encoder once and cache memory ---
            src_key_padding_mask = self.create_padding_mask(src, self.pad_token_id)
            src_emb = self.src_embedding(src) * math.sqrt(self.embedding_dim)
            src_emb = self.src_pos_encoding(src_emb)
            src_emb = self.dropout(src_emb)
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            
            # Start with BOS token
            generated = torch.full((batch_size, 1), bos_token, device=self.device, dtype=torch.long)
            
            # --- Fix 4: Track which sequences have finished ---
            done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            
            for _ in range(max_length - 1):
                tgt_seq_len = generated.size(1)
                tgt_mask = self.create_causal_mask(tgt_seq_len).to(self.device)
                
                tgt_emb = self.tgt_embedding(generated) * math.sqrt(self.embedding_dim)
                tgt_emb = self.tgt_pos_encoding(tgt_emb)
                tgt_emb = self.dropout(tgt_emb)
                
                # Decoder only — encoder memory is already computed
                dec_out = self.transformer.decoder(
                    tgt_emb, memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                output = self.output_projection(dec_out)
                
                # Get next token predictions (last position)
                next_token_logits = output[:, -1, :]  # [batch_size, vocab_size]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
                
                # --- Fix 4: Finished sequences emit pad instead of a real token ---
                next_tokens = torch.where(
                    done.unsqueeze(1),
                    torch.full_like(next_tokens, pad_token),
                    next_tokens,
                )
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=1)
                
                # Update done mask and check for full completion
                done = done | (next_tokens.squeeze(1) == eos_token)
                if done.all():
                    break
        
        return generated

    def generate_beam(self, src, beam_size=5, length_penalty=1.0, early_stopping=True, max_length=100,
                      stochastic=True, nucl_p=0.95, temperature=1.0, rng_gen=None,
                      bos_token=2, eos_token=3, pad_token=0):
        """
        Beam search or nucleus sampling decoding for sequence generation.
        Args:
            src: Source sequence [batch_size, src_seq_len]
            beam_size: Number of beams
            length_penalty: Length penalty for beam search
            early_stopping: Whether to stop early when enough hypotheses are found
            max_length: Maximum generation length
            stochastic: If True, use nucleus sampling; else, standard beam search
            nucl_p: Nucleus cutoff probability (for nucleus sampling)
            temperature: Softmax temperature (for stochastic sampling)
            rng_gen: Optional torch.Generator for reproducibility
            bos_token, eos_token, pad_token: Special token ids
        Returns:
            decoded: [max_length, batch_size] tensor of generated tokens
            tgt_len: [batch_size] tensor of output lengths
            generated_hyps: List of BeamHypotheses objects
        """
        self.eval()
        device = self.device
        src = src.to(device)
        batch_size = src.size(0)
        n_words = self.vocab_size

        # Expand source for beam size
        src_rep = src.unsqueeze(1).expand((batch_size, beam_size, src.size(1))).contiguous().view(batch_size * beam_size, src.size(1))

        # Generated tokens: [max_length, batch_size * beam_size]
        generated = torch.full((max_length, batch_size * beam_size), pad_token, dtype=torch.long, device=device)
        generated[0].fill_(bos_token)

        # Hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_length, length_penalty, early_stopping) for _ in range(batch_size)]

        # Beam scores
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        if not stochastic:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # Done flags
        done = [False for _ in range(batch_size)]
        cur_len = 1

        # --- Fix 1: Run encoder once and cache memory (reused every beam step) ---
        with torch.no_grad():
            src_key_padding_mask = self.create_padding_mask(src_rep, pad_token)
            src_emb = self.src_embedding(src_rep) * math.sqrt(self.embedding_dim)
            src_emb = self.src_pos_encoding(src_emb)
            src_emb = self.dropout(src_emb)
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        while cur_len < max_length:
            # Prepare decoder input
            input_ids = generated[:cur_len, :].transpose(0, 1)  # [batch_size * beam_size, cur_len]
            # Decoder expects [batch, seq_len]
            tgt_mask = self.create_causal_mask(cur_len).to(device)
            # Forward pass — decoder only, encoder memory is already computed
            with torch.no_grad():
                tgt_key_padding_mask = self.create_padding_mask(input_ids, pad_token)
                tgt_emb = self.tgt_embedding(input_ids) * math.sqrt(self.embedding_dim)
                tgt_emb = self.tgt_pos_encoding(tgt_emb)
                tgt_emb = self.dropout(tgt_emb)
                output = self.transformer.decoder(
                    tgt_emb, memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                logits = self.output_projection(output)  # [batch_size * beam_size, cur_len, vocab_size]
                scores = logits[:, -1, :]  # [batch_size * beam_size, vocab_size]

            if stochastic:
                scores = scores / temperature
                log_probs = torch.log_softmax(scores, dim=-1)
                probs = torch.exp(log_probs)
                # Nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cum_probs < nucl_p
                # Always include the first token
                nucleus_mask[:, 0] = True
                filtered_probs = sorted_probs * nucleus_mask.float()
                filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
                # Sample next tokens
                next_tokens = torch.multinomial(filtered_probs, num_samples=1, generator=rng_gen)
                next_words = sorted_indices.gather(1, next_tokens)
                next_scores = log_probs.gather(1, next_words) + beam_scores[:, None]
                next_scores = next_scores.view(batch_size, beam_size)
                next_words = next_words.view(batch_size, beam_size)
            else:
                log_probs = torch.log_softmax(scores, dim=-1)
                _scores = log_probs + beam_scores[:, None].expand_as(log_probs)
                _scores = _scores.view(batch_size, beam_size * n_words)
                next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)

            # Prepare next beam
            next_batch_beam = []
            for sent_id in range(batch_size):
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, pad_token, 0)] * beam_size)
                    continue
                next_sent_beam = []
                for i, (idx, value) in enumerate(zip(next_words[sent_id], next_scores[sent_id])):
                    if stochastic:
                        beam_id = i
                        word_id = idx.item()
                    else:
                        beam_id = idx.item() // n_words
                        word_id = idx.item() % n_words
                    if word_id == eos_token or cur_len + 1 == max_length:
                        hyp = generated[:cur_len, sent_id * beam_size + beam_id].clone().cpu()
                        generated_hyps[sent_id].add(hyp, value.item())
                    else:
                        next_sent_beam.append((value.item(), word_id, sent_id * beam_size + beam_id))
                    if len(next_sent_beam) == beam_size:
                        break
                if not stochastic:
                    assert len(next_sent_beam) == 0 if cur_len + 1 == max_length else len(next_sent_beam) == beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token, 0)] * beam_size
                if stochastic and len(next_sent_beam) < beam_size:
                    next_sent_beam.extend([(-1e9, pad_token, 0)] * (beam_size - len(next_sent_beam)))
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)
            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new_tensor([x[0] for x in next_batch_beam])
            beam_words = generated.new_tensor([x[1] for x in next_batch_beam])
            beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=device, dtype=torch.long)
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            cur_len += 1
            if all(done):
                break
        # Select best hypotheses
        tgt_len = torch.zeros(batch_size, dtype=torch.long)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)
        decoded = torch.full((tgt_len.max().item(), batch_size), pad_token, dtype=torch.long, device=device)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = eos_token
        return decoded, tgt_len, generated_hyps


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        self.max_len = max_len - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9
    def __len__(self):
        return len(self.hyp)
    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / (len(hyp) ** self.length_penalty)
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
    def is_done(self, best_sum_logprobs):
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / (self.max_len ** self.length_penalty)


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
            - decoding_method: 'greedy', 'beam', or 'nucleus'
            - beam_size: Number of beams for beam/nucleus search
            - p_nucleus: Nucleus cutoff probability for nucleus sampling
            - temperature_nucleus: Temperature for nucleus sampling
    
    Returns:
        TransformerRegressor model
    """
    return TransformerRegressor(vocab_size=vocab_size, **hyperparams)


def train_step(
    model,
    batch,
    criterion,
    optimizer,
    *,
    scaler=None,
    amp_enabled=False,
    amp_dtype=torch.float16,
    grad_clip=None,
):
    """Single training step"""
    model.train()
    
    # Move batch to model's device
    src = batch['input'].to(model.device, non_blocking=True)  # scrambled sequences
    tgt = batch['target'].to(model.device, non_blocking=True)  # simple sequences
    
    # Prepare target input and output
    tgt_input = tgt[:, :-1]  # All but last token
    tgt_output = tgt[:, 1:]  # All but first token (shifted)
    
    # Forward pass
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=model.device.type,
        dtype=amp_dtype,
        enabled=amp_enabled and model.device.type == "cuda",
    ):
        output = model(src, tgt_input)

        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        # Calculate loss (ignore padding tokens)
        loss = criterion(output, tgt_output)
    
    # Calculate accuracy (ignore padding tokens)
    with torch.no_grad():
        preds = torch.argmax(output, dim=-1)
        mask = tgt_output != 0  # 0 is pad token
        correct = (preds == tgt_output) & mask
        accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    
    # Backward pass
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    
    return loss.item(), accuracy


def validate_step(model, batch, criterion, *, amp_enabled=False, amp_dtype=torch.float16):
    """Single validation step"""
    model.eval()
    
    with torch.no_grad():
        # Move batch to model's device
        src = batch['input'].to(model.device, non_blocking=True)
        tgt = batch['target'].to(model.device, non_blocking=True)
        
        # Prepare target input and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        with torch.amp.autocast(
            device_type=model.device.type,
            dtype=amp_dtype,
            enabled=amp_enabled and model.device.type == "cuda",
        ):
            # Forward pass
            output = model(src, tgt_input)

            # Reshape for loss calculation
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt_output)
        
        # Calculate accuracy (ignore padding tokens)
        preds = torch.argmax(output, dim=-1)
        mask = tgt_output != 0  # 0 is pad token
        correct = (preds == tgt_output) & mask
        accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    
    return loss.item(), accuracy


def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    epochs,
    run_name='default_run',
    early_stopping_patience=None,
    early_stopping_min_delta=1e-4,
    save_models=True,
    amp_enabled=False,
    amp_dtype=torch.float16,
    scaler=None,
    start_epoch=0,
    initial_best_val_loss=None,
    scheduler=None,
    grad_clip=None,
):
    """Trains a transformer model with optional checkpointing after each epoch and optional early stopping.

    Performs training and validation loops for the specified number of epochs,
    optionally saving the model after each epoch (in 'models/' directory) and deleting
    the previous epoch's checkpoint. Tracks and returns training/validation losses.
    Optionally implements early stopping based on validation loss.

    Args:
        model (torch.nn.Module): Transformer model to be trained
        optimizer (torch.optim.Optimizer): Optimizer for training (e.g. AdamW)
        criterion (callable): Loss function (e.g. nn.CrossEntropyLoss)
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        epochs (int): Number of complete passes through the training data
        run_name (str): Name for this training run (for saving models)
        early_stopping_patience (int, optional): Number of epochs with no improvement after which training will stop
        early_stopping_min_delta (float): Minimum change to qualify as an improvement
        save_models (bool): Whether to save model checkpoints during training

    Returns:
        tuple: Four lists containing:
            - train_losses (list[float]): Average training loss per epoch
            - val_losses (list[float]): Average validation loss per epoch
            - train_accuracies (list[float]): Average training accuracy per epoch
            - val_accuracies (list[float]): Average validation accuracy per epoch
    """
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    # Early stopping variables
    best_val_loss = float('inf') if initial_best_val_loss is None else initial_best_val_loss
    patience_counter = 0
    best_epoch = start_epoch
    
    # Create models directory if saving is enabled
    if save_models:
        output_dir = os.path.join('models', run_name)
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up a single global progress bar across all epochs (train + val batches)
    steps_per_epoch = len(train_loader) + (len(val_loader) if val_loader is not None else 0)
    total_steps = epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, desc=f"Training {run_name}", unit="batch", dynamic_ncols=True)

    for local_epoch in range(epochs):
        epoch = start_epoch + local_epoch
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, batch in enumerate(train_loader, 1):
            loss, acc = train_step(
                model,
                batch,
                criterion,
                optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                grad_clip=grad_clip,
            )
            train_loss += loss
            train_acc += acc
            pbar.set_postfix(
                epoch=f"{epoch + 1}/{start_epoch + epochs}",
                phase="train",
                loss=f"{train_loss / batch_idx:.4f}",
                acc=f"{train_acc / batch_idx:.4f}",
            )
            pbar.update(1)
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, 1):
                loss, acc = validate_step(
                    model,
                    batch,
                    criterion,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                val_loss += loss
                val_acc += acc
                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{start_epoch + epochs}",
                    phase="val",
                    loss=f"{val_loss / batch_idx:.4f}",
                    acc=f"{val_acc / batch_idx:.4f}",
                )
                pbar.update(1)
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = train_acc / len(train_loader)
        epoch_val_acc = val_acc / len(val_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        if scheduler is not None:
            scheduler.step()

        # Early stopping logic
        if early_stopping_patience is not None:
            if epoch_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                # Save best model (only if saving is enabled)
                if save_models:
                    best_model_path = os.path.join(output_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'train_loss': epoch_train_loss,
                        'val_loss': epoch_val_loss,
                        'model_args': model.model_hyperparams,
                    }, best_model_path)
                    print(f"New best model saved at epoch {epoch + 1} with val_loss: {epoch_val_loss:.4f}")
                else:
                    print(f"New best validation loss at epoch {epoch + 1}: {epoch_val_loss:.4f} (model not saved)")
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered! Best model was at epoch {best_epoch} with val_loss: {best_val_loss:.4f}")
                    # finalize progress bar gracefully
                    try:
                        pbar.total = pbar.n
                        pbar.refresh()
                    except Exception:
                        pass
                    break
        
        # Delete previous epoch's model and save current model (only if saving is enabled)
        if save_models:
            # Delete previous epoch's model if it exists
            if epoch > 0:
                prev_model_path = os.path.join(output_dir, f'model_epoch_{epoch}.pt')
                if os.path.exists(prev_model_path):
                    os.remove(prev_model_path)
            
            # Save current model
            model_save_path = os.path.join(output_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'model_args': model.model_hyperparams,
            }, model_save_path)
            
            print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
                  f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f} | Model saved to {model_save_path}")
        else:
            print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
                  f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")
    
    # Close global progress bar
    try:
        pbar.close()
    except Exception:
        pass

    # Final summary
    if early_stopping_patience is not None:
        print(f"\nTraining completed. Best model: epoch {best_epoch}, val_loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def load_transformer_model(model_class, model_path, optimizer=None, device='cpu'):
    """
    Load a saved transformer model checkpoint.
    
    Args:
        model_class: The model class (needed to initialize empty model)
        model_path: Path to the .pt checkpoint file
        optimizer: Optional optimizer to load state into
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model with trained weights
        optimizer: Optimizer with saved state (if provided)
        epoch: The epoch number when saved
        train_loss: Training loss at that epoch
        val_loss: Validation loss at that epoch
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
        
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Verify CUDA availability if requested
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model - override device in model_args to ensure compatibility
    model_args = checkpoint['model_args'].copy()
    model_args['device'] = device  # Override saved device with current device
    model = model_class(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize optimizer if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Extract training info
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    return {
        'model': model,
        'optimizer': optimizer if optimizer else None,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    }


def decode_with_model(model, src, max_length, decoding_method='greedy', beam_size=5, 
                     p_nucleus=0.9, temperature_nucleus=1.0, bos_token=2, eos_token=3, pad_token=0):
    """
    Decode sequences using various decoding methods.
    
    Args:
        model: Transformer model
        src: Source sequences [batch_size, seq_len]
        max_length: Maximum generation length
        decoding_method: 'greedy', 'beam', or 'nucleus'
        beam_size: Number of beams for beam/nucleus search
        p_nucleus: Nucleus probability cutoff
        temperature_nucleus: Temperature for nucleus sampling
        bos_token, eos_token, pad_token: Special tokens
        
    Returns:
        tuple: (decoded_sequences, beam_hypotheses_or_None)
    """
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


def clean_seq(arr, pad_token=0, eos_token=3):
    """
    Remove PAD tokens and truncate at first EOS (inclusive) for fair comparison/printing.
    
    Args:
        arr: Token sequence
        pad_token: Padding token to remove
        eos_token: End-of-sequence token to stop at
        
    Returns:
        list: Cleaned token sequence
    """
    out = []
    for x in arr:
        if x == pad_token:
            continue
        out.append(int(x))
        if x == eos_token:
            break
    return out
