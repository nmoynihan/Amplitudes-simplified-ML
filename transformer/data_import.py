'''Functions for importing a pre-processing data'''
import math
import os
import random
import ast
from collections.abc import Iterator

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset

class TransformerDataset(Dataset):
    def __init__(self, csv_file, max_length=None, dynamic_padding=False):
        """
        Dataset for transformer training from CSV file(s)
        
        Args:
            csv_file: Path to CSV file with 'simple' and 'scrambled' columns,
                     or list of paths to multiple CSV files (will be combined and shuffled)
            max_length: Maximum sequence length for padding (if None, uses max length in dataset)
            dynamic_padding: If True, return unpadded examples and let the DataLoader
                     collate function pad only to each batch's max length.
        """
        self.BOS_TOKEN = 2
        self.EOS_TOKEN = 3
        self.PAD_TOKEN = 0
        self.dynamic_padding = dynamic_padding
        
        # Handle both single file and list of files
        if isinstance(csv_file, list):
            # Load and combine multiple CSV files
            dfs = []
            for file in csv_file:
                df = pd.read_csv(file)
                dfs.append(df)
            self.data = pd.concat(dfs, ignore_index=True)
            # Shuffle the combined data
            self.data = self.data.sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(f"Loaded and shuffled {len(csv_file)} CSV files with {len(self.data)} total examples")
        else:
            # Single file
            self.data = pd.read_csv(csv_file)
        
        # Parse string representations of lists into actual lists
        self.simple_sequences = []
        self.scrambled_sequences = []
        
        for _, row in self.data.iterrows():
            # Parse the string representations of lists
            simple = ast.literal_eval(row['simple'])
            scrambled = ast.literal_eval(row['scrambled'])
            
            # Add BOS and EOS tokens
            simple_with_tokens = [self.BOS_TOKEN] + simple + [self.EOS_TOKEN]
            scrambled_with_tokens = [self.BOS_TOKEN] + scrambled + [self.EOS_TOKEN]
            
            self.simple_sequences.append(simple_with_tokens)
            self.scrambled_sequences.append(scrambled_with_tokens)
        
        # Determine max length for fixed padding. In dynamic mode, max_length is
        # only an optional truncation cap.
        if dynamic_padding:
            self.max_length = max_length
        elif max_length is None:
            self.max_length = max(
                max(len(seq) for seq in self.simple_sequences),
                max(len(seq) for seq in self.scrambled_sequences)
            )
        else:
            self.max_length = max_length
    
    def __len__(self):
        return len(self.simple_sequences)
    
    def __getitem__(self, idx):
        simple_seq = self.truncate_sequence(self.simple_sequences[idx])
        scrambled_seq = self.truncate_sequence(self.scrambled_sequences[idx])

        if self.dynamic_padding:
            return {
                'input': torch.tensor(scrambled_seq, dtype=torch.long),
                'target': torch.tensor(simple_seq, dtype=torch.long),
                'input_length': len(scrambled_seq),
                'target_length': len(simple_seq)
            }
        
        # Pad sequences to max_length
        simple_padded = self.pad_sequence(simple_seq)
        scrambled_padded = self.pad_sequence(scrambled_seq)
        
        return {
            'input': torch.tensor(scrambled_padded, dtype=torch.long),  # scrambled is input
            'target': torch.tensor(simple_padded, dtype=torch.long),    # simple is target/output
            'input_length': len(scrambled_seq),
            'target_length': len(simple_seq)
        }

    def sequence_length(self, idx):
        simple_len = len(self.simple_sequences[idx])
        scrambled_len = len(self.scrambled_sequences[idx])
        if self.max_length is not None:
            simple_len = min(simple_len, self.max_length)
            scrambled_len = min(scrambled_len, self.max_length)
        return max(simple_len, scrambled_len)

    def truncate_sequence(self, sequence):
        if self.max_length is not None and len(sequence) > self.max_length:
            return sequence[:self.max_length]
        return sequence
    
    def pad_sequence(self, sequence):
        """Pad sequence to max_length with PAD_TOKEN"""
        if len(sequence) >= self.max_length:
            return sequence[:self.max_length]
        else:
            return sequence + [self.PAD_TOKEN] * (self.max_length - len(sequence))


def _pad_1d(seq, target_len, pad_token):
    if seq.numel() >= target_len:
        return seq[:target_len]
    return torch.cat(
        [seq, torch.full((target_len - seq.numel(),), pad_token, dtype=seq.dtype)]
    )


def dynamic_pad_collate(batch, pad_token=0):
    max_input_len = max(item['input'].numel() for item in batch)
    max_target_len = max(item['target'].numel() for item in batch)
    return {
        'input': torch.stack([_pad_1d(item['input'], max_input_len, pad_token) for item in batch]),
        'target': torch.stack([_pad_1d(item['target'], max_target_len, pad_token) for item in batch]),
        'input_length': torch.tensor([item['input_length'] for item in batch], dtype=torch.long),
        'target_length': torch.tensor([item['target_length'] for item in batch], dtype=torch.long),
    }


class BucketBatchSampler(Sampler[list[int]]):
    """Batch indices with similar sequence lengths to reduce padding."""

    def __init__(
        self,
        lengths,
        batch_size,
        *,
        shuffle=True,
        bucket_size_multiplier=100,
        seed=42,
        drop_last=False,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start:batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
            self.epoch += 1

        yield from batches

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


def _subset_lengths(subset):
    if not isinstance(subset, Subset):
        return [subset.sequence_length(i) for i in range(len(subset))]
    return [subset.dataset.sequence_length(i) for i in subset.indices]


def load_and_prepare_data(
    csv_file,
    batch_size=32,
    max_length=None,
    train_split=0.8,
    dynamic_padding=True,
    bucketing=True,
    bucket_size_multiplier=100,
    num_workers=0,
    pin_memory=False,
):
    """
    Load data and create train/validation splits
    
    Args:
        csv_file: Path to CSV file (or list of paths to multiple CSV files)
        batch_size: Batch size for training
        max_length: Maximum sequence length
        train_split: Fraction of data to use for training
        dynamic_padding: Pad to per-batch lengths instead of global dataset max.
        bucketing: Group similarly sized examples together when dynamic_padding is enabled.
        bucket_size_multiplier: Number of batches per shuffled length bucket.
        num_workers: DataLoader worker processes.
        pin_memory: Pin DataLoader memory for faster CUDA host-to-device copies.
    
    Returns:
        train_loader, val_loader
    """
    # Normalize paths to handle both 'data/file.csv' and 'folder/file.csv' formats
    data_dir = os.path.join(os.getcwd(), 'data')
    
    if isinstance(csv_file, list):
        # Multiple files: normalize each path
        normalized_files = []
        for file in csv_file:
            if not os.path.isabs(file):
                # Relative path: join with data directory
                full_path = os.path.join(data_dir, file)
            else:
                full_path = file
            normalized_files.append(full_path)
        csv_file = normalized_files
    else:
        # Single file: normalize path
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(data_dir, csv_file)
    
    # Load full dataset
    full_dataset = TransformerDataset(csv_file, max_length, dynamic_padding=dynamic_padding)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    collate_fn = dynamic_pad_collate if dynamic_padding else None

    if dynamic_padding and bucketing:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=BucketBatchSampler(
                _subset_lengths(train_dataset),
                batch_size,
                shuffle=True,
                bucket_size_multiplier=bucket_size_multiplier,
            ),
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=BucketBatchSampler(
                _subset_lengths(val_dataset),
                batch_size,
                shuffle=False,
                bucket_size_multiplier=bucket_size_multiplier,
            ),
            collate_fn=collate_fn,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Load data
    filename = 'w1.csv'
    csv_file = os.getcwd()+'/'+filename
    
    # Set up the dataloaders
    train_loader, val_loader = load_and_prepare_data(
        csv_file, 
        batch_size=16, 
        max_length=None, 
        train_split=0.8
    )
    
    # Test the data loader
    print("\nSample batch:")
    for batch in train_loader:
        print(f"Input shape: {batch['input'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Input sample: {batch['input'][0][:20]}...")  # First 20 tokens
        print(f"Target sample: {batch['target'][0][:20]}...")  # First 20 tokens
        break
