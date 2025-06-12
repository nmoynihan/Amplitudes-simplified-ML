'''Functions for importing a pre-processing data'''
import os
import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader

class TransformerDataset(Dataset):
    def __init__(self, csv_file, max_length=None):
        """
        Dataset for transformer training from CSV file
        
        Args:
            csv_file: Path to CSV file with 'simple' and 'scrambled' columns
            max_length: Maximum sequence length for padding (if None, uses max length in dataset)
        """
        self.data = pd.read_csv(csv_file)
        self.BOS_TOKEN = 2
        self.EOS_TOKEN = 3
        self.PAD_TOKEN = 0
        
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
        
        # Determine max length for padding
        if max_length is None:
            self.max_length = max(
                max(len(seq) for seq in self.simple_sequences),
                max(len(seq) for seq in self.scrambled_sequences)
            )
        else:
            self.max_length = max_length
    
    def __len__(self):
        return len(self.simple_sequences)
    
    def __getitem__(self, idx):
        simple_seq = self.simple_sequences[idx]
        scrambled_seq = self.scrambled_sequences[idx]
        
        # Pad sequences to max_length
        simple_padded = self.pad_sequence(simple_seq)
        scrambled_padded = self.pad_sequence(scrambled_seq)
        
        return {
            'input': torch.tensor(scrambled_padded, dtype=torch.long),  # scrambled is input
            'target': torch.tensor(simple_padded, dtype=torch.long),    # simple is target/output
            'input_length': len(scrambled_seq),
            'target_length': len(simple_seq)
        }
    
    def pad_sequence(self, sequence):
        """Pad sequence to max_length with PAD_TOKEN"""
        if len(sequence) >= self.max_length:
            return sequence[:self.max_length]
        else:
            return sequence + [self.PAD_TOKEN] * (self.max_length - len(sequence))


def load_and_prepare_data(csv_file, batch_size=32, max_length=None, train_split=0.8):
    """
    Load data and create train/validation splits
    
    Args:
        csv_file: Path to CSV file
        batch_size: Batch size for training
        max_length: Maximum sequence length
        train_split: Fraction of data to use for training
    
    Returns:
        train_loader, val_loader, dataset_info
    """
    # Load full dataset
    full_dataset = TransformerDataset(csv_file, max_length)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
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