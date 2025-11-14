#!/usr/bin/env python3
"""
Test script to analyze and visualize the distribution of token integers in tokenized datasets.

This script:
1. Reads a tokenized CSV file with 'simple' and 'scrambled' columns
2. Extracts all integers from the token vectors in each column
3. Computes normalized frequency distributions for each column
4. Plots both distributions on the same histogram for comparison

Usage:
    python test_token_distribution.py [filepath]
    
    If filepath is not provided, defaults to data/gi_4pt_tok.csv
    
Example:
    python test_token_distribution.py
    python test_token_distribution.py data/gi_5pt_tok.csv
    python test_token_distribution.py data/10k_set/gi_4pt_tok.csv
"""

# IMPORTANT: Import torch BEFORE numpy/pandas to avoid BLAS/MKL conflicts on macOS
import torch
import sys
import ast
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

# Now safe to import numpy/pandas after torch
import pandas as pd
import numpy as np
# matplotlib imported lazily in plot_distributions to avoid conflicts with PyTorch

# Add transformer directory to path
sys.path.append(str(Path(__file__).parent.parent / "transformer"))
# Lazy import transformer functions to avoid initialization issues
# from transformer_functions import load_transformer_model, decode_with_model, clean_seq, TransformerRegressor


def parse_token_vector(token_str: str) -> List[int]:
    """
    Parse a string representation of a token vector into a list of integers.
    
    Args:
        token_str: String representation like "[5, 4, 7, 6, 13, ...]"
    
    Returns:
        List of integers
    """
    try:
        # Use ast.literal_eval to safely parse the string
        return ast.literal_eval(token_str)
    except:
        # Fallback: manual parsing if ast fails
        token_str = token_str.strip('[]')
        return [int(x.strip()) for x in token_str.split(',') if x.strip()]


def extract_all_tokens(df: pd.DataFrame, column_name: str) -> List[int]:
    """
    Extract all token integers from a column of token vectors.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to extract from
    
    Returns:
        List of all token integers found
    """
    all_tokens = []
    
    for idx, row in df.iterrows():
        token_vector = parse_token_vector(row[column_name])
        all_tokens.extend(token_vector)
    
    return all_tokens


def compute_frequency_distribution(tokens: List[int]) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Compute frequency distribution of tokens.
    
    Args:
        tokens: List of token integers
    
    Returns:
        Tuple of (counts_dict, normalized_freq_dict)
    """
    # Count occurrences
    counts = Counter(tokens)
    
    # Compute normalized frequencies
    total = len(tokens)
    normalized = {token: count / total for token, count in counts.items()}
    
    return dict(counts), normalized


def plot_distributions(simple_dist: Dict[int, float], 
                       scrambled_dist: Dict[int, float],
                       predicted_dist: Dict[int, float] = None,
                       output_file: str = None,
                       show_plot: bool = True):
    """
    Plot normalized frequency distributions for both columns and optionally predictions.
    
    Args:
        simple_dist: Normalized frequency distribution for 'simple' column
        scrambled_dist: Normalized frequency distribution for 'scrambled' column
        predicted_dist: Optional normalized frequency distribution for model predictions
        output_file: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    # Lazy import matplotlib to avoid conflicts with PyTorch model loading
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Get all unique tokens across all distributions
    all_dists = [simple_dist, scrambled_dist]
    if predicted_dist:
        all_dists.append(predicted_dist)
    all_tokens = sorted(set(token for dist in all_dists for token in dist.keys()))
    
    # Create arrays for plotting
    simple_freqs = [simple_dist.get(token, 0) for token in all_tokens]
    scrambled_freqs = [scrambled_dist.get(token, 0) for token in all_tokens]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Bar width and positions
    if predicted_dist:
        bar_width = 0.25
        x_pos = np.arange(len(all_tokens))
        
        # Create bars
        bars1 = ax.bar(x_pos - bar_width, simple_freqs, bar_width, 
                       label='Simple', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x_pos, scrambled_freqs, bar_width,
                       label='Scrambled', alpha=0.8, color='coral')
        predicted_freqs = [predicted_dist.get(token, 0) for token in all_tokens]
        bars3 = ax.bar(x_pos + bar_width, predicted_freqs, bar_width,
                       label='Predicted', alpha=0.8, color='mediumseagreen')
    else:
        bar_width = 0.35
        x_pos = np.arange(len(all_tokens))
        
        # Create bars
        bars1 = ax.bar(x_pos - bar_width/2, simple_freqs, bar_width, 
                       label='Simple', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x_pos + bar_width/2, scrambled_freqs, bar_width,
                       label='Scrambled', alpha=0.8, color='coral')
    
    # Customize plot
    ax.set_xlabel('Token Integer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Frequency', fontsize=12, fontweight='bold')
    title = 'Token Distribution: Simple vs Scrambled'
    if predicted_dist:
        title += ' vs Model Predictions'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_tokens, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars (only for significant values)
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            if value > 0.01:  # Only label if frequency > 1%
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=7, rotation=0)
    
    add_value_labels(bars1, simple_freqs)
    add_value_labels(bars2, scrambled_freqs)
    if predicted_dist:
        add_value_labels(bars3, predicted_freqs)
    
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    # Show if requested
    if show_plot:
        plt.show()
    
    plt.close()


def print_statistics(tokens: List[int], column_name: str, counts: Dict[int, int], 
                     normalized: Dict[int, float]):
    """Print statistics about the token distribution."""
    print(f"\n{'='*80}")
    print(f"Statistics for '{column_name}' column:")
    print(f"{'='*80}")
    
    print(f"\nTotal tokens: {len(tokens):,}")
    print(f"Unique tokens: {len(counts)}")
    print(f"Token range: [{min(counts.keys())}, {max(counts.keys())}]")
    
    # Top 10 most frequent tokens
    print(f"\nTop 10 most frequent tokens:")
    print(f"{'Token':<10} {'Count':<15} {'Frequency':<15}")
    print(f"{'-'*40}")
    for token, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        freq = normalized[token]
        print(f"{token:<10} {count:<15,} {freq:<15.6f}")
    
    # Least frequent tokens (if there are more than 10)
    if len(counts) > 10:
        print(f"\nBottom 10 least frequent tokens:")
        print(f"{'Token':<10} {'Count':<15} {'Frequency':<15}")
        print(f"{'-'*40}")
        for token, count in sorted(counts.items(), key=lambda x: x[1])[:10]:
            freq = normalized[token]
            print(f"{token:<10} {count:<15,} {freq:<15.6f}")


def run_model_inference(n_particles: int, test_filepath: Path, device: str = None) -> List[int]:
    """
    Load trained model and run inference on test data.
    
    Args:
        n_particles: Number of particles (4 or 5)
        test_filepath: Path to test data CSV file
        device: Device to run on (defaults to cpu for stability)
    
    Returns:
        List of all predicted tokens
    """
    if device is None:
        device = 'cpu'  # Use CPU by default for stability
    
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path(__file__).parent.parent / "models" / f"model_{n_particles}pt.pt"
    print(f"\nLoading model from: {model_path}")
    
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        return None
    
    try:
        # Lazy import to avoid initialization issues
        from transformer_functions import TransformerRegressor, decode_with_model, clean_seq
        
        # Load checkpoint manually to avoid device issues
        print("Loading checkpoint...")
        checkpoint = torch.load(str(model_path), map_location='cpu')
        
        # Create model with CPU device
        print("Initializing model...")
        model_args = checkpoint['model_args'].copy()
        model_args['device'] = 'cpu'  # Override saved device
        model = TransformerRegressor(**model_args)
        
        # Load weights
        print("Loading model weights...")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model loaded successfully on CPU")
    except Exception as e:
        print(f"✗ Error loading model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load test data
    print(f"Loading test data from: {test_filepath}")
    if not test_filepath.exists():
        print(f"✗ Test file not found: {test_filepath}")
        return None
    
    try:
        # Only load first 100 rows for faster testing
        test_df = pd.read_csv(test_filepath, nrows=100)
        print(f"✓ Loaded {len(test_df):,} test samples (limited to first 100 for speed)")
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return None
    
    # Run inference
    print(f"Running greedy decoding on test set...")
    all_predicted_tokens = []
    
    # Get special tokens from model
    pad_token = model.pad_token_id
    bos_token = 2  # Typically BOS
    eos_token = 3  # Typically EOS
    
    # Determine max length (use longest sequence in test set + buffer)
    max_src_len = 0
    for idx, row in test_df.iterrows():
        src_tokens = parse_token_vector(row['simple'])
        max_src_len = max(max_src_len, len(src_tokens))
    max_length = max_src_len * 3  # Give model plenty of room
    
    batch_size = 32  # Process in batches for efficiency
    
    with torch.no_grad():
        for start_idx in range(0, len(test_df), batch_size):
            end_idx = min(start_idx + batch_size, len(test_df))
            batch_df = test_df.iloc[start_idx:end_idx]
            
            # Prepare batch
            batch_src = []
            for _, row in batch_df.iterrows():
                src_tokens = parse_token_vector(row['simple'])
                batch_src.append(src_tokens)
            
            # Pad batch
            max_len_batch = max(len(seq) for seq in batch_src)
            padded_batch = []
            for seq in batch_src:
                padded_seq = seq + [pad_token] * (max_len_batch - len(seq))
                padded_batch.append(padded_seq)
            
            # Convert to tensor
            src_tensor = torch.tensor(padded_batch, dtype=torch.long).to(device)
            
            # Generate predictions
            predictions, _ = decode_with_model(
                model, src_tensor, max_length=max_length,
                decoding_method='greedy',
                bos_token=bos_token, eos_token=eos_token, pad_token=pad_token
            )
            
            # Extract tokens from predictions
            for pred_seq in predictions:
                cleaned_seq = clean_seq(pred_seq.cpu().tolist(), pad_token=pad_token, eos_token=eos_token)
                # Remove BOS and EOS tokens for fair comparison with target
                cleaned_seq = [t for t in cleaned_seq if t not in [bos_token, eos_token]]
                all_predicted_tokens.extend(cleaned_seq)
            
            if (start_idx // batch_size + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{len(test_df)} samples...")
    
    print(f"✓ Generated predictions for {len(test_df)} samples")
    print(f"✓ Extracted {len(all_predicted_tokens):,} predicted tokens")
    
    return all_predicted_tokens


def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    if len(sys.argv) > 2:
        print("\nUsage: python test_token_distribution.py [filepath]")
        print("  If filepath is not provided, defaults to data/gi_4pt_tok.csv")
        print("\nExample:")
        print("  python test_token_distribution.py")
        print("  python test_token_distribution.py data/gi_5pt_tok.csv")
        sys.exit(1)
    
    # Determine file path
    if len(sys.argv) == 2:
        filepath = Path(sys.argv[1])
    else:
        # Default to data/gi_4pt_tok.csv
        filepath = Path(__file__).parent.parent / "data" / "gi_4pt_tok.csv"
    
    print("\n" + "="*80)
    print("TOKEN DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing file: {filepath}")
    
    # Check if file exists
    if not filepath.exists():
        print(f"\n✗ Error: File not found: {filepath}")
        sys.exit(1)
    
    # Load the dataset
    print(f"\nLoading dataset...")
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"\n✗ Error loading file: {e}")
        sys.exit(1)
    
    # Check if required columns exist
    if 'simple' not in df.columns or 'scrambled' not in df.columns:
        print(f"\n✗ Error: Expected columns 'simple' and 'scrambled', found: {df.columns.tolist()}")
        sys.exit(1)
    
    # Extract tokens from 'simple' column
    print(f"\nExtracting tokens from 'simple' column...")
    simple_tokens = extract_all_tokens(df, 'simple')
    simple_counts, simple_normalized = compute_frequency_distribution(simple_tokens)
    print(f"✓ Extracted {len(simple_tokens):,} tokens ({len(simple_counts)} unique)")
    
    # Extract tokens from 'scrambled' column
    print(f"\nExtracting tokens from 'scrambled' column...")
    scrambled_tokens = extract_all_tokens(df, 'scrambled')
    scrambled_counts, scrambled_normalized = compute_frequency_distribution(scrambled_tokens)
    print(f"✓ Extracted {len(scrambled_tokens):,} tokens ({len(scrambled_counts)} unique)")
    
    # Print statistics
    print_statistics(simple_tokens, 'simple', simple_counts, simple_normalized)
    print_statistics(scrambled_tokens, 'scrambled', scrambled_counts, scrambled_normalized)
    
    # Compare distributions
    print(f"\n{'='*80}")
    print("DISTRIBUTION COMPARISON")
    print(f"{'='*80}")
    
    # Tokens unique to each column
    simple_only = set(simple_counts.keys()) - set(scrambled_counts.keys())
    scrambled_only = set(scrambled_counts.keys()) - set(simple_counts.keys())
    common_tokens = set(simple_counts.keys()) & set(scrambled_counts.keys())
    
    print(f"\nTokens only in 'simple': {len(simple_only)}")
    if simple_only:
        print(f"  {sorted(simple_only)}")
    
    print(f"\nTokens only in 'scrambled': {len(scrambled_only)}")
    if scrambled_only:
        print(f"  {sorted(scrambled_only)}")
    
    print(f"\nCommon tokens: {len(common_tokens)}")
    
    # Try to determine n_particles from filename and run model inference
    predicted_normalized = None
    predicted_counts = None
    predicted_tokens = None
    
    # Try to extract n_particles from filename (e.g., "gi_4pt_tok.csv" -> 4)
    import re
    match = re.search(r'(\d+)pt', filepath.name)
    if match:
        n_particles = int(match.group(1))
        if n_particles in [4, 5]:
            print(f"{'='*80}")
            print(f"MODEL INFERENCE (N={n_particles})")
            print(f"{'='*80}")
            
            # Construct test file path
            test_filepath = Path(__file__).parent.parent / "data" / "test_set" / f"gi_{n_particles}pt_tok.csv"
            
            # Run model inference
            predicted_tokens = run_model_inference(n_particles, test_filepath)
            
            if predicted_tokens:
                predicted_counts, predicted_normalized = compute_frequency_distribution(predicted_tokens)
                print_statistics(predicted_tokens, 'predicted', predicted_counts, predicted_normalized)
                
                # Compare predicted with target
                print(f"\n{'='*80}")
                print("PREDICTION vs TARGET COMPARISON")
                print(f"{'='*80}")
                
                pred_only = set(predicted_counts.keys()) - set(scrambled_counts.keys())
                target_only = set(scrambled_counts.keys()) - set(predicted_counts.keys())
                common_pred_target = set(predicted_counts.keys()) & set(scrambled_counts.keys())
                
                print(f"\nTokens only in predictions: {len(pred_only)}")
                if pred_only:
                    print(f"  {sorted(pred_only)}")
                
                print(f"\nTokens only in target: {len(target_only)}")
                if target_only:
                    print(f"  {sorted(target_only)}")
                
                print(f"\nCommon tokens: {len(common_pred_target)}")
        else:
            print(f"\n⚠ N={n_particles} particles detected, but only N=4 or N=5 models are available")
    else:
        print(f"\n⚠ Could not determine n_particles from filename: {filepath.name}")
        print(f"  Skipping model inference (expected format: 'gi_Npt_tok.csv')")
    
    # Generate output filename for plot
    output_file = filepath.parent / f"{filepath.stem}_distribution.png"
    
    # Plot distributions
    print(f"\n{'='*80}")
    print("PLOTTING DISTRIBUTIONS")
    print(f"{'='*80}")
    print(f"\nGenerating plot...")
    
    try:
        plot_distributions(simple_normalized, scrambled_normalized, 
                         predicted_dist=predicted_normalized,
                         output_file=str(output_file), show_plot=True)
        print(f"✓ Plot generated successfully")
    except Exception as e:
        print(f"✗ Error generating plot: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
