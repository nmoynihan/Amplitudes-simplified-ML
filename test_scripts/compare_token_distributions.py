#!/usr/bin/env python3
"""
Script to compare token distributions between two tokenized datasets.

This script:
1. Reads two tokenized CSV files (default: gi_5pt_tok.csv and relabM_alt_5pt_tok.csv)
2. Extracts all integers from token vectors in 'simple' and 'scrambled' columns
3. Computes normalized frequency distributions for each column in each file
4. Plots all distributions on the same figure for comparison
5. Prints statistics about average sequence lengths for each column/file

Usage:
    python compare_token_distributions.py
    python compare_token_distributions.py file1.csv file2.csv
    
Example:
    python compare_token_distributions.py
    python compare_token_distributions.py data/gi_4pt_tok.csv data/relabM_alt_4pt_tok.csv
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def get_sequence_lengths(df: pd.DataFrame, column_name: str) -> List[int]:
    """
    Get the length of each sequence in a column.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column
    
    Returns:
        List of sequence lengths
    """
    lengths = []
    
    for idx, row in df.iterrows():
        token_vector = parse_token_vector(row[column_name])
        lengths.append(len(token_vector))
    
    return lengths


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


def plot_comparison(distributions: Dict[str, Dict[int, float]], 
                    output_file: str = None,
                    show_plot: bool = True):
    """
    Plot all distributions on the same figure for comparison.
    
    Args:
        distributions: Dict mapping label to normalized frequency distribution
                      e.g., {'File1 Simple': {...}, 'File1 Scrambled': {...}, ...}
        output_file: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    # Get all unique tokens across all distributions
    all_tokens = sorted(set(token for dist in distributions.values() for token in dist.keys()))
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Define colors and styles for each distribution
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
    alphas = [0.7, 0.7, 0.7, 0.7]
    
    # Calculate bar positions
    n_distributions = len(distributions)
    bar_width = 0.8 / n_distributions
    x_pos = np.arange(len(all_tokens))
    
    # Plot each distribution
    for i, (label, dist) in enumerate(distributions.items()):
        freqs = [dist.get(token, 0) for token in all_tokens]
        offset = (i - n_distributions/2 + 0.5) * bar_width
        
        ax.bar(x_pos + offset, freqs, bar_width, 
               label=label, alpha=alphas[i % len(alphas)], 
               color=colors[i % len(colors)])
    
    # Customize plot
    ax.set_xlabel('Token Integer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Token Distribution Comparison Across Datasets', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_tokens, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    # Show if requested
    if show_plot:
        plt.show()
    
    plt.close()


def print_length_statistics(lengths: List[int], label: str):
    """
    Print statistics about sequence lengths.
    
    Args:
        lengths: List of sequence lengths
        label: Label for this dataset/column
    """
    print(f"\n  {label}:")
    print(f"    Average length: {np.mean(lengths):.2f}")
    print(f"    Median length:  {np.median(lengths):.0f}")
    print(f"    Min length:     {np.min(lengths)}")
    print(f"    Max length:     {np.max(lengths)}")
    print(f"    Std deviation:  {np.std(lengths):.2f}")


def print_token_statistics(tokens: List[int], label: str, counts: Dict[int, int], 
                           normalized: Dict[int, float]):
    """Print statistics about the token distribution."""
    print(f"\n  {label}:")
    print(f"    Total tokens:  {len(tokens):,}")
    print(f"    Unique tokens: {len(counts)}")
    print(f"    Token range:   [{min(counts.keys())}, {max(counts.keys())}]")
    
    # Top 5 most frequent tokens
    print(f"    Top 5 tokens:  ", end='')
    top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(", ".join([f"{token}({normalized[token]:.3f})" for token, count in top5]))


def main():
    """Main function to run the comparison analysis."""
    # Parse command line arguments
    if len(sys.argv) > 3:
        print("\nUsage: python compare_token_distributions.py [file1] [file2]")
        print("  If files are not provided, defaults to:")
        print("    file1: data/gi_5pt_tok.csv")
        print("    file2: data/relabM_alt_5pt_tok.csv")
        print("\nExample:")
        print("  python compare_token_distributions.py")
        print("  python compare_token_distributions.py data/gi_4pt_tok.csv data/relabM_alt_4pt_tok.csv")
        sys.exit(1)
    
    # Determine file paths
    if len(sys.argv) == 3:
        filepath1 = Path(sys.argv[1])
        filepath2 = Path(sys.argv[2])
    else:
        # Default files
        base_dir = Path(__file__).parent.parent / "data"
        filepath1 = base_dir / "gi_5pt_tok.csv"
        filepath2 = base_dir / "relabM_alt_5pt_tok.csv"
    
    print("\n" + "="*80)
    print("TOKEN DISTRIBUTION COMPARISON")
    print("="*80)
    print(f"\nFile 1: {filepath1}")
    print(f"File 2: {filepath2}")
    
    # Check if files exist
    if not filepath1.exists():
        print(f"\n✗ Error: File not found: {filepath1}")
        sys.exit(1)
    if not filepath2.exists():
        print(f"\n✗ Error: File not found: {filepath2}")
        sys.exit(1)
    
    # Load datasets
    print(f"\n{'='*80}")
    print("LOADING DATASETS")
    print(f"{'='*80}")
    
    print(f"\nLoading File 1: {filepath1.name}")
    try:
        df1 = pd.read_csv(filepath1)
        print(f"✓ Loaded {len(df1):,} rows")
    except Exception as e:
        print(f"\n✗ Error loading file: {e}")
        sys.exit(1)
    
    print(f"\nLoading File 2: {filepath2.name}")
    try:
        df2 = pd.read_csv(filepath2)
        print(f"✓ Loaded {len(df2):,} rows")
    except Exception as e:
        print(f"\n✗ Error loading file: {e}")
        sys.exit(1)
    
    # Check if required columns exist
    for df, fname in [(df1, filepath1.name), (df2, filepath2.name)]:
        if 'simple' not in df.columns or 'scrambled' not in df.columns:
            print(f"\n✗ Error in {fname}: Expected columns 'simple' and 'scrambled', found: {df.columns.tolist()}")
            sys.exit(1)
    
    # Extract tokens and compute distributions for File 1
    print(f"\n{'='*80}")
    print(f"ANALYZING FILE 1: {filepath1.name}")
    print(f"{'='*80}")
    
    print(f"\nExtracting tokens from 'simple' column...")
    f1_simple_tokens = extract_all_tokens(df1, 'simple')
    f1_simple_counts, f1_simple_normalized = compute_frequency_distribution(f1_simple_tokens)
    print(f"✓ Extracted {len(f1_simple_tokens):,} tokens ({len(f1_simple_counts)} unique)")
    
    print(f"\nExtracting tokens from 'scrambled' column...")
    f1_scrambled_tokens = extract_all_tokens(df1, 'scrambled')
    f1_scrambled_counts, f1_scrambled_normalized = compute_frequency_distribution(f1_scrambled_tokens)
    print(f"✓ Extracted {len(f1_scrambled_tokens):,} tokens ({len(f1_scrambled_counts)} unique)")
    
    # Extract tokens and compute distributions for File 2
    print(f"\n{'='*80}")
    print(f"ANALYZING FILE 2: {filepath2.name}")
    print(f"{'='*80}")
    
    print(f"\nExtracting tokens from 'simple' column...")
    f2_simple_tokens = extract_all_tokens(df2, 'simple')
    f2_simple_counts, f2_simple_normalized = compute_frequency_distribution(f2_simple_tokens)
    print(f"✓ Extracted {len(f2_simple_tokens):,} tokens ({len(f2_simple_counts)} unique)")
    
    print(f"\nExtracting tokens from 'scrambled' column...")
    f2_scrambled_tokens = extract_all_tokens(df2, 'scrambled')
    f2_scrambled_counts, f2_scrambled_normalized = compute_frequency_distribution(f2_scrambled_tokens)
    print(f"✓ Extracted {len(f2_scrambled_tokens):,} tokens ({len(f2_scrambled_counts)} unique)")
    
    # Get sequence lengths for all columns
    print(f"\n{'='*80}")
    print("SEQUENCE LENGTH STATISTICS")
    print(f"{'='*80}")
    
    f1_simple_lengths = get_sequence_lengths(df1, 'simple')
    f1_scrambled_lengths = get_sequence_lengths(df1, 'scrambled')
    f2_simple_lengths = get_sequence_lengths(df2, 'simple')
    f2_scrambled_lengths = get_sequence_lengths(df2, 'scrambled')
    
    print(f"\n{filepath1.name}:")
    print_length_statistics(f1_simple_lengths, "Simple column")
    print_length_statistics(f1_scrambled_lengths, "Scrambled column")
    
    print(f"\n{filepath2.name}:")
    print_length_statistics(f2_simple_lengths, "Simple column")
    print_length_statistics(f2_scrambled_lengths, "Scrambled column")
    
    # Print token distribution statistics
    print(f"\n{'='*80}")
    print("TOKEN DISTRIBUTION STATISTICS")
    print(f"{'='*80}")
    
    print(f"\n{filepath1.name}:")
    print_token_statistics(f1_simple_tokens, "Simple column", 
                          f1_simple_counts, f1_simple_normalized)
    print_token_statistics(f1_scrambled_tokens, "Scrambled column", 
                          f1_scrambled_counts, f1_scrambled_normalized)
    
    print(f"\n{filepath2.name}:")
    print_token_statistics(f2_simple_tokens, "Simple column", 
                          f2_simple_counts, f2_simple_normalized)
    print_token_statistics(f2_scrambled_tokens, "Scrambled column", 
                          f2_scrambled_counts, f2_scrambled_normalized)
    
    # Prepare distributions for plotting
    distributions = {
        f'{filepath1.stem} - Simple': f1_simple_normalized,
        f'{filepath1.stem} - Scrambled': f1_scrambled_normalized,
        f'{filepath2.stem} - Simple': f2_simple_normalized,
        f'{filepath2.stem} - Scrambled': f2_scrambled_normalized,
    }
    
    # Generate output filename for plot
    output_file = Path(__file__).parent.parent / "data" / "token_distribution_comparison.png"
    
    # Plot distributions
    print(f"\n{'='*80}")
    print("PLOTTING DISTRIBUTIONS")
    print(f"{'='*80}")
    print(f"\nGenerating comparison plot...")
    
    try:
        plot_comparison(distributions, output_file=str(output_file), show_plot=False)
        print(f"✓ Plot generated successfully")
    except Exception as e:
        print(f"✗ Error generating plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare token coverage between datasets
    print(f"\n{'='*80}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'='*80}")
    
    f1_all_tokens = set(f1_simple_counts.keys()) | set(f1_scrambled_counts.keys())
    f2_all_tokens = set(f2_simple_counts.keys()) | set(f2_scrambled_counts.keys())
    
    common_tokens = f1_all_tokens & f2_all_tokens
    only_in_f1 = f1_all_tokens - f2_all_tokens
    only_in_f2 = f2_all_tokens - f1_all_tokens
    
    print(f"\nToken coverage:")
    print(f"  Tokens in {filepath1.name}: {len(f1_all_tokens)}")
    print(f"  Tokens in {filepath2.name}: {len(f2_all_tokens)}")
    print(f"  Common tokens: {len(common_tokens)}")
    print(f"  Only in {filepath1.name}: {len(only_in_f1)}")
    if only_in_f1:
        print(f"    {sorted(only_in_f1)}")
    print(f"  Only in {filepath2.name}: {len(only_in_f2)}")
    if only_in_f2:
        print(f"    {sorted(only_in_f2)}")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
