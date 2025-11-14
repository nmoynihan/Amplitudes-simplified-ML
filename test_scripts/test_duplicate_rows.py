#!/usr/bin/env python3
"""
Test script to check for duplicate rows within and between tokenized datasets.

This script checks for duplicates in:
- data/gi_{n}pt_tok.csv (main dataset)
- data/10k_set/gi_{n}pt_tok.csv (10k subset)

For N = 4 or 5 particles.

It performs three types of checks:
1. Duplicates within the main dataset (both columns and first column only)
2. Duplicates within the 10k subset (both columns and first column only)
3. Duplicates between the two datasets (both columns and first column only)

Usage:
    python test_duplicate_rows.py <n_particles>
    where n_particles is 4 or 5
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Tuple, List, Dict


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load a CSV dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    return pd.read_csv(filepath)


def find_duplicates_both_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """
    Find duplicate rows based on both columns.
    
    Returns:
        Tuple of (duplicate_rows_df, list_of_indices)
    """
    # Find duplicates considering both columns
    duplicated_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicated_mask]
    duplicate_indices = duplicate_rows.index.tolist()
    
    return duplicate_rows, duplicate_indices


def find_duplicates_first_column(df: pd.DataFrame, column_name: str) -> Tuple[pd.DataFrame, List[int]]:
    """
    Find duplicate rows based on first column only.
    
    Returns:
        Tuple of (duplicate_rows_df, list_of_indices)
    """
    # Find duplicates considering only the first column
    duplicated_mask = df.duplicated(subset=[column_name], keep=False)
    duplicate_rows = df[duplicated_mask]
    duplicate_indices = duplicate_rows.index.tolist()
    
    return duplicate_rows, duplicate_indices


def find_duplicates_second_column(df: pd.DataFrame, column_name: str) -> Tuple[pd.DataFrame, List[int]]:
    """
    Find duplicate rows based on second column only.
    
    Returns:
        Tuple of (duplicate_rows_df, list_of_indices)
    """
    # Find duplicates considering only the second column
    duplicated_mask = df.duplicated(subset=[column_name], keep=False)
    duplicate_rows = df[duplicated_mask]
    duplicate_indices = duplicate_rows.index.tolist()
    
    return duplicate_rows, duplicate_indices


def find_duplicates_between_datasets(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    column_name: str = None,
    both_columns: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Find rows that appear in both datasets using efficient merge.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        column_name: Column name to compare (for single column comparison)
        both_columns: If True, compare both columns; if False, only specified column
    
    Returns:
        Tuple of (indices_in_df1, indices_in_df2)
    """
    # Add temporary index columns to track original indices
    df1_temp = df1.copy()
    df2_temp = df2.copy()
    df1_temp['_idx1'] = df1_temp.index
    df2_temp['_idx2'] = df2_temp.index
    
    if both_columns:
        # Compare both columns using merge
        merged = df1_temp.merge(
            df2_temp, 
            on=['simple', 'scrambled'], 
            how='inner'
        )
        indices_df1 = merged['_idx1'].tolist()
        indices_df2 = merged['_idx2'].tolist()
    else:
        # Compare only specified column
        merged = df1_temp.merge(
            df2_temp, 
            on=[column_name], 
            how='inner'
        )
        indices_df1 = merged['_idx1'].tolist()
        indices_df2 = merged['_idx2'].tolist()
    
    return indices_df1, indices_df2


def print_separator(char: str = "=", length: int = 80):
    """Print a separator line."""
    print(char * length)


def test_dataset_duplicates(n_particles: int):
    """Test for duplicates in datasets for n_particles (4 or 5)."""
    
    print_separator()
    print(f"TESTING {n_particles}-PARTICLE DATASETS")
    print_separator()
    print()
    
    # Define file paths
    base_path = Path(__file__).parent.parent / "data"
    main_file = base_path / f"gi_{n_particles}pt_tok.csv"
    subset_file = base_path / "10k_set" / f"gi_{n_particles}pt_tok.csv"
    
    # Load datasets
    print(f"Loading datasets...")
    try:
        df_main = load_dataset(main_file)
        df_subset = load_dataset(subset_file)
        print(f"✓ Main dataset: {len(df_main)} rows")
        print(f"✓ 10k subset: {len(df_subset)} rows")
        print()
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    # ========================================================================
    # 1. Check for duplicates within main dataset
    # ========================================================================
    print_separator("-")
    print("1. DUPLICATES WITHIN MAIN DATASET")
    print_separator("-")
    
    # Both columns
    dup_main_both, indices_main_both = find_duplicates_both_columns(df_main)
    print(f"\n1a. Duplicates (both columns):")
    print(f"    Number of duplicate rows: {len(dup_main_both)}")
    if len(dup_main_both) > 0:
        print(f"    Indices: {sorted(indices_main_both)}")
        # Group duplicates
        unique_duplicates = df_main[df_main.duplicated(keep=False)].drop_duplicates()
        print(f"    Number of unique duplicate patterns: {len(unique_duplicates)}")
    else:
        print(f"    ✓ No duplicates found")
    
    # First column only
    dup_main_first, indices_main_first = find_duplicates_first_column(df_main, 'simple')
    print(f"\n1b. Duplicates (first column 'simple' only):")
    print(f"    Number of duplicate rows: {len(dup_main_first)}")
    if len(dup_main_first) > 0:
        print(f"    Indices: {sorted(indices_main_first)}")
        # Group duplicates
        unique_duplicates = df_main[df_main.duplicated(subset=['simple'], keep=False)].drop_duplicates(subset=['simple'])
        print(f"    Number of unique duplicate patterns: {len(unique_duplicates)}")
    else:
        print(f"    ✓ No duplicates found")
    
    # Second column only
    dup_main_second, indices_main_second = find_duplicates_second_column(df_main, 'scrambled')
    print(f"\n1c. Duplicates (second column 'scrambled' only):")
    print(f"    Number of duplicate rows: {len(dup_main_second)}")
    if len(dup_main_second) > 0:
        print(f"    Indices: {sorted(indices_main_second)}")
        # Group duplicates
        unique_duplicates = df_main[df_main.duplicated(subset=['scrambled'], keep=False)].drop_duplicates(subset=['scrambled'])
        print(f"    Number of unique duplicate patterns: {len(unique_duplicates)}")
    else:
        print(f"    ✓ No duplicates found")
    
    print()
    
    # ========================================================================
    # 2. Check for duplicates within 10k subset
    # ========================================================================
    print_separator("-")
    print("2. DUPLICATES WITHIN 10K SUBSET")
    print_separator("-")
    
    # Both columns
    dup_subset_both, indices_subset_both = find_duplicates_both_columns(df_subset)
    print(f"\n2a. Duplicates (both columns):")
    print(f"    Number of duplicate rows: {len(dup_subset_both)}")
    if len(dup_subset_both) > 0:
        print(f"    Indices: {sorted(indices_subset_both)}")
        # Group duplicates
        unique_duplicates = df_subset[df_subset.duplicated(keep=False)].drop_duplicates()
        print(f"    Number of unique duplicate patterns: {len(unique_duplicates)}")
    else:
        print(f"    ✓ No duplicates found")
    
    # First column only
    dup_subset_first, indices_subset_first = find_duplicates_first_column(df_subset, 'simple')
    print(f"\n2b. Duplicates (first column 'simple' only):")
    print(f"    Number of duplicate rows: {len(dup_subset_first)}")
    if len(dup_subset_first) > 0:
        print(f"    Indices: {sorted(indices_subset_first)}")
        # Group duplicates
        unique_duplicates = df_subset[df_subset.duplicated(subset=['simple'], keep=False)].drop_duplicates(subset=['simple'])
        print(f"    Number of unique duplicate patterns: {len(unique_duplicates)}")
    else:
        print(f"    ✓ No duplicates found")
    
    # Second column only
    dup_subset_second, indices_subset_second = find_duplicates_second_column(df_subset, 'scrambled')
    print(f"\n2c. Duplicates (second column 'scrambled' only):")
    print(f"    Number of duplicate rows: {len(dup_subset_second)}")
    if len(dup_subset_second) > 0:
        print(f"    Indices: {sorted(indices_subset_second)}")
        # Group duplicates
        unique_duplicates = df_subset[df_subset.duplicated(subset=['scrambled'], keep=False)].drop_duplicates(subset=['scrambled'])
        print(f"    Number of unique duplicate patterns: {len(unique_duplicates)}")
    else:
        print(f"    ✓ No duplicates found")
    
    print()
    
    # ========================================================================
    # 3. Check for duplicates between datasets
    # ========================================================================
    print_separator("-")
    print("3. DUPLICATES BETWEEN MAIN DATASET AND 10K SUBSET")
    print_separator("-")
    
    # Both columns
    print(f"\n3a. Duplicates (both columns):")
    print(f"    Searching for common rows between datasets...")
    indices_main_cross_both, indices_subset_cross_both = find_duplicates_between_datasets(
        df_main, df_subset, both_columns=True
    )
    print(f"    Number of common rows: {len(indices_main_cross_both)}")
    if len(indices_main_cross_both) > 0:
        print(f"    Main dataset indices: {sorted(indices_main_cross_both)}")
        print(f"    10k subset indices: {sorted(indices_subset_cross_both)}")
    else:
        print(f"    ✓ No common rows found")
    
    # First column only
    print(f"\n3b. Duplicates (first column 'simple' only):")
    print(f"    Searching for common 'simple' values between datasets...")
    indices_main_cross_first, indices_subset_cross_first = find_duplicates_between_datasets(
        df_main, df_subset, column_name='simple', both_columns=False
    )
    print(f"    Number of common 'simple' values: {len(indices_main_cross_first)}")
    if len(indices_main_cross_first) > 0:
        print(f"    Main dataset indices: {sorted(indices_main_cross_first)}")
        print(f"    10k subset indices: {sorted(indices_subset_cross_first)}")
    else:
        print(f"    ✓ No common 'simple' values found")
    
    # Second column only
    print(f"\n3c. Duplicates (second column 'scrambled' only):")
    print(f"    Searching for common 'scrambled' values between datasets...")
    indices_main_cross_second, indices_subset_cross_second = find_duplicates_between_datasets(
        df_main, df_subset, column_name='scrambled', both_columns=False
    )
    print(f"    Number of common 'scrambled' values: {len(indices_main_cross_second)}")
    if len(indices_main_cross_second) > 0:
        print(f"    Main dataset indices: {sorted(indices_main_cross_second)}")
        print(f"    10k subset indices: {sorted(indices_subset_cross_second)}")
    else:
        print(f"    ✓ No common 'scrambled' values found")
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("-")
    print("SUMMARY")
    print_separator("-")
    print(f"\nTotal duplicate issues found:")
    total_issues = (
        (len(dup_main_both) > 0) +
        (len(dup_main_first) > 0) +
        (len(dup_main_second) > 0) +
        (len(dup_subset_both) > 0) +
        (len(dup_subset_first) > 0) +
        (len(dup_subset_second) > 0) +
        (len(indices_main_cross_both) > 0) +
        (len(indices_main_cross_first) > 0) +
        (len(indices_main_cross_second) > 0)
    )
    print(f"  - Main dataset (both columns): {len(dup_main_both)} duplicate rows")
    print(f"  - Main dataset (first column): {len(dup_main_first)} duplicate rows")
    print(f"  - Main dataset (second column): {len(dup_main_second)} duplicate rows")
    print(f"  - 10k subset (both columns): {len(dup_subset_both)} duplicate rows")
    print(f"  - 10k subset (first column): {len(dup_subset_first)} duplicate rows")
    print(f"  - 10k subset (second column): {len(dup_subset_second)} duplicate rows")
    print(f"  - Between datasets (both columns): {len(indices_main_cross_both)} common rows")
    print(f"  - Between datasets (first column): {len(indices_main_cross_first)} common rows")
    print(f"  - Between datasets (second column): {len(indices_main_cross_second)} common rows")
    print()
    
    if total_issues == 0:
        print("✓ ALL CHECKS PASSED - No duplicate issues found!")
    else:
        print(f"✗ {total_issues} DUPLICATE ISSUE(S) DETECTED")
    
    print()


def main():
    """Main function to run all tests."""
    # Check for command line argument
    if len(sys.argv) != 2:
        print("\nUsage: python test_duplicate_rows.py <n_particles>")
        print("  where n_particles is 4 or 5")
        print("\nExample:")
        print("  python test_duplicate_rows.py 4")
        print("  python test_duplicate_rows.py 5")
        sys.exit(1)
    
    try:
        n_particles = int(sys.argv[1])
        if n_particles not in [4, 5]:
            print(f"\nError: n_particles must be 4 or 5, got {n_particles}")
            sys.exit(1)
    except ValueError:
        print(f"\nError: n_particles must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)
    
    print("\n")
    print_separator("=")
    print("DUPLICATE ROWS TEST SCRIPT")
    print("Testing tokenized datasets for duplicate rows")
    print_separator("=")
    print()
    
    # Test specified n-particle datasets
    test_dataset_duplicates(n_particles)
    
    print_separator("=")
    print("TEST COMPLETE")
    print_separator("=")
    print()


if __name__ == "__main__":
    main()
