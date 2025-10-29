# Paolo's copy of test_data.py
# Done so that it can act on any csv file from the terminal

# Example usage:
# python test_data_paolo.py data/10k_set/gi_4pt_tok 4 None


import sys
import pandas as pd
import numpy as np
import ast
import argparse
import pathlib

# import the tokenizer and numerically_equivalent function
from data_generation.Tokenizer import ScatteringAmplitudeTokenizer, numerically_equivalent
tok = ScatteringAmplitudeTokenizer(max_particles=8)

def num_check_csv(input_csv: pathlib.Path,num_pts: int,num_samples: int|None) -> int:
    """Check numerical equivalence of simple,scrambled pairs in CSV file."""
    # Load CSV file
    df = pd.read_csv(input_csv)
    # Check required columns
    for col in ("simple","scrambled"):
        if col not in df.columns:
            print(f"ERROR: missing '{col}' column in {input_csv}", file=sys.stderr)
            return 2
    # Limit the number of samples if specified
    if num_samples is not None:
        df = df.head(num_samples)
        print(f"Using {len(df)} samples from CSV file (limited by num_samples={num_samples})")
    else:
        print(f"Using all {len(df)} samples from CSV file")

    # print(df)

    matches = 0
    unmatch_idxs = []
    disallowed_count = 0
    other_error_count = 0

    # Read tokens directly from DataFrame columns
    for i, row in df.iterrows():
        try:
            a_tokens = ast.literal_eval(row["simple"])
            b_tokens = ast.literal_eval(row["scrambled"])
            
            ok, _ = numerically_equivalent(
                tokenizer=tok, a_tokens=a_tokens, b_tokens=b_tokens,
                N=num_pts, samples=3, M=1.0, tol_abs=1e-8, tol_rel=1e-8,
                seed=123, return_details=True
            )
            
            if ok: matches += 1
            else:  unmatch_idxs.append(i)
        except Exception as e:
            print(f"Error at row {i}: {e}")
            if "disallowed expression" in str(e).lower():
                disallowed_count += 1
            else:
                other_error_count += 1
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1}: ok={matches}, disallowed={disallowed_count}, other_err={other_error_count}")

    
    print('###########')
    print(f"Total matches: {matches}")
    print(f"Total unmatches: {len(unmatch_idxs)}")
    print(f"Unmatched indices: {unmatch_idxs}")
    print(f"Disallowed expressions: {disallowed_count}")
    print(f"Other errors: {other_error_count}")
    print(f"Total processed: {matches + len(unmatch_idxs) + disallowed_count + other_error_count}")

    if unmatch_idxs:
        idx = unmatch_idxs[0]
        a = ast.literal_eval(df.loc[idx, "simple"])
        b = ast.literal_eval(df.loc[idx, "scrambled"])
        print("First mismatch:")
        print("  simple  :", tok.decode_infix(a))
        print("  scrambled:", tok.decode_infix(b))
    return 0 if not unmatch_idxs and other_error_count == 0 else 1


# make this file callable from command line
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Check numerical equivalence for CSV simple,scrambled token columns.")
    p.add_argument("input_csv", type=pathlib.Path)
    p.add_argument("num_pts", type=int, help="Number of external legs (N)")
    p.add_argument("--num-samples", type=int, default=None, help="Limit rows")
    args = p.parse_args()
    sys.exit(num_check_csv(args.input_csv, args.num_pts, args.num_samples))