
import pandas as pd
import numpy as np
import ast

num_pts = 5
num_samples = 3  # Number of samples to use from CSV. If None, use all samples
# Load CSV file
# fpath = '../data/10k_set/'
fpath = '../data/'
df = pd.read_csv(fpath+f"gi_{num_pts}pt_tok.csv")

# Limit the number of samples if specified
if num_samples is not None:
    df = df.head(num_samples)
    print(f"Using {len(df)} samples from CSV file (limited by num_samples={num_samples})")
    print(df)
else:
    print(f"Using all {len(df)} samples from CSV file")

'''
# Parse the string representation of lists into actual Python lists
col1 = df.iloc[:, 0].apply(ast.literal_eval).tolist()
col2 = df.iloc[:, 1].apply(ast.literal_eval).tolist()

# Combine columns into one list of lists (or keep separate if you want)
all_vectors = col1 + col2

# Find maximum length
max_len = max(len(v) for v in all_vectors)

# Function to pad vectors with -1
def pad_vector(v, length, pad_value=-1):
    return v + [pad_value] * (length - len(v))

# Apply padding
padded_col1 = np.array([pad_vector(v, max_len) for v in col1])
padded_col2 = np.array([pad_vector(v, max_len) for v in col2])

# If you want them together as a single numpy array:
padded_array = np.stack([padded_col1, padded_col2], axis=1)


#%%
#print('Unique simple:',np.unique(padded_array[:,0,:],axis=0).shape)
#print('Unique pair:',np.unique(padded_array[:,:,:],axis=0).shape)
'''

#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "data_generation"))
from Tokenizer import ScatteringAmplitudeTokenizer, numerically_equivalent
tok = ScatteringAmplitudeTokenizer(max_particles=8)

matches = 0
unmatch_idxs = []
disallowed_count = 0
other_error_count = 0

# Comment out the old array-based approach
# for d_idx in range(padded_array.shape[0]):
#     try:
#         # Convert numpy arrays to Python lists and remove padding
#         a_tokens = padded_array[d_idx, 0].tolist()
#         b_tokens = padded_array[d_idx, 1].tolist()
#         
#         # Remove padding (-1 values) from the end
#         a_tokens = [t for t in a_tokens if t != -1]
#         b_tokens = [t for t in b_tokens if t != -1]

# Read tokens directly from DataFrame columns
for d_idx in range(len(df)):
    try:
        # Read tokens directly from DataFrame columns using ast.literal_eval
        a_tokens = ast.literal_eval(df.iloc[d_idx]['simple'])
        b_tokens = ast.literal_eval(df.iloc[d_idx]['scrambled'])
        
        ok, _ = numerically_equivalent(
            tokenizer=tok,
            a_tokens=a_tokens,   # Python list, no padding
            b_tokens=b_tokens,   # Python list, no padding
            N=num_pts,           # total external legs
            samples=3,           # number of random phase‑space points
            M=1.0,               # scalar mass
            tol_abs=1e-8,
            tol_rel=1e-8,
            seed=123,
            return_details=True
        )
        if ok:
            matches += 1
        else:
            unmatch_idxs.append(d_idx)
    except Exception as e:
        # Distinguish between disallowed expressions and other errors
        error_msg = str(e).lower()
        if "disallowed expression" in error_msg:
            disallowed_count += 1
        else:
            other_error_count += 1
    
    if d_idx % 1000 == 0:
        print(f"Processed {d_idx}, Matches: {matches}, Disallowed: {disallowed_count}, Other errors: {other_error_count}")
    
print('###########')
print(f"Total matches: {matches}")
print(f"Total unmatches: {len(unmatch_idxs)}")
print(f"Unmatched indices: {unmatch_idxs}")
print(f"Disallowed expressions: {disallowed_count}")
print(f"Other errors: {other_error_count}")
print(f"Total processed: {matches + len(unmatch_idxs) + disallowed_count + other_error_count}")
if unmatch_idxs:
    idx = unmatch_idxs[0]
    simple_tokens = ast.literal_eval(df.iloc[idx]['simple'])
    scrambled_tokens = ast.literal_eval(df.iloc[idx]['scrambled'])
    print(
        "First Mismatch: Simple",
        tok.decode_infix(simple_tokens),
        ", Scrambled,",
        tok.decode_infix(scrambled_tokens)
    )
    # Re-run numerical equivalence to extract detailed mismatch diagnostics
    try:
        ok_detail, details = numerically_equivalent(
            tokenizer=tok,
            a_tokens=simple_tokens,
            b_tokens=scrambled_tokens,
            N=num_pts,
            samples=5,
            M=2.0,
            tol_abs=1e-8,
            tol_rel=1e-8,
            seed=123,
            return_details=True
        )
        # Expect ok_detail to be False here; print structured details if available
        print("Numerical mismatch details raw:", details)
        if isinstance(details, dict):
            max_abs = details.get('max_abs') or details.get('abs_max') or details.get('abs_err')
            max_rel = details.get('max_rel') or details.get('rel_max') or details.get('rel_err')
            print(f"Max abs diff: {max_abs}\nMax rel diff: {max_rel}")
            # Optionally show a single sample if present
            sample_vals = details.get('samples') or details.get('per_sample')
            if sample_vals:
                # sample_vals might be a list of tuples (a_val, b_val, abs_diff, rel_diff)
                first = sample_vals[0]
                print("First sample comparison:", first)
    except Exception as diag_err:
        print("(Diagnostics) Failed to compute mismatch details:", diag_err)
else:
    print("No mismatches recorded; unmatch_idxs is empty.")
