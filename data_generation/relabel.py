import re
import csv
import argparse
import pathlib

# 1→5, 2→1, 3→2, 4→3, 5→4 (everyone shifts down by one; 1 wraps to 5)
idx_map = {'1': '5', '2': '1', '3': '2', '4': '3', '5': '4'}

# Match ONLY e_#, p_#, F_# where # is a single digit (1–5 in our case)
# \b ensures we hit whole tokens and don't touch things like M^2 or numeric literals.
pattern = re.compile(r'\b([epF])_(\d)\b')

def _remap_token(match: re.Match) -> str:
    var, i = match.group(1), match.group(2)   # var is e/p/F, i is the digit
    return f"{var}_{idx_map.get(i, i)}"       # swap 1–5 via idx_map; leave others as-is

def remap_csv(in_path: str, out_path: str) -> None:
    """
    Read a CSV, remap indices for e_i, p_i, F_i (1→5→4→3→2→1 cycle), and write a new CSV.
    All other text is left untouched.
    """
    # Read all rows
    with open(in_path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))

    # Apply the regex substitution to every cell
    remapped = [[pattern.sub(_remap_token, cell) for cell in row] for row in rows]

    # Write out the transformed CSV
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(remapped)

# Usage:
# remap_csv('input.csv', 'output.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relabel amplitude expressions held in a CSV file."
    )
    parser.add_argument("input_csv",  type=pathlib.Path, help="Path to the input CSV.")
    parser.add_argument("output_csv", type=pathlib.Path, help="Where to write the tokenised CSV.")

    args = parser.parse_args()
    remap_csv(args.input_csv, args.output_csv)