import h5py

filename = "amplitude_dataset.hdf5"
with h5py.File(filename, 'r') as f:
    simple_exprs = f["simple"][:]       # read all simple expressions
    scrambled_exprs = f["scrambled"][:]   # read all scrambled expressions

# Print a few entries to inspect
for i in range(min(5, len(simple_exprs))):
    print(f"Sample {i}:")
    print("  Simple:    ", simple_exprs[i])
    print("  Scrambled: ", scrambled_exprs[i])