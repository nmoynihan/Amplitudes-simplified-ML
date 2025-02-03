import random
import h5py
import numpy as np

#############################################
# Simple Amplitude Generation for Pure Gluons
#############################################

def generate_simple_gluon_term(n):
    """
    Generates a single “simple” term for an n-gluon scattering amplitude.
    
    For each gluon i=1,...,n, include one factor e_i.p_j (with j≠i).
    Include (n-2) factors of p_i.p_j in the denominator to yield the correct
    overall mass dimension of 4-n.
    """
    # Numerator: one factor for each gluon.
    numerator_factors = []
    for i in range(1, n+1):
        choices = [j for j in range(1, n+1) if j != i]
        chosen = random.choice(choices)
        numerator_factors.append(f"e{i}.p{chosen}")
    numerator_expr = " * ".join(numerator_factors)
    
    # Denominator: choose (n-2) unique momentum dot products.
    if n - 2 > 0:
        candidate_pairs = [(i, j) for i in range(1, n+1) for j in range(i+1, n+1)]
        denominator_pairs = random.sample(candidate_pairs, n - 2)
        denominator_factors = [f"p{a}.p{b}" for (a,b) in denominator_pairs]
        denominator_expr = " * ".join(denominator_factors)
        term = f"({numerator_expr})/({denominator_expr})"
    else:
        term = numerator_expr
    return term

def generate_simple_gluon_amplitude(n, num_terms=1):
    """
    Generates a simple amplitude expression for n-gluon scattering as a sum of terms.
    """
    terms = [generate_simple_gluon_term(n) for _ in range(num_terms)]
    return " + ".join(terms)

#####################################
# Scrambling Functions with Momentum Conservation
#####################################

def momentum_substitution(momentum_index, total_particles):
    """
    Returns a substitution string implementing momentum conservation for p_{momentum_index}.
    """
    others = [f"p{k}" for k in range(1, total_particles+1) if k != momentum_index]
    return "(-" + "+".join(others) + ")"

def momentum_conservation_substitution(expr, total_particles):
    """
    Replaces one occurrence of a momentum label in the expression with its momentum-conservation substitute.
    """
    idx = random.randint(1, total_particles)
    target = f"p{idx}"
    substitution = momentum_substitution(idx, total_particles)
    if target in expr:
        new_expr = expr.replace(target, substitution, 1)
        return new_expr
    else:
        return expr

def scramble_expression(expr, total_particles=5, include_mass=False):
    """
    Scrambles a simple amplitude expression using one of several operations.
    Operations include trivial multiplications/additions and applying momentum conservation.
    """
    ops = [0, 1, 3]  # 3 = applying momentum conservation to a momentum label in expr
    if include_mass:
        ops.append(2)
    op = random.choice(ops)
    
    if op == 0:
        # Multiply-by-1 using either an epsilon or momentum factor with momentum substitution.
        i = random.randint(1, total_particles)
        j = random.randint(1, total_particles)
        substituted = momentum_substitution(j, total_particles)
        if random.choice([True, False]):
            numerator = f"e{i}.{substituted}"
            denominator = f"e{i}.p{j}"
        else:
            numerator = f"p{i}.{substituted}"
            denominator = f"p{i}.p{j}"
        fraction = f"({numerator})/({denominator})"
        scrambled = f"({expr})*{fraction}"
    
    elif op == 1:
        # Add zero using transversality: e_i.p_i = 0.
        i = random.randint(1, total_particles)
        j_choices = [k for k in range(1, total_particles+1) if k != i]
        j = random.choice(j_choices)
        addition = f"(e{i}.p{i})/(e{i}.p{j})"
        scrambled = f"({expr})+{addition}"
    
    elif op == 2:
        # Multiply-by-1 using masses.
        # Todo: Implement this operation.
        pass
    elif op == 3:
        # Apply momentum conservation directly to the original expression.
        scrambled = momentum_conservation_substitution(expr, total_particles)
    """
    Note: Another thing that could added to scramble is to use the fact that eta_{\mu\nu} can be written in terms of \epsilon_\mu(k)\bar{\epsilon}_\nu(k) + k_\mu k_\nu / k^2 + gauge terms. This is a fairly horrible scramble, but is worth putting in.
    """
    
    return scrambled

def multi_scramble(expr, times=3, total_particles=5, include_mass=False):
    """
    Applies the scramble_expression function repeatedly.
    """
    scrambled_expr = expr
    for _ in range(times):
        scrambled_expr = scramble_expression(scrambled_expr, total_particles, include_mass)
    return scrambled_expr

##############################################
# Dataset Generation and Writing to HDF5 File
##############################################

def generate_dataset(num_samples, n_gluons, num_terms_simple=1, scramble_times=3):
    """
    For each sample, generate a simple amplitude expression and a scrambled version.
    
    Returns:
      (list of simple expressions, list of scrambled expressions)
    """
    simple_exprs = []
    scrambled_exprs = []
    for _ in range(num_samples):
        simple_expr = generate_simple_gluon_amplitude(n_gluons, num_terms=num_terms_simple)
        scrambled_expr = multi_scramble(simple_expr, times=scramble_times,
                                        total_particles=n_gluons, include_mass=False)
        simple_exprs.append(simple_expr)
        scrambled_exprs.append(scrambled_expr)
    return simple_exprs, scrambled_exprs

def write_dataset_to_hdf5(filename, simple_exprs, scrambled_exprs):
    """
    Writes the dataset to an HDF5 file with two datasets: 'simple' and 'scrambled'.
    """
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(filename, "w") as f:
        f.create_dataset("simple", data=simple_exprs, dtype=dt)
        f.create_dataset("scrambled", data=scrambled_exprs, dtype=dt)
    print("Dataset saved to", filename)

#########################################
# Main Execution
#########################################

if __name__ == "__main__":
    num_samples = 10000    # Total number of amplitude examples.
    n_gluons_max = 4           # Number of external gluons (1 polarization vector per gluon).
    n_gluons_min = 2
    num_terms_simple = 1   # Number of terms summed in the simple amplitude.
    scramble_times = 3     # Number of scrambling operations applied.
    i=n_gluons_min
    while (i<= n_gluons_max):
      # Generate the dataset.
      simple_exprs, scrambled_exprs = generate_dataset(num_samples, i,
                                                      num_terms_simple, scramble_times)
      
      # Write the dataset to an HDF5 file.
      output_filename = f"amplitude_dataset{i}.hdf5"
      write_dataset_to_hdf5(output_filename, simple_exprs, scrambled_exprs)
      i=i+1