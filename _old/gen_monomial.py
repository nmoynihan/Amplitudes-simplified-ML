import random
import sympy as sp

def generate_sympy_monomial(n, n_gluons, n_gravitons, dim):
    """
    Generate a Sympy expression of total momentum (mass) dimension 'dim'.
    The expression is built from dot products:
      - p_i.p_j  (uses 2 momenta -> +2 to momentum count)
      - e_i.p_j  (uses 1 momentum -> +1 to momentum count)
      - e_i.e_j  (uses 0 momenta -> +0 to momentum count)
      
    Subject to:
      - The first n_gluons labels are gluons, each with 1 polarization vector e_i.
      - The next n_gravitons labels are gravitons, each with 2 polarization vectors.
      - The rest (n - n_gluons - n_gravitons) are scalars (unused here except as potential momentum labels).
      - We must use exactly each polarization index exactly once (gluon i -> 1 copy of e_i, graviton j -> 2 copies of e_j).
      - The total usage of momentum vectors must be dim (the sum of 2 for p_i.p_j + 1 for e_i.p_j + 0 for e_i.e_j).
    """
    
    # -- 1) Sympy setup: a 'dot' function and symbols for p_i and e_i
    dot = sp.Function('dot', commutative=True)  # commutative so factor ordering won't matter
    # We'll create symbolic placeholders for p_1, ..., p_n
    p_symbols = [sp.Symbol(f'p{i}', commutative=True) for i in range(1, n+1)]
    
    # We won't explicitly create separate e_1, e_2, etc. in Sympy here,
    # but we will store their "indices" and build dot(e_i, p_j) as dot(Symbol('e_i'), Symbol('p_j')).

    # -- 2) Collect all polarization "indices" (labels) into a list we will partition:
    #     - For gluons: exactly 1 copy of each label i in 1..n_gluons
    #     - For gravitons: 2 copies of each label j in n_gluons+1..n_gluons+n_gravitons
    pol_indices = []
    for i in range(1, n_gluons + 1):
        pol_indices.append(i)
    for j in range(n_gluons + 1, n_gluons + n_gravitons + 1):
        pol_indices.append(j)
        pol_indices.append(j)
    
    total_pols = len(pol_indices)  # = n_gluons + 2*n_gravitons
    
    # -- 3) We want to find nonnegative integers x, y, z satisfying:
    #       x + 2z = total_pols   (x factors e_i.p_j use one polarization each, e_i.e_j uses two)
    #       x + 2y = dim          (x factors e_i.p_j each use one momentum, p_i.p_j uses two)
    #
    #   => from x + 2z = total_pols, we get x = total_pols - 2z
    #      from x + 2y = dim => total_pols - 2z + 2y = dim
    #      => 2y - 2z = dim - total_pols
    #      => y - z = (dim - total_pols)/2
    #      => y = z + (dim - total_pols)/2
    #
    # We'll collect all integer z >= 0 for which x, y >= 0 and see if there's a valid solution. 
    # Then pick one at random for variety.
    
    valid_solutions = []
    for z_candidate in range(total_pols//2 + 1):  # z can go up to total_pols//2
        x_candidate = total_pols - 2*z_candidate
        # must be >= 0
        if x_candidate < 0:
            continue
        # from y = z + (dim - total_pols)/2
        # we also need (dim - total_pols) to be even for y to be integral
        shift = (dim - total_pols)
        if shift % 2 != 0:
            # no integral solution for y if shift is odd
            continue
        y_candidate = z_candidate + shift//2
        if y_candidate < 0:
            continue
        # we have a valid triple (x_candidate, y_candidate, z_candidate)
        valid_solutions.append((x_candidate, y_candidate, z_candidate))
    
    # If no valid solutions, return a trivial expression "1"
    if not valid_solutions:
        return sp.Integer(1)
    
    # Choose a random solution to get variety
    x, y, z = random.choice(valid_solutions)
    # x = # of e_i.p_j factors
    # y = # of p_i.p_j factors
    # z = # of e_i.e_j factors
    
    # -- 4) Build the random factors
    # We'll:
    #   (a) Shuffle pol_indices, then pick x of them for e_i.p_j factors,
    #   (b) from the remainder, group them in pairs to form e_i.e_j factors,
    #   (c) build y p_i.p_j factors by picking random pairs of momentum indices,
    #   (d) shuffle them all, then form a product.

    factors = []
    
    # 4a) Shuffle and pick x polarizations for e_i.p_j
    random.shuffle(pol_indices)
    
    ep_indices = pol_indices[:x]   # e_i's that will go with p_j
    remaining  = pol_indices[x:]   # leftover for e_i.e_j
    
    # build the e_i.p_j factors
    for e_idx in ep_indices:
        # choose a random momentum index in [1..n], say m_idx
        momentum_choices = [mn for mn in range(1, n+1) if mn != e_idx]
        m_idx = random.choice(momentum_choices)
        e_sym = sp.Symbol(f"e{e_idx}", commutative=True)
        p_sym = sp.Symbol(f"p{m_idx}", commutative=True)
        factors.append(dot(e_sym, p_sym))
    
    # 4b) From the remaining polarization indices, pair them up for e_i.e_j
    # We have z pairs => 2z leftover polarizations
    # (We must have exactly 2z = len(remaining) from the valid solution.)
    #eej_pairs = []
    for i in range(z):
        #print("Remaining:", remaining)
        # We need to remove any side-by-side duplicates to avoid ei.ei terms
        # Check for duplicates
        for _ in range(50):  # Try 50 times de-shuffle the remaining indices
            for j in range(0, len(remaining), 2):
                    if remaining[j] == remaining[j+1]:
                        print("Duplicate in e_i.e_j:", remaining, ", shuffling.")
                        valid = False
                        break
                    else:
                        valid = True
            if valid:
                break
                #print("No duplicates in e_i.e_j:", remaining)
            else:
                valid = True
                random.shuffle(remaining)
        
        i1 = remaining[2*i]
        i2 = remaining[2*i + 1]
        while i1 == i2:
            i2 = random.choice(remaining)  # Pick a new random index if they are the same
            print("Same index in e_i.e_j:", i1, i1, "Choosing new index for e_i.e_j", i2)
        e1 = sp.Symbol(f"e{i1}", commutative=True)
        e2 = sp.Symbol(f"e{i2}", commutative=True)
        factors.append(dot(e1, e2))
    # 4c) Now build y factors of p_i.p_j
    # We just pick y random pairs of momentum indices in [1..n].
    for _ in range(y):
        i1 = random.randint(1, n)
        i2 = random.choice([mn for mn in range(1, n+1) if mn != i1])
        p1 = sp.Symbol(f"p{i1}", commutative=True)
        p2 = sp.Symbol(f"p{i2}", commutative=True)
        factors.append(dot(p1, p2))
    
    # 4d) Shuffle all factors and multiply them into a single Sympy expression
    random.shuffle(factors)
    if not factors:
        return sp.Integer(1)
    
    monomial = sp.Mul(*factors)  # multiply all factors together
    
    return monomial


# --------------------------------------------------------------------
# Example usage (run several times to see random variety):
if __name__ == "__main__":
    expr = generate_sympy_monomial(n=5, n_gluons=5, n_gravitons=1, dim=3)
    print("Random Sympy Monomial:", expr)
