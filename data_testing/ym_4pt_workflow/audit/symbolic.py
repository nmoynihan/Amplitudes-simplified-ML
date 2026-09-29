"""Independent exact audit of the specified scalar-contraction CSV.

No repository parser, amplitude implementation, or model is imported.
SymPy 1.14; all algebra is rational and no floating-point tolerance is used.
"""
import ast
import hashlib
from pathlib import Path
import re
import sympy as sp

from ._io import command_line, read_source, require, write_result

DOT = re.compile(r'([pe])_([1-4])\s*·\s*([pe])_([1-4])')
atom_labels = {}

def dot_atom(x, i, y, j):
    pair = tuple(sorted(((x, int(i)), (y, int(j)))))
    name = ''.join(f'{a}{b}' for a, b in pair)
    symbol = sp.Symbol(name)
    atom_labels[symbol] = pair
    return symbol

def parse(expression):
    if re.search(r"[A-Za-z_]", DOT.sub("0", expression)):
        raise ValueError("Unsupported identifier outside a scalar contraction")
    atoms = {}
    def replace(m):
        symbol = dot_atom(*m.groups())
        atoms[str(symbol)] = symbol
        return str(symbol)
    tree = ast.parse(DOT.sub(replace, expression).strip().replace('^', '**'), mode='eval')
    def walk(n):
        if isinstance(n, ast.Expression): return walk(n.body)
        if isinstance(n, ast.Name) and n.id in atoms: return atoms[n.id]
        if isinstance(n, ast.Constant) and type(n.value) is int: return sp.Integer(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -walk(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd): return walk(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = walk(n.left), walk(n.right)
            if isinstance(n.op, ast.Add): return a+b
            if isinstance(n.op, ast.Sub): return a-b
            if isinstance(n.op, ast.Mult): return a*b
            if isinstance(n.op, ast.Div): return a/b
            if isinstance(n.op, ast.Pow) and b.is_Integer: return a**b
        raise ValueError(ast.dump(n))
    return walk(tree)

a, b = sp.symbols('a b')  # a=s12/2; b=s23/2=s14/2

def pp(i, j):
    if i == j: return sp.Integer(0)
    return {(1,2):a, (1,3):-a-b, (1,4):b,
            (2,3):b, (2,4):-a-b, (3,4):a}[tuple(sorted((i,j)))]

def ep(i, j):
    if i == j: return sp.Integer(0)
    independent = [k for k in range(1,5) if k != i]
    if j == independent[-1]:
        return -sum(dot_atom('e',i,'p',k) for k in independent[:-1])
    return dot_atom('e',i,'p',j)

def on_shell(expr):
    rules = {}
    for x in expr.free_symbols:
        if x not in atom_labels: continue
        (t,i), (u,j) = atom_labels[x]
        if t == u == 'p': rules[x] = pp(i,j)
        elif t == 'e' and u == 'p': rules[x] = ep(i,j)
    return sp.cancel(expr.xreplace(rules))

def relabel(expr, order):
    rules = {}
    for x in expr.free_symbols:
        (t,i), (u,j) = atom_labels[x]
        rules[x] = dot_atom(t, order[i-1], u, order[j-1])
    return expr.xreplace(rules)

def ward(expr, leg):
    rules = {}
    for x in expr.free_symbols:
        (t,i), (u,j) = atom_labels[x]
        rules[x] = dot_atom('p' if t == 'e' and i == leg else t, i,
                            'p' if u == 'e' and j == leg else u, j)
    return on_shell(expr.xreplace(rules))

def ee(i,j): return dot_atom('e',i,'e',j)
def raw_ep(i,j): return dot_atom('e',i,'p',j)

def current(i,j):
    # J_ij^mu = e_i.e_j (p_i-p_j)^mu
    #          + 2(e_i.p_j)e_j^mu - 2(e_j.p_i)e_i^mu.
    # It is the contracted standard three-gluon kinematic vertex.
    return {('p',i):ee(i,j), ('p',j):-ee(i,j),
            ('e',j):2*raw_ep(i,j), ('e',i):-2*raw_ep(j,i)}

def vector_dot(v,w):
    return sp.expand(sum(c*d*dot_atom(t,i,u,j)
                      for (t,i),c in v.items() for (u,j),d in w.items()))

def zero(expr):
    value = sp.cancel(expr)
    require(value == 0, f"Expected an exact zero residual, got {value}")
    return str(value)

def run(source: Path, output_dir: Path) -> dict:
    """Prove tensor identities with exact rational scalar contractions."""
    source, data, rows = read_source(source)
    A = parse(rows[0][1])
    reduced = on_shell(A)
    J12J34 = vector_dot(current(1,2), current(3,4))
    J23J41 = vector_dot(current(2,3), current(4,1))
    contact = 2*ee(1,3)*ee(2,4)-ee(1,2)*ee(3,4)-ee(1,4)*ee(2,3)
    # Normalization: stripped color-ordered vertices J/sqrt(2), C/2.
    ref = on_shell(J12J34/(4*a) + J23J41/(4*b) + contact/2)
    result = {'source':str(source), 'sha256':hashlib.sha256(data).hexdigest(),
              'bytes':len(data), 'records':len(rows), 'row_id':rows[0][0],
              'sympy_version':sp.__version__, 'arithmetic':'exact rational symbolic algebra',
              'assumptions':['all momenta outgoing','p_i^2=0','sum_i p_i=0','e_i.p_i=0'],
              'dimension_specific_Gram_identities_used':False,
              'raw_additive_terms':len(sp.Add.make_args(sp.expand(A)))}
    # Check momentum degree 0 and polarization multi-degree (1,1,1,1), termwise.
    degrees = set()
    for term in sp.Add.make_args(sp.expand(A)):
        deg = [0]*5
        for sym, power in term.as_powers_dict().items():
            if sym not in atom_labels: continue
            for typ,i in atom_labels[sym]:
                deg[0 if typ == 'p' else i] += int(power)
        degrees.add(tuple(deg))
    require(degrees == {(0,1,1,1,1)}, f"Unexpected termwise degrees: {degrees}")
    result['termwise_degrees_momentum_e1_e2_e3_e4'] = [list(d) for d in degrees]
    result['feynman_reference_difference'] = zero(reduced-ref)
    result['ward_residuals'] = [zero(ward(A,i)) for i in range(1,5)]
    result['cyclic_residuals'] = [zero(on_shell(relabel(A,list(range(k+1,5))+list(range(1,k+1))))-reduced) for k in (1,2,3)]
    result['reflection_residual'] = zero(on_shell(relabel(A,[4,3,2,1]))-reduced)
    result['photon_decoupling_residual'] = zero(reduced+on_shell(relabel(A,[2,1,3,4]))+on_shell(relabel(A,[2,3,1,4])))
    result['BCJ_s12_A1234_minus_s13_A1324'] = zero(a*reduced-(-a-b)*on_shell(relabel(A,[1,3,2,4])))
    num, den = sp.fraction(reduced)
    result['reduced_denominator'] = str(sp.factor(den))
    require(sp.cancel(den/(a*b)).is_number, f"Unexpected reduced denominator: {den}")
    result['no_nonadjacent_or_double_poles'] = True
    # Residues in s=2a and t=2b: lim s*A = J12.J34/2, and analogously.
    result['s_residue_difference'] = zero(sp.cancel(2*a*reduced).subs(a,0)-on_shell(J12J34/2).subs(a,0))
    result['t_residue_difference'] = zero(sp.cancel(2*b*reduced).subs(b,0)-on_shell(J23J41/2).subs(b,0))
    result['s_current_conservation'] = zero(on_shell(vector_dot(current(1,2),{('p',1):1,('p',2):1})))
    result['t_current_conservation'] = zero(on_shell(vector_dot(current(2,3),{('p',2):1,('p',3):1})))
    result['u_zero_finite_expression'] = str(sp.factor(reduced.subs(b,-a)))
    result['s_residue'] = str(sp.factor(on_shell(J12J34/2).subs(a,0)))
    result['t_residue'] = str(sp.factor(on_shell(J23J41/2).subs(b,0)))
    result['reduced_expression'] = str(reduced)
    write_result(output_dir, 'symbolic_results.json', result)
    return result


if __name__ == '__main__':
    command_line(run, 'Exact scalar-contraction four-gluon tensor audit')
