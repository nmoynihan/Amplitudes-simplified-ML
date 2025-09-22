#!/usr/bin/env python3
# gen_data.py 
#
# Build (simple , scrambled) training pairs for an N‑point amplitude with
#   • p₁ , p_N    scalars
#   • p₂…p_{N-1}  photons  (each owns e_i, F_i)
#
# “simple”  = a gauge‑invariant monomial containing each F_i exactly once
#             (weight‑1), possibly via Tr(F·F) or Tr(F·F·F) and possibly by p.F.F.p or similar.
# “scrambled” = that monomial rewritten into e·e/e·p/p·p and hit by
#               MIN_SCRAMBLES…MAX_SCRAMBLES algebraic identities.
#
# Output: CSV with two columns: simple, scrambled (plus tokenised file)

from __future__ import annotations
import random, re, time, json, math, csv, os
from itertools import product
from typing import List, Tuple
import importlib

# ╭──────────────────────────────────────────────────────────────────╮
# │  Pretty‑printing helpers                                         │
# ╰──────────────────────────────────────────────────────────────────╯
DOT = "·"                          # U+00B7, nicer than plain '.'

def dot(a: str, b: str) -> str: return f"{a} {DOT} {b}"
def Tr(*Fs: str) -> str:          return "Tr(" + f" {DOT} ".join(Fs) + ")"

def p(i): return f"p_{i}"
def e(i): return f"e_{i}"
def F(i): return f"F_{i}"

# ╭──────────────────────────────────────────────────────────────────╮
# │  Rewriting rules: GI block → e·e / e·p / p·p                    │
# ╰──────────────────────────────────────────────────────────────────╯
_RE_pFp    = re.compile(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)")
_RE_pFFp   = re.compile(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)")
_RE_Tr2    = re.compile(r"Tr\(F_(\d+)\s*·\s*F_(\d+)\)")
_RE_Tr3    = re.compile(r"Tr\(F_(\d+)\s*·\s*F_(\d+)\s*·\s*F_(\d+)\)")
_RE_pp     = re.compile(r"p_(\d+)\s*·\s*p_(\d+)")

def _rw_pFp(i,j,k):
    # p_i·F_j·p_k = (p_i·p_j)(e_j·p_k) - (p_i·e_j)(p_j·p_k)
    a = f"({dot(p(i), p(j))})*({dot(e(j), p(k))})"
    b = f"({dot(p(i), e(j))})*({dot(p(j), p(k))})"
    return f"({a} - {b})"

def _rw_pFFp(i,j,k,l):
    t1 = f"({dot(p(i), p(j))})*({dot(e(j), p(k))})*({dot(e(k), p(l))})"
    t2 = f"({dot(p(i), p(j))})*({dot(e(j), e(k))})*({dot(p(k), p(l))})"
    t3 = f"({dot(p(i), e(j))})*({dot(p(j), p(k))})*({dot(e(k), p(l))})"
    t4 = f"({dot(p(i), e(j))})*({dot(p(j), e(k))})*({dot(p(k), p(l))})"
    return f"({t1} - {t2} - {t3} + {t4})"

def _rw_Tr2(j,k):
    a = f"({dot(p(j), p(k))})*({dot(e(j), e(k))})"
    b = f"({dot(e(j), p(k))})*({dot(p(j), e(k))})"
    return f"(2*({b} - {a}))"

def _rw_Tr3(j,k,l):
    """
    Expand  Tr(F_j F_k F_l) into 8 terms of ordinary dot‑products.
    """
    terms = []
    for s1,s2,s3 in product((0,1), repeat=3):
        sign = "-" if (s1+s2+s3)%2 else "+"
        a1,b1 = (p(j),e(j)) if s1==0 else (e(j),p(j))
        a2,b2 = (p(k),e(k)) if s2==0 else (e(k),p(k))
        a3,b3 = (p(l),e(l)) if s3==0 else (e(l),p(l))
        term = f"({dot(b1,a2)})*({dot(b2,a3)})*({dot(a1,b3)})"
        terms.append(f"{sign}{term}")
    # join with spaces so later regexp scrambling still separates factors
    expr = " ".join(terms).replace("  ", " ")
    # move leading '+' if any
    return f"({expr.lstrip('+ ')})"

# Quick test of the rewriting rules
#print(_rw_Tr3(1,2,3))
#print(_rw_Tr2(1,2))
#print(_rw_pFFp(1,2,3,4))



def rewrite_gi(block: str) -> str:
    if (m:=_RE_pFFp.fullmatch(block)):   return _rw_pFFp(*map(int,m.groups()))
    if (m:=_RE_pFp.fullmatch(block)):    return _rw_pFp (*map(int,m.groups()))
    if (m:=_RE_Tr2.fullmatch(block)):    return _rw_Tr2 (*map(int,m.groups()))
    if (m:=_RE_Tr3.fullmatch(block)):    return _rw_Tr3 (*map(int,m.groups()))
    return block   # p·p passes through

# ╭──────────────────────────────────────────────────────────────────╮
# │  Strict GI‑monomial generator (weight 1)                         │
# ╰──────────────────────────────────────────────────────────────────╯
PAIR_PROB     = 0.4   # probability to pair two photons into one block
EXTRA_SCALARS = 4     # ≤ how many p·p factors to add

def _singleF(j,N):
    a,b = random.sample([x for x in range(1,N+1) if x!=j], 2)
    return f"{p(a)} {DOT} {F(j)} {DOT} {p(b)}"

def _doubleF(j,k,N):
    a,b = random.sample([x for x in range(1,N+1) if x not in (j,k)], 2)
    return f"{p(a)} {DOT} {F(j)} {DOT} {F(k)} {DOT} {p(b)}"

def _tr2(j,k):       return Tr(F(j),F(k))
def _tr3(j,k,l):     return Tr(F(j),F(k),F(l))
def _scalar(N):
    i,j = random.sample(range(1,N+1),2)
    return dot(p(i),p(j))

def strict_gi_monomial(N:int) -> str:
    remaining = list(range(2,N))   # photon labels
    random.shuffle(remaining)
    factors: List[str] = []

    while remaining:
        r  = len(remaining)
        ops = []
        if r>=3: ops.append("tr3")
        if r>=2: ops += ["tr2","doubleF"]
        if r>=1: ops.append("singleF")
        kind = random.choice(ops)

        if kind=="tr3":
            j,k,l = random.sample(remaining,3)
            factors.append(_tr3(j,k,l))
            remaining = [x for x in remaining if x not in (j,k,l)]
        elif kind=="tr2":
            j,k = random.sample(remaining,2)
            factors.append(_tr2(j,k))
            remaining = [x for x in remaining if x not in (j,k)]
        elif kind=="doubleF":
            j,k = random.sample(remaining,2)
            factors.append(_doubleF(j,k,N))
            remaining = [x for x in remaining if x not in (j,k)]
        else: # singleF
            j  = remaining.pop()
            factors.append(_singleF(j,N))

    for _ in range(random.randint(0,EXTRA_SCALARS)):
        factors.append(_scalar(N))

    random.shuffle(factors)
    return "*".join(factors)

# ╭──────────────────────────────────────────────────────────────────╮
# │  Canonicalisation (lightweight, semantics‑preserving)            │
# ╰──────────────────────────────────────────────────────────────────╯
def _canon_Tr2(term:str) -> str:
    m = _RE_Tr2.fullmatch(term)
    if not m:
        return term
    a,b = map(int, m.groups())
    a,b = (a,b) if a<=b else (b,a)
    return Tr(F(a), F(b))

def _canon_Tr3(term:str) -> str:
    m = _RE_Tr3.fullmatch(term)
    if not m:
        return term
    a,b,c = map(int, m.groups())
    rotations = [(a,b,c), (b,c,a), (c,a,b)]
    aa,bb,cc = min(rotations)
    return Tr(F(aa), F(bb), F(cc))

def _canon_pp(term:str) -> str:
    m = _RE_pp.fullmatch(term)
    if not m:
        return term
    i,j = map(int, m.groups())
    i,j = (i,j) if i<=j else (j,i)
    return dot(p(i), p(j))

def _factor_key(term:str) -> tuple:
    # Category order: Tr3 < Tr2 < p·F·F·p < p·F·p < p·p < other
    if _RE_Tr3.fullmatch(term): cat = 0; norm = _canon_Tr3(term)
    elif _RE_Tr2.fullmatch(term): cat = 1; norm = _canon_Tr2(term)
    elif _RE_pFFp.fullmatch(term): cat = 2; norm = term
    elif _RE_pFp.fullmatch(term):  cat = 3; norm = term
    elif _RE_pp.fullmatch(term):   cat = 4; norm = _canon_pp(term)
    else:                          cat = 5; norm = term
    return (cat, norm)

def _normalise_factor_str(term:str) -> str:
    if _RE_Tr2.fullmatch(term):
        return _canon_Tr2(term)
    if _RE_Tr3.fullmatch(term):
        return _canon_Tr3(term)
    if _RE_pp.fullmatch(term):
        return _canon_pp(term)
    return term

def canonicalise_gi_product(prod:str, strict: bool=False) -> str:
    # prod is a top-level '*' joined product of GI factors
    factors = prod.split("*") if prod else []
    canon = []
    for f in factors:
        f = f.strip()
        f = _normalise_factor_str(f)
        canon.append(f)
    if strict:
        canon.sort(key=_normalise_factor_str)
    else:
        canon.sort(key=_factor_key)
    return "*".join(canon)

def canonicalise_denominator(den:str) -> str:
    # Denominator is a '*' joined product of p·p terms
    if not den:
        return den
    fs = [ _canon_pp(f.strip()) for f in den.split("*") ]
    fs.sort(key=_factor_key)
    return "*".join(fs)

# ╭──────────────────────────────────────────────────────────────────╮
# │  Scramblers (no symmetric‑dot)                                   |
# ╰──────────────────────────────────────────────────────────────────╯
def _mc_terms(idx:int,N:int) -> List[str]:
    return [f"{dot(p(k),p(idx))}" for k in range(1,N+1) if k!=idx]

def scr_mul_by_one(expr:str,N:int)->str:
    i,j = random.sample(range(1,N+1),2)
    one = f"({dot(p(i),p(j))})/({dot(p(i),p(j))})"
    return f"({expr})*{one}"

def scr_add_zero_gauge(expr:str,Ngamma:int,N:int)->str:
    i = random.randint(2,Ngamma+1)
    rhs = " + ".join(dot(e(i),p(k)) for k in range(1,N+1))
    term = f"({rhs})"
    return f"({expr}) + ({term})"

def scr_Ptot_dot_pk(expr:str,N:int)->str:
    k = random.randint(1,N)
    term = " + ".join(dot(p(s),p(k)) for s in range(1,N+1))
    return f"({expr}) + ({term})"

_SCRAMBLERS = [
    lambda e,Ng,Nt: scr_mul_by_one(e,Nt),
    lambda e,Ng,Nt: scr_add_zero_gauge(e,Ng,Nt),
    lambda e,Ng,Nt: scr_Ptot_dot_pk(e,Nt),
]

def _scramble_legacy(expr:str,Ngamma:int,N:int,max_scr:int)->str:
    n = random.randint(0, max_scr) if max_scr > 0 else 0
    out = expr
    for _ in range(n):
        out = random.choice(_SCRAMBLERS)(out,Ngamma,N)
    return out

# Extra scramblers: targeted MC substitution and commutativity flips
_RE_DOT = re.compile(r"(p_\d+|e_\d+)\s*·\s*(p_\d+|e_\d+)")

def scr_mc_substitute_ei_pk(expr:str,Ngamma:int,N:int)->str:
    i = random.randint(2,Ngamma+1)
    k = random.randint(1,N)
    pattern = re.escape(dot(e(i), p(k)))
    replacement = " - (" + " + ".join(dot(e(i),p(s)) for s in range(1,N+1) if s!=k) + ")"
    return re.sub(pattern, replacement, expr, count=1)

def scr_commute_dot(expr:str,Ngamma:int,N:int)->str:
    matches = list(_RE_DOT.finditer(expr))
    if not matches:
        return expr
    m = random.choice(matches)
    a,b = m.group(1), m.group(2)
    return expr[:m.start()] + dot(b,a) + expr[m.end():]

_SCRAMBLERS.extend([
    lambda e,Ng,Nt: scr_mc_substitute_ei_pk(e,Ng,Nt),
    lambda e,Ng,Nt: scr_commute_dot(e,Ng,Nt),
])

# Safety: cap output growth
def scramble(expr:str,Ngamma:int,N:int,min_scr:int=0,max_scr:int=0, max_len:int=4000)->str:
    # Choose number of scrambles in [min_scr, max_scr], clamp to valid range
    try:
        min_scr = int(min_scr)
        max_scr = int(max_scr)
    except Exception:
        min_scr = 0
        max_scr = 0
    if min_scr < 0:
        min_scr = 0
    if max_scr < 0:
        max_scr = 0
    if min_scr > max_scr:
        min_scr = max_scr
    n = random.randint(min_scr, max_scr) if max_scr > 0 else 0
    out = expr
    for _ in range(n):
        cand = random.choice(_SCRAMBLERS)(out,Ngamma,N)
        out = cand if len(cand) <= max_len else out
    return out

# Numeric evaluator (Minkowski +,−,−,−)
def _mdot(a, b):
    return float(a[0]*b[0] - (a[1]*b[1] + a[2]*b[2] + a[3]*b[3]))

def _to_float_expr(expr:str, P:dict, E:dict) -> str:
    def repl(m):
        a, b = m.group(1), m.group(2)
        va = P[a] if a.startswith('p_') else E[a]
        vb = P[b] if b.startswith('p_') else E[b]
        return f"({_mdot(va, vb):.17g})"
    # Replace all p/e dot products by their numeric Minkowski inner product.
    expr_num = _RE_DOT.sub(repl, expr)
    # Our tokeniser uses '^' for exponentiation; Python's AST treats '^' as bitwise XOR.
    # Convert caret powers into Python '**' so the safe evaluator recognises them as ast.Pow.
    # This is a simple textual replacement because '^' is not otherwise used in our grammar.
    if '^' in expr_num:
        expr_num = expr_num.replace('^', '**')
    return expr_num

import ast

class _SafeEval(ast.NodeVisitor):
    def visit_Expression(self, node):
        return self.visit(node.body)
    def visit_BinOp(self, node):
        l = self.visit(node.left)
        r = self.visit(node.right)
        if isinstance(node.op, ast.Add): return l + r
        if isinstance(node.op, ast.Sub): return l - r
        if isinstance(node.op, ast.Mult): return l * r
        if isinstance(node.op, ast.Div): return l / r
        if isinstance(node.op, ast.Pow): return l ** r
        raise ValueError("disallowed operator")
    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd): return +v
        if isinstance(node.op, ast.USub): return -v
        raise ValueError("disallowed unary operator")
    def visit_Num(self, node):  # pragma: no cover
        val = node.n
        if isinstance(val, (int,float)):
            return float(val)
        raise ValueError("non-real literal")
    def visit_Constant(self, node):
        if isinstance(node.value, (int,float)):
            return float(node.value)
        raise ValueError("disallowed constant")
    def generic_visit(self, node):
        raise ValueError("disallowed expression")

def _safe_eval_float(expr: str) -> float:
    tree = ast.parse(expr, mode='eval')
    return _SafeEval().visit(tree)

def eval_infix_numeric(expr: str, momenta, pols) -> float:
    N = len(momenta)
    P = {f"p_{i}": momenta[i-1] for i in range(1,N+1)}
    E = {f"e_{i}": pols[i-2] for i in range(2,N)}
    # --- Field‑strength expansions -----------------------------------------
    # decode_infix may yield expressions containing Tr((F_i·F_j)), Tr((F_i·F_j·F_k)),
    # and also GI blocks like p_i·F_j·p_k or p_i·F_j·F_k·p_l.
    # Expand all of these into p/e dot products using the same algebraic identities
    # as the dataset generator so numeric evaluation only sees p_*/e_* tokens.
    def _expand_traces(s: str) -> str:
        # Normalise inner spacing & remove double parentheses like (F_2·F_3)
        changed = True
        while changed:
            changed = False
            # Tr of two F
            def rep2(m):
                nonlocal changed
                changed = True
                j,k = map(int, m.groups())
                return _rw_Tr2(j,k)  # defined above
            # Allow optional parentheses around the F chain
            s_new = re.sub(r"Tr\(\(?F_(\d+)\s*·\s*F_(\d+)\)?\)", rep2, s)
            # Tr of three F
            def rep3(m):
                nonlocal changed
                changed = True
                j,k,l = map(int, m.groups())
                return _rw_Tr3(j,k,l)
            s_new = re.sub(r"Tr\(\(?F_(\d+)\s*·\s*F_(\d+)\s*·\s*F_(\d+)\)?\)", rep3, s_new)
            s = s_new
        return s

    def _expand_gi_blocks(s: str) -> str:
        changed = True
        while changed:
            changed = False
            # p_i · F_j · p_k
            def rep_pfp(m):
                nonlocal changed
                changed = True
                i, j, k = map(int, m.groups())
                return _rw_pFp(i, j, k)
            s_new = re.sub(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)", rep_pfp, s)

            # p_i · F_j · F_k · p_l
            def rep_pffp(m):
                nonlocal changed
                changed = True
                i, j, k, l = map(int, m.groups())
                return _rw_pFFp(i, j, k, l)
            s_new = re.sub(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)", rep_pffp, s_new)
            s = s_new
        return s

    expr_expanded = _expand_traces(expr)
    expr_expanded = _expand_gi_blocks(expr_expanded)
    expr_f = _to_float_expr(expr_expanded, P, E)
    # Final safety: balance any stray parentheses to avoid SyntaxError
    def _balance_parens_str(s: str) -> str:
        out = []
        bal = 0
        for ch in s:
            if ch == '(':
                bal += 1
                out.append(ch)
            elif ch == ')':
                if bal > 0:
                    bal -= 1
                    out.append(ch)
                else:
                    # drop unmatched closing
                    continue
            else:
                out.append(ch)
        if bal > 0:
            out.append(')' * bal)
        return ''.join(out)
    expr_f = _balance_parens_str(expr_f)
    return float(_safe_eval_float(expr_f))

# Debug helper: return the numeric-ready string after all expansions and substitutions
def to_numeric_string(expr: str, momenta, pols) -> str:
    N = len(momenta)
    P = {f"p_{i}": momenta[i-1] for i in range(1,N+1)}
    E = {f"e_{i}": pols[i-2] for i in range(2,N)}

    def _expand_traces(s: str) -> str:
        changed = True
        while changed:
            changed = False
            def rep2(m):
                nonlocal changed
                changed = True
                j,k = map(int, m.groups())
                return _rw_Tr2(j,k)
            s_new = re.sub(r"Tr\(\(?F_(\d+)\s*·\s*F_(\d+)\)?\)", rep2, s)
            def rep3(m):
                nonlocal changed
                changed = True
                j,k,l = map(int, m.groups())
                return _rw_Tr3(j,k,l)
            s_new = re.sub(r"Tr\(\(?F_(\d+)\s*·\s*F_(\d+)\s*·\s*F_(\d+)\)?\)", rep3, s_new)
            s = s_new
        return s

    def _expand_gi_blocks(s: str) -> str:
        changed = True
        while changed:
            changed = False
            def rep_pfp(m):
                nonlocal changed
                changed = True
                i, j, k = map(int, m.groups())
                return _rw_pFp(i, j, k)
            s_new = re.sub(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)", rep_pfp, s)
            def rep_pffp(m):
                nonlocal changed
                changed = True
                i, j, k, l = map(int, m.groups())
                return _rw_pFFp(i, j, k, l)
            s_new = re.sub(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)", rep_pffp, s_new)
            s = s_new
        return s

    expr_expanded = _expand_traces(expr)
    expr_expanded = _expand_gi_blocks(expr_expanded)
    expr_f = _to_float_expr(expr_expanded, P, E)
    # Balance for debug visibility
    def _balance_parens_str(s: str) -> str:
        out = []
        bal = 0
        for ch in s:
            if ch == '(':
                bal += 1
                out.append(ch)
            elif ch == ')':
                if bal > 0:
                    bal -= 1
                    out.append(ch)
                else:
                    continue
            else:
                out.append(ch)
        if bal > 0:
            out.append(')' * bal)
        return ''.join(out)
    expr_f = _balance_parens_str(expr_f)
    if '^' in expr_f:
        expr_f = expr_f.replace('^', '**')
    return expr_f

# ╭──────────────────────────────────────────────────────────────────╮
# │  Dataset construction & I/O                                      │
# ╰──────────────────────────────────────────────────────────────────╯
def _random_denominator(N:int, base_leg:int=1) -> str:
    k = random.randint(1, min(3, N-1))
    js = random.sample([j for j in range(1,N+1) if j!=base_leg], k)
    factors = [dot(p(base_leg), p(j)) for j in js]
    return "*".join(factors)

def _gauge_denominator(N:int, style:str="shared", prefer_scalars:bool=True) -> str:
    """
    Build a product of (n · p_i) over photon legs i ∈ {2..N-1}.
    style = 'shared' uses a single reference n for all photons; 'per-photon' picks one per i.
    If prefer_scalars, choose n from {1, N} to avoid (nearly) zero denominators.
    """
    photons = list(range(2, N))
    if not photons:
        return ""
    if prefer_scalars:
        pool = [1, N]
    else:
        pool = list(range(1, N+1))
    factors: list[str] = []
    if style == "shared":
        # Choose one n distinct from all photons; prefer scalars
        cand = [x for x in pool if x not in photons]
        nref = random.choice(cand) if cand else random.choice(pool)
        for i in photons:
            factors.append(dot(p(nref), p(i)))
    else:  # per-photon
        for i in photons:
            cand = [x for x in pool if x != i]
            nref = random.choice(cand) if cand else random.choice(pool)
            factors.append(dot(p(nref), p(i)))
    return "*".join(factors)

def build_dataset(N:int, num_samples:int, max_scr:int=3, min_scr:int=0, seed:int|None=None,
                  use_denominators:bool=True, validate:bool=True,
                  M:float=2.0, tol_rel:float=1e-8, tol_abs:float=1e-10,
                  min_terms:int=1, max_terms:int=1,
                  log_path:str|None=None,
                  log_examples:int=5,
                  parity_checks:int=1,
                  scramble_checks:int=1,
                  progress_cb=None) -> List[Tuple[str,str]]:
    """
    Build dataset of (simple, scrambled) expression pairs.

    New (polynomial) behaviour:
      When min_terms or max_terms > 1 a polynomial with T terms is generated where
        T ~ Uniform{min_terms..max_terms}.
      Each term is an independently generated GI monomial (legacy behaviour) possibly
      with its own denominator. All terms are constrained to have the same "mass dimension"
      proxy, implemented as TWO constraints:
          (A) Denominator matching: either all terms have a denominator or none do, AND
              the number of p·p factors in each denominator is identical (length match).
          (B) Numerator factor-type signature match: For each numerator we build a 5‑tuple
              (#Tr3, #Tr2, #p·F·F·p, #p·F·p, #p·p_extra). Here #p·p_extra counts only the
              explicit scalar p·p factors appended during generation, not those inside GI blocks.
              All terms in the polynomial must share this signature. After a bounded number of
              attempts (30) the constraint is relaxed to avoid infinite loops.
      (If after several attempts a matching term is not found, we relax the constraint to
       avoid infinite loops.)

    Coefficients: Each term receives a random integer coefficient c in [-100,100] excluding 0.
      Formatting rules:
          • Coefficient 1 is omitted ("A" not "1*A").
          • Coefficient -1 is rendered as a leading minus ("-A").
          • Other coefficients appear as "c*A".
          • Polynomial is joined by ' + ' and ' - ' with minimal parentheses.
      Scrambling is applied once to the fully expanded (rewritten) polynomial.
    """
    # Clamp & sanity for terms
    try:
        min_terms = int(min_terms)
        max_terms = int(max_terms)
    except Exception:
        min_terms = max_terms = 1
    if min_terms < 1: min_terms = 1
    if max_terms < min_terms: max_terms = min_terms
    if seed is not None:
        random.seed(seed)
    Ngamma = N-2
    data=[]
    attempts=0
    parity_fail = 0
    scramble_fail = 0
    parity_examples: list[tuple[str,str,str]] = []  # (simple, expanded, reason)
    scramble_examples: list[tuple[str,str,str]] = []
    if log_path:
        # Initialise / truncate log file
        with open(log_path, 'w', encoding='utf-8') as lf:
            lf.write(f"# gen_data log for N={N}\n")
            lf.write(f"# target_samples={num_samples} min_terms={min_terms} max_terms={max_terms} max_scr={max_scr} min_scr={min_scr} seed={seed}\n")
    while len(data) < num_samples:
        attempts += 1
        # Decide how many terms this sample will have
        T = random.randint(min_terms, max_terms)

        def _numerator_signature(simple_num: str) -> tuple[int,int,int,int,int]:
            """Compute factor-type signature for the GI numerator product.
            Categories (order fixed): Tr3, Tr2, pFFp, pFp, scalar_pp.
            """
            if not simple_num:
                return (0,0,0,0,0)
            tr3=tr2=pffp=pfp=pp=0
            for f in simple_num.split('*'):
                f=f.strip()
                if _RE_Tr3.fullmatch(f): tr3+=1; continue
                if _RE_Tr2.fullmatch(f): tr2+=1; continue
                if _RE_pFFp.fullmatch(f): pffp+=1; continue
                if _RE_pFp.fullmatch(f): pfp+=1; continue
                if _RE_pp.fullmatch(f): pp+=1; continue
            return (tr3,tr2,pffp,pfp,pp)

        def _generate_monomial() -> tuple[str,str,int,bool,tuple[int,int,int,int,int]]:
            """Return (simple_term, expanded_term, denom_len, has_denom, signature)."""
            gi = strict_gi_monomial(N)
            simple_num = canonicalise_gi_product(gi, strict=True)
            den_parts: list[str] = []
            if use_denominators and random.random() < 0.6:
                den_parts.append(_random_denominator(N, base_leg=1))
            den_parts.append(_gauge_denominator(N, style="shared", prefer_scalars=True))
            den_parts = [d for d in den_parts if d]
            if den_parts:
                denom = "*".join(den_parts)
                simple_den = canonicalise_denominator(denom)
                simple_term = f"({simple_num})/({simple_den})"
                expd_num = "*".join(rewrite_gi(b) for b in simple_num.split("*"))
                expd_term = f"({expd_num})/({simple_den})"
                denom_len = len(simple_den.split('*'))
                has_denom = True
            else:
                simple_term = simple_num
                expd_term = "*".join(rewrite_gi(b) for b in simple_num.split("*"))
                denom_len = 0
                has_denom = False
            sig = _numerator_signature(simple_num)
            return simple_term, expd_term, denom_len, has_denom, sig

        terms_simple: list[str] = []
        terms_expanded: list[str] = []
        coeffs: list[int] = []

        # Generate first term unconditionally
        s0, e0, denom_len_ref, has_denom_ref, sig_ref = _generate_monomial()
        terms_simple.append(s0)
        terms_expanded.append(e0)
        coeffs.append(random.choice([c for c in range(-9,10) if c != 0]))

        # Subsequent terms: enforce same denominator length & presence (proxy for mass dimension)
        for _ in range(T-1):
            sx = ex = None  # type: ignore
            dlen = 0; hden = False; sig_cur = None
            for attempt in range(30):  # bounded attempts to find structurally matching term
                sx_try, ex_try, dlen_try, hden_try, sig_try = _generate_monomial()
                sx, ex, dlen, hden, sig_cur = sx_try, ex_try, dlen_try, hden_try, sig_try
                if (hden == has_denom_ref and dlen == denom_len_ref and sig_cur == sig_ref):
                    break
            if sx is None or ex is None:  # fallback (shouldn't happen)
                sx, ex, dlen, hden, sig_cur = _generate_monomial()
            terms_simple.append(sx)
            terms_expanded.append(ex)
            coeffs.append(random.choice([c for c in range(-100,101) if c != 0]))

        def _format_poly(terms: list[str], coeffs: list[int]) -> str:
            out_parts: list[str] = []
            for i, (t, c) in enumerate(zip(terms, coeffs)):
                abs_c = abs(c)
                sign = '-' if c < 0 else '+'
                if abs_c == 1:
                    core = t
                else:
                    core = f"{abs_c}*{t}"
                if i == 0:
                    if sign == '-':
                        out_parts.append(f"-{core}")
                    else:
                        out_parts.append(core)
                else:
                    out_parts.append(f" {sign} {core}")
            return ''.join(out_parts)

        simple_poly = _format_poly(terms_simple, coeffs)
        expanded_poly = _format_poly(terms_expanded, coeffs)

        # (NEW) Parity validation: ensure GI simple polynomial numerically matches expanded form
        if validate:
            ok_parity = True
            parity_reason = ''
            for _ in range(max(0, int(parity_checks))):
                try:
                    try:
                        from .kinematics import generate_kinematics as _gk  # type: ignore
                    except Exception:
                        _gk = importlib.import_module('data_generation.kinematics').generate_kinematics  # type: ignore
                except Exception:
                    _gk = importlib.import_module('kinematics').generate_kinematics  # type: ignore
                momenta, pols = _gk(N, M=M)
                try:
                    v_simple_gi   = eval_infix_numeric(simple_poly,   momenta, pols)
                    v_expanded    = eval_infix_numeric(expanded_poly, momenta, pols)
                except Exception as ex_par:
                    ok_parity = False
                    parity_reason = f"exception:{ex_par}"; break
                if not (math.isfinite(v_simple_gi) and math.isfinite(v_expanded)):
                    ok_parity = False
                    parity_reason = 'non-finite'; break
                if not (abs(v_simple_gi - v_expanded) <= max(tol_abs, tol_rel*max(1.0, abs(v_expanded)))):
                    ok_parity = False
                    parity_reason = f"mismatch|Δ={abs(v_simple_gi - v_expanded):.3e}"; break
            if not ok_parity:
                parity_fail += 1
                if log_path and len(parity_examples) < log_examples:
                    parity_examples.append((simple_poly, expanded_poly, parity_reason))
                continue

        # Scramble AFTER confirming parity (scrambled only depends on expanded_poly)
        scrambled = scramble(expanded_poly, Ngamma, N, min_scr, max_scr)

        if validate:
            ok_scramble = True
            scramble_reason = ''
            for _ in range(max(0, int(scramble_checks))):
                try:
                    try:
                        from .kinematics import generate_kinematics as _gk  # type: ignore
                    except Exception:
                        _gk = importlib.import_module('data_generation.kinematics').generate_kinematics  # type: ignore
                except Exception:
                    _gk = importlib.import_module('kinematics').generate_kinematics  # type: ignore
                momenta, pols = _gk(N, M=M)
                try:
                    v_expanded = eval_infix_numeric(expanded_poly, momenta, pols)
                    v_scr      = eval_infix_numeric(scrambled,     momenta, pols)
                except Exception as ex_scr:
                    ok_scramble = False
                    scramble_reason = f"exception:{ex_scr}"; break
                if not (math.isfinite(v_expanded) and math.isfinite(v_scr)):
                    ok_scramble = False
                    scramble_reason = 'non-finite'; break
                if not (abs(v_expanded - v_scr) <= max(tol_abs, tol_rel*max(1.0, abs(v_expanded)))):
                    ok_scramble = False
                    scramble_reason = f"mismatch|Δ={abs(v_expanded - v_scr):.3e}"; break
            if not ok_scramble:
                scramble_fail += 1
                if log_path and len(scramble_examples) < log_examples:
                    scramble_examples.append((expanded_poly, scrambled, scramble_reason))
                continue

        data.append((simple_poly, scrambled))
        if progress_cb:
            try:
                progress_cb(len(data))
            except Exception:
                pass
    if log_path:
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(f"# SUMMARY\n")
            lf.write(f"accepted={len(data)} parity_fail={parity_fail} scramble_fail={scramble_fail} attempts={attempts}\n")
            if parity_examples:
                lf.write("# PARITY_FAIL_EXAMPLES\n")
                for s,e,r in parity_examples:
                    lf.write(f"reason={r}\nSIMPLE={s}\nEXPANDED={e}\n---\n")
            if scramble_examples:
                lf.write("# SCRAMBLE_FAIL_EXAMPLES\n")
                for e,sc,r in scramble_examples:
                    lf.write(f"reason={r}\nEXPANDED={e}\nSCRAMBLED={sc}\n---\n")
    return data

# ──────────────────────────────────────────────────────────────────────────────
# Parallel generation utilities
# ──────────────────────────────────────────────────────────────────────────────
def _init_worker(seed_base:int):  # per-process seeding
    try:
        random.seed(seed_base + os.getpid())
    except Exception:
        random.seed()

def _batch_worker(batch_size:int, N:int, cfg:dict):
    out: list[Tuple[str,str]] = []
    # local lightweight wrapper around build_dataset inner loop using existing function
    # We call build_dataset with small num_samples=batch_size and validation params.
    pairs = build_dataset(
        N=N,
        num_samples=batch_size,
        max_scr=cfg['max_scr'],
        min_scr=cfg['min_scr'],
        seed=None,  # already seeded per process
        use_denominators=cfg['use_denominators'],
        validate=cfg['validate'],
        M=cfg['M'], tol_rel=cfg['tol_rel'], tol_abs=cfg['tol_abs'],
        min_terms=cfg['min_terms'], max_terms=cfg['max_terms'],
        log_path=None, log_examples=0,
        parity_checks=cfg['parity_checks'], scramble_checks=cfg['scramble_checks'],
        progress_cb=None
    )
    out.extend(pairs)
    return out

def build_dataset_parallel(N:int, target:int, workers:int=4, batch_size:int=200, **kwargs) -> List[Tuple[str,str]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    pbar = None
    try:  # optional tqdm
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(total=target, desc='Generating', unit='pair')
    except Exception:
        pbar = None
    cfg = dict(
        max_scr=kwargs.get('max_scr',3),
        min_scr=kwargs.get('min_scr',0),
        use_denominators=kwargs.get('use_denominators',True),
        validate=kwargs.get('validate',True),
        M=kwargs.get('M',2.0), tol_rel=kwargs.get('tol_rel',1e-8), tol_abs=kwargs.get('tol_abs',1e-10),
        min_terms=kwargs.get('min_terms',1), max_terms=kwargs.get('max_terms',1),
        parity_checks=kwargs.get('parity_checks',1), scramble_checks=kwargs.get('scramble_checks',1),
    )
    seed = kwargs.get('seed', None) or int(time.time())
    results: list[Tuple[str,str]] = []
    seen: set[Tuple[str,str]] = set()
    # pbar already initialised if tqdm available
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(seed,)) as ex:
        futures = {ex.submit(_batch_worker, batch_size, N, cfg): 0 for _ in range(workers)}
        while futures and len(results) < target:
            for fut in as_completed(list(futures.keys())):
                try:
                    batch = fut.result()
                except Exception:
                    batch = []
                added = 0
                for pair in batch:
                    if pair in seen:
                        continue
                    seen.add(pair)
                    results.append(pair)
                    added += 1
                    if len(results) >= target:
                        break
                if pbar and added:
                    pbar.update(added)
                # replace completed future if still need more
                futures.pop(fut, None)
                if len(results) < target:
                    futures[ex.submit(_batch_worker, batch_size, N, cfg)] = 0
                if len(results) >= target:
                    break
    if pbar is not None:
        pbar.close()
    return results[:target]

def write_txt(pairs:List[Tuple[str,str]],path:str)->None:
    with open(path,"w",encoding="utf-8") as f:
        for s,t in pairs: f.write(f"{s}\t{t}\n")

# New: CSV writer
def write_csv(pairs:List[Tuple[str,str]], path:str) -> None:
    # Proper CSV writing with escaping; newline="" for Windows correctness
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # header
        w.writerow(["simple", "scrambled"])
        for s, t in pairs:
            w.writerow([s, t])

# New: In-memory deduplication of (simple, scrambled) pairs
def dedupe_pairs(pairs: List[Tuple[str, str]], keep: str = "first") -> tuple[List[Tuple[str, str]], int]:
    """
    Remove exact duplicate pairs. By default keeps the first occurrence.

    keep: 'first' | 'last'
    Returns: (deduped_pairs, removed_count)
    """
    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")
    if not pairs:
        return pairs, 0

    if keep == "first":
        seen: set[Tuple[str, str]] = set()
        out: list[Tuple[str, str]] = []
        for item in pairs:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        removed = len(pairs) - len(out)
        return out, removed
    else:
        # keep == 'last': record last index of each pair
        last_idx: dict[Tuple[str, str], int] = {}
        for i, item in enumerate(pairs):
            last_idx[item] = i
        out: list[Tuple[str, str]] = []
        for i, item in enumerate(pairs):
            if last_idx[item] == i:
                out.append(item)
        removed = len(pairs) - len(out)
        return out, removed

def tokenise_txt(inp:str,out:str,max_particles:int=8)->None:
    # Lazy import to avoid hard dependency at module import time
    import Tokenizer  # type: ignore
    tok = Tokenizer.ScatteringAmplitudeTokenizer(max_particles=max_particles)
    with open(inp,encoding="utf-8") as fi, open(out,"w",encoding="utf-8") as fo:
        for line in fi:
            s,t = line.rstrip("\n").split("\t")
            fo.write(json.dumps(tok.encode_infix(s))+"\t"+json.dumps(tok.encode_infix(t))+"\n")

# New: tokenise CSV using csv.reader/writer
def tokenise_csv(inp:str,out:str,max_particles:int=8)->None:
    # Lazy import to avoid hard dependency at module import time
    import Tokenizer  # type: ignore
    tok = Tokenizer.ScatteringAmplitudeTokenizer(max_particles=max_particles)
    with open(inp, newline="", encoding="utf-8") as fi, open(out, "w", newline="", encoding="utf-8") as fo:
        r = csv.reader(fi)
        w = csv.writer(fo)
        # Write header for tokenised output
        w.writerow(["simple", "scrambled"])
        first = True
        for row in r:
            if not row:
                continue
            # Skip input header if present
            if first and len(row) >= 2 and row[0].strip().lower() == "simple" and row[1].strip().lower() == "scrambled":
                first = False
                continue
            first = False
            s = row[0]
            t = row[1] if len(row) > 1 else ""
            w.writerow([json.dumps(tok.encode_infix(s)), json.dumps(tok.encode_infix(t))])

# ╭──────────────────────────────────────────────────────────────────╮
# │  Quick‑start driver                                              │
# ╰──────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate (simple,scrambled) amplitude expression pairs.")
    parser.add_argument('-N', type=int, default=4, help='Total legs (2 scalars + photons + scalar)')
    parser.add_argument('-n', '--nsamples', type=int, default=50000, help='Target number of pairs')
    parser.add_argument('--min-terms', type=int, default=1)
    parser.add_argument('--max-terms', type=int, default=6)
    parser.add_argument('--max-scrambles', type=int, default=5)
    parser.add_argument('--min-scrambles', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=1, help='>1 enables multiprocessing')
    parser.add_argument('--batch-size', type=int, default=250, help='Batch size per worker submission')
    parser.add_argument('--parity-checks', type=int, default=1, help='Numeric parity validation samples per expression')
    parser.add_argument('--scramble-checks', type=int, default=1, help='Numeric validation samples for scrambled equality')
    parser.add_argument('--no-validate', action='store_true', help='Disable numeric validation entirely')
    parser.add_argument('--no-denominators', action='store_true', help='Disable random denominators')
    parser.add_argument('--output-prefix', default='gi', help='Output file prefix base')
    args = parser.parse_args()

    N = args.N
    NSAMPLES = args.nsamples
    MAX_SCRAMBLES = args.max_scrambles
    MIN_SCRAMBLES = args.min_scrambles
    MIN_TERMS = args.min_terms
    MAX_TERMS = args.max_terms
    SEED = args.seed

    RAW = f"{args.output_prefix}_{N}pt.csv"
    TOK = f"{args.output_prefix}_{N}pt_tok.csv"
    LOG = f"gen_data_{N}pt.log"

    t0 = time.perf_counter()
    if args.workers > 1:
        pairs = build_dataset_parallel(
            N=N, target=NSAMPLES, workers=args.workers, batch_size=args.batch_size,
            max_scr=MAX_SCRAMBLES, min_scr=MIN_SCRAMBLES, seed=SEED,
            use_denominators=not args.no_denominators, validate=not args.no_validate,
            min_terms=MIN_TERMS, max_terms=MAX_TERMS,
            parity_checks=args.parity_checks, scramble_checks=args.scramble_checks
        )
    else:
        # Sequential path with tqdm progress if available
        pbar = None
        def _noop_progress(n: int) -> None:
            return None
        _prog = _noop_progress
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=NSAMPLES, desc='Generating', unit='pair')
            def _prog(n: int) -> None:  # type: ignore
                if pbar is not None:
                    pbar.n = n
                    pbar.refresh()
        except Exception:
            pbar = None
        oversampled = NSAMPLES  # oversampling less necessary with progress feedback
        pairs = build_dataset(
            N, num_samples=oversampled, max_scr=MAX_SCRAMBLES, min_scr=MIN_SCRAMBLES, seed=SEED,
            use_denominators=not args.no_denominators, validate=not args.no_validate,
            min_terms=MIN_TERMS, max_terms=MAX_TERMS,
            log_path=LOG, log_examples=5,
            parity_checks=args.parity_checks, scramble_checks=args.scramble_checks,
            progress_cb=_prog
        )
        if pbar is not None:
            pbar.close()
    t1 = time.perf_counter()

    before = len(pairs)
    pairs, removed = dedupe_pairs(pairs, keep="first")
    if len(pairs) > NSAMPLES:
        pairs = pairs[:NSAMPLES]
    final = len(pairs)
    write_csv(pairs, RAW)
    tokenise_csv(RAW, TOK)
    t2 = time.perf_counter()

    print(f"{final} pairs --> {RAW}")
    print(f"  generation : {(t1-t0):.2f}s")
    if args.workers > 1:
        print(f"  mode       : multiprocessing ({args.workers} workers, batch={args.batch_size})")
    else:
        print(f"  mode       : sequential")
    print(f"  dedupe     : removed {removed} duplicates (before={before})")
    print(f"  write+tok  : {(t2-t1):.2f}s")
    if not args.workers > 1:
        print(f"  log file   : {LOG}")
