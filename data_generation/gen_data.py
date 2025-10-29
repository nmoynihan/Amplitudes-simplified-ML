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
import random, re, time, json, math, csv, sys
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

def _rw_pFPRODp(*idxs):
    """
    Expand p_i · F_{j1} · F_{j2} · ... · F_{jn} · p_k
    where F_m = p_m ⊗ e_m − e_m ⊗ p_m, and 'dot' contracts consecutive vectors.

    Args: (i, j1, j2, ..., jn, k)
    """
    assert len(idxs) >= 3, "Need (i, j1, ..., jn, k)"
    i, *Fs, k = idxs
    n = len(Fs)

    def vec(tag, t):
        return p(t) if tag == 'p' else e(t)

    terms = []
    for mask in range(1 << n):
        # sign = (-1)^{# of 'swap' choices}  (i.e. how many times we take -e⊗p)
        minus = (mask.bit_count() % 2) == 1
        factors = []
        prev = ('p', i)

        for bit, j in enumerate(Fs):
            choose_swap = (mask >> bit) & 1  # 0 -> (p_j, e_j), 1 -> (e_j, p_j)
            L = ('e', j) if choose_swap else ('p', j)
            R = ('p', j) if choose_swap else ('e', j)
            factors.append(f"({dot(vec(*prev), vec(*L))})")
            prev = R

        factors.append(f"({dot(vec(*prev), p(k))})")
        term = "*".join(factors)
        terms.append((-1 if minus else 1, term))

    # Assemble with signs to match your style: (t1 - t2 - t3 + t4)
    pieces = []
    for sgn, t in terms:
        if not pieces:
            pieces.append(t if sgn > 0 else f"-{t}")
        else:
            pieces.append(("+ " if sgn > 0 else "- ") + t)

    return "(" + " ".join(pieces) + ")"

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

# Paolo: function below is inside eval_infix_numeric but I copied it here so I can call it from testing code
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

            # --- general n ≥ 3 ---
            # n ≥ 3: capture the whole chain once, then extract all F indices
            def rep_pf_chain(m):
                nonlocal changed
                changed = True
                i = int(m.group(1))
                chain = m.group(2)           # the whole "· F_x · F_y · F_z ..." substring
                k = int(m.group(4))
                Fs = list(map(int, re.findall(r"F_(\d+)", chain)))
                # Guard (should be ≥3 by pattern, but harmless):
                if len(Fs) < 3:
                    return m.group(0)
                return _rw_pFPRODp(i, *Fs, k)

            # Match any length ≥ 3 in one shot
            pattern_n_ge_3 = r"p_(\d+)((?:\s*·\s*F_(\d+)){3,})\s*·\s*p_(\d+)"
            s_new = re.sub(pattern_n_ge_3, rep_pf_chain, s_new)

            s = s_new
            
        return s

def eval_infix_numeric(expr: str, momenta, pols) -> float:
    """Parse -> expand GI blocks into AST -> numerically evaluate.

    This replaces fragile text substitutions with an AST-based expansion so
    operator binding is preserved.
    """
    # Build name->vectors maps
    N = len(momenta)
    P = {f"p_{i}": momenta[i-1] for i in range(1, N+1)}
    E = {f"e_{i}": pols[i-2] for i in range(2, N)}

    # --- small AST representation ------------------------------------------------
    class ASTNode: pass

    class Number(ASTNode):
        def __init__(self, v: float): self.v = float(v)

    class Vec(ASTNode):
        def __init__(self, tag: str, idx: int):
            self.tag = tag  # 'p' or 'e' or 'F'
            self.idx = idx

    class DotChain(ASTNode):
        def __init__(self, parts: list):
            # parts: list of Vec nodes (p/e/F)
            self.parts = parts

    class BinOp(ASTNode):
        def __init__(self, op: str, left: ASTNode, right: ASTNode):
            self.op = op
            self.left = left
            self.right = right

    class UnaryOp(ASTNode):
        def __init__(self, op: str, operand: ASTNode):
            self.op = op
            self.operand = operand

    # --- tokenizer ---------------------------------------------------------------
    import re
    token_re = re.compile(r"\s*(\d+\.\d+|\d+|p_\d+|e_\d+|F_\d+|Tr\b|\*\*|\^|\+|\-|\*|/|\(|\)|\.|·|,)")

    def tokenize(s: str):
        s = s.replace('^', '**')
        pos = 0
        toks = []
        while pos < len(s):
            m = token_re.match(s, pos)
            if not m:
                # try identifier like Tr(F_2 · F_3)
                # capture names like Tr or bare words
                m2 = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", s[pos:])
                if m2:
                    toks.append(m2.group(1))
                    pos += m2.end()
                    continue
                # otherwise skip troublesome characters (like '·' handled above)
                toks.append(s[pos])
                pos += 1
            else:
                tok = m.group(1)
                toks.append(tok)
                pos = m.end()
        return toks

    # --- recursive-descent parser (handles + - * / and implicit dot-chains) -------
    class Parser:
        def __init__(self, toks):
            self.toks = toks
            self.i = 0

        def peek(self):
            return self.toks[self.i] if self.i < len(self.toks) else None
        def pop(self):
            t = self.peek(); self.i += 1; return t

        def parse(self):
            return self.parse_expr()

        def parse_expr(self):
            node = self.parse_term()
            while True:
                t = self.peek()
                if t == '+' or t == '-':
                    op = self.pop()
                    right = self.parse_term()
                    node = BinOp(op, node, right)
                else:
                    break
            return node

        def parse_term(self):
            node = self.parse_factor()
            while True:
                t = self.peek()
                if t == '*' or t == '/':
                    op = self.pop()
                    right = self.parse_factor()
                    node = BinOp(op, node, right)
                else:
                    break
            return node

        def parse_factor(self):
            t = self.peek()
            if t == '+':
                self.pop(); return self.parse_factor()
            if t == '-':
                self.pop(); return UnaryOp('-', self.parse_factor())
            node = self.parse_power()
            return node

        def parse_power(self):
            node = self.parse_primary()
            while True:
                t = self.peek()
                if t == '**':
                    self.pop(); right = self.parse_primary(); node = BinOp('**', node, right)
                else:
                    break
            return node

        def parse_primary(self):
            t = self.peek()
            if t == '(':
                self.pop()
                node = self.parse_expr()
                if self.peek() == ')': self.pop()
                return node
            if t == 'Tr':
                # consume Tr ( ... )
                self.pop()
                if self.peek() == '(':
                    self.pop()
                    # collect inside until matching ) as raw tokens for further processing
                    # parse a comma or dot-separated list of F_i chains
                    args = []
                    while True:
                        # parse possible F chain like F_2 · F_3 · F_4
                        parts = []
                        while True:
                            tok = self.peek()
                            if tok and re.match(r'F_\d+', str(tok)):
                                parts.append(self.pop())
                                if self.peek() == '·' or self.peek() == '.':
                                    self.pop(); continue
                                else:
                                    break
                            else:
                                break
                        args.extend(parts)
                        if self.peek() == ',': self.pop(); continue
                        break
                    if self.peek() == ')': self.pop()
                    # represent Tr as a special DotChain with 'Tr' marker
                    vecs = [Vec('F', int(re.search(r'\d+', p).group())) for p in args]
                    return DotChain(vecs + ['__TR__'])
            # parse a possible dot-chain: sequence of p_/e_/F_ separated by '·' or '.'
            parts = []
            while True:
                tok = self.peek()
                if tok and re.match(r'(p_|e_|F_)\d+', str(tok)):
                    m = re.match(r'(p_|e_|F_)(\d+)', tok)
                    tag = m.group(1)[0]
                    idx = int(m.group(2))
                    parts.append(Vec(tag, idx))
                    self.pop()
                    if self.peek() in ('·', '.'):
                        self.pop(); continue
                    else:
                        break
                else:
                    break
            if parts:
                # single identifier or dot-chain
                if len(parts) == 1:
                    return parts[0]
                return DotChain(parts)
            # number literal
            if t and re.match(r'\d+(?:\.\d+)?', str(t)):
                self.pop(); return Number(float(t))
            # fallback: consume token and treat as identifier (shouldn't happen)
            if t is not None:
                self.pop(); return Number(0.0)
            return Number(0.0)

    # --- AST expansion: convert DotChain/Tr and F nodes into arithmetic AST using dot products ----
    def mk_dot(lhs: ASTNode, rhs: ASTNode):
        # returns AST node representing numeric dot(lhs, rhs) using _mdot at eval time
        return ('DOT', lhs, rhs)

    def ast_add(a, b): return BinOp('+', a, b)
    def ast_sub(a, b): return BinOp('-', a, b)
    def ast_mul(a, b): return BinOp('*', a, b)
    def ast_div(a, b): return BinOp('/', a, b)

    # helpers to build Vec AST wrappers for dot operands
    def vec_node(v: Vec):
        return ('VEC', v.tag, v.idx)

    def expand_dotchain(dc: DotChain):
        # dc.parts is list of Vec nodes or special Tr marker
        parts = dc.parts
        # If this is a Tr marker at end
        if parts and parts[-1] == '__TR__':
            # parts before last are F vectors
            Fs = [p for p in parts[:-1]]
            # If length 2 -> Tr2, length 3 -> Tr3
            if len(Fs) == 2:
                j = Fs[0].idx; k = Fs[1].idx
                # produce AST for 2*( (e_j·p_k)*(p_j·e_k) - (p_j·p_k)*(e_j·e_k) ) with sign per _rw_Tr2
                # _rw_Tr2 returns 2*(b - a) where a=(p_j·p_k)*(e_j·e_k), b=(e_j·p_k)*(p_j·e_k)
                a = ast_mul(mk_dot(vec_node(Vec('p', j)), vec_node(Vec('p', k))), mk_dot(vec_node(Vec('e', j)), vec_node(Vec('e', k))))
                b = ast_mul(mk_dot(vec_node(Vec('e', j)), vec_node(Vec('p', k))), mk_dot(vec_node(Vec('p', j)), vec_node(Vec('e', k))))
                return ast_mul(Number(2.0), ast_sub(b, a))
            elif len(Fs) == 3:
                # expand Tr3 by enumerating sign choices like current string-based _rw_Tr3
                j,k,l = Fs[0].idx, Fs[1].idx, Fs[2].idx
                terms = []
                from itertools import product
                for s1,s2,s3 in product((0,1), repeat=3):
                    sign = -1 if (s1+s2+s3)%2 else 1
                    a1 = ('p' if s1==0 else 'e', j)
                    b1 = ('e' if s1==0 else 'p', j)
                    a2 = ('p' if s2==0 else 'e', k)
                    b2 = ('e' if s2==0 else 'p', k)
                    a3 = ('p' if s3==0 else 'e', l)
                    b3 = ('e' if s3==0 else 'p', l)
                    # term = (b1·a2)*(b2·a3)*(a1·b3)
                    t = ast_mul(ast_mul(mk_dot(vec_node(Vec(b1[0], b1[1])), vec_node(Vec(a2[0], a2[1]))), mk_dot(vec_node(Vec(b2[0], b2[1])), vec_node(Vec(a3[0], a3[1])))), mk_dot(vec_node(Vec(a1[0], a1[1])), vec_node(Vec(b3[0], b3[1]))))
                    terms.append((sign, t))
                # sum up with signs
                acc = None
                for sgn, t in terms:
                    if acc is None:
                        acc = t if sgn>0 else UnaryOp('-', t)
                    else:
                        acc = ast_add(acc, t) if sgn>0 else ast_sub(acc, t)
                return acc
        # otherwise it's a chain like p_i · F_j · F_k · p_l or p_i · p_j etc.
        # find indices of any F in the chain
        Fs = [p for p in parts if isinstance(p, Vec) and p.tag == 'F']
        Ps = [p for p in parts if isinstance(p, Vec) and p.tag in ('p','e')]
        # handle simple p·p or p·e dot sequences as product of dot ops
        # If any F present, use the pF...p identities
        if any(isinstance(p, Vec) and p.tag == 'F' for p in parts):
            # Expect first and last to be p
            # extract indices of p at ends and list of F indices
            if not (isinstance(parts[0], Vec) and parts[0].tag == 'p' and isinstance(parts[-1], Vec) and parts[-1].tag == 'p'):
                # fallback: treat as zero
                return Number(0.0)
            i = parts[0].idx
            k = parts[-1].idx
            F_idxs = [p.idx for p in parts if p.tag == 'F']
            if len(F_idxs) == 1:
                j = F_idxs[0]
                # p_i·F_j·p_k = (p_i·p_j)*(e_j·p_k) - (p_i·e_j)*(p_j·p_k)
                term1 = ast_mul(mk_dot(vec_node(Vec('p', i)), vec_node(Vec('p', j))), mk_dot(vec_node(Vec('e', j)), vec_node(Vec('p', k))))
                term2 = ast_mul(mk_dot(vec_node(Vec('p', i)), vec_node(Vec('e', j))), mk_dot(vec_node(Vec('p', j)), vec_node(Vec('p', k))))
                return ast_sub(term1, term2)
            elif len(F_idxs) == 2:
                j, m = F_idxs[0], F_idxs[1]
                # p_i·F_j·F_m·p_k per _rw_pFFp
                t1 = ast_mul(ast_mul(mk_dot(vec_node(Vec('p', i)), vec_node(Vec('p', j))), mk_dot(vec_node(Vec('e', j)), vec_node(Vec('p', m)))), mk_dot(vec_node(Vec('e', m)), vec_node(Vec('p', k))))
                t2 = ast_mul(ast_mul(mk_dot(vec_node(Vec('p', i)), vec_node(Vec('p', j))), mk_dot(vec_node(Vec('e', j)), vec_node(Vec('e', m)))), mk_dot(vec_node(Vec('p', m)), vec_node(Vec('p', k))))
                t3 = ast_mul(ast_mul(mk_dot(vec_node(Vec('p', i)), vec_node(Vec('e', j))), mk_dot(vec_node(Vec('p', j)), vec_node(Vec('p', m)))), mk_dot(vec_node(Vec('e', m)), vec_node(Vec('p', k))))
                t4 = ast_mul(ast_mul(mk_dot(vec_node(Vec('p', i)), vec_node(Vec('e', j))), mk_dot(vec_node(Vec('p', j)), vec_node(Vec('e', m)))), mk_dot(vec_node(Vec('p', m)), vec_node(Vec('p', k))))
                # (t1 - t2 - t3 + t4)
                return ast_add(ast_sub(ast_sub(t1, t2), t3), t4)
            else:
                # general case p_i · F_j1 · ... · F_jn · p_k: implement via expansion over mask
                i = parts[0].idx
                k = parts[-1].idx
                idxs = [p.idx for p in parts if p.tag == 'F']
                terms = []
                for mask in range(1 << len(idxs)):
                    minus = (mask.bit_count() % 2) == 1
                    factors = []
                    prev = ('p', i)
                    for bit, j in enumerate(idxs):
                        choose_swap = (mask >> bit) & 1
                        L = ('e', j) if choose_swap else ('p', j)
                        R = ('p', j) if choose_swap else ('e', j)
                        factors.append(mk_dot(vec_node(Vec(prev[0], prev[1])), vec_node(Vec(L[0], L[1]))))
                        prev = R
                    factors.append(mk_dot(vec_node(Vec(prev[0], prev[1])), vec_node(Vec('p', k))))
                    # multiply factors
                    acc = None
                    for f in factors:
                        acc = f if acc is None else ast_mul(acc, f)
                    terms.append((-1 if minus else 1, acc))
                acc = None
                for sgn, t in terms:
                    if acc is None:
                        acc = t if sgn>0 else UnaryOp('-', t)
                    else:
                        acc = ast_add(acc, t) if sgn>0 else ast_sub(acc, t)
                return acc
        else:
            # purely p/e dot chain like p_a · p_b · p_c -> multiply successive dot products? For our use,
            # p·p·p not meaningful; but patterns in code are p·p or e·p etc. For safety, if length==2 return dot, else chain as product
            acc = None
            for a,b in zip(parts, parts[1:]):
                d = mk_dot(vec_node(a), vec_node(b))
                acc = d if acc is None else ast_mul(acc, d)
            return acc if acc is not None else Number(0.0)

    # --- evaluator: compute numeric value from expanded AST --------------------------------
    def eval_ast(node):
        # node can be Number, BinOp, UnaryOp, or tuple ('DOT', lhs, rhs) or ('VEC', tag, idx)
        if isinstance(node, Number):
            return float(node.v)
        if isinstance(node, BinOp):
            l = eval_ast(node.left)
            r = eval_ast(node.right)
            if node.op == '+': return l + r
            if node.op == '-': return l - r
            if node.op == '*': return l * r
            if node.op == '/': return l / r
            if node.op == '**': return l ** r
        if isinstance(node, UnaryOp):
            v = eval_ast(node.operand)
            if node.op == '-': return -v
            return v
        if isinstance(node, tuple):
            if node[0] == 'DOT':
                # node = ('DOT', lhs, rhs) where lhs/rhs are ('VEC', tag, idx) or nested dot/expr
                lhs = node[1]; rhs = node[2]
                # lhs/rhs expected to be ('VEC', tag, idx)
                if lhs[0] == 'VEC' and rhs[0] == 'VEC':
                    tagl, il = lhs[1], lhs[2]
                    tagr, ir = rhs[1], rhs[2]
                    if tagl == 'p' and tagr == 'p':
                        return _mdot(P[f'p_{il}'], P[f'p_{ir}'])
                    if tagl == 'e' and tagr == 'p':
                        return _mdot(E[f'e_{il}'], P[f'p_{ir}'])
                    if tagl == 'p' and tagr == 'e':
                        return _mdot(P[f'p_{il}'], E[f'e_{ir}'])
                    if tagl == 'e' and tagr == 'e':
                        return _mdot(E[f'e_{il}'], E[f'e_{ir}'])
                    # F tags shouldn't appear here
                # fallback
                return float(0.0)
            if node[0] == 'VEC':
                # a bare vector used where a numeric needed -> not allowed
                return float(0.0)
        # Unknown
        return float(0.0)

    # --- top-level: parse, expand nodes, then evaluate -------------------------------------
    toks = tokenize(expr)
    parser = Parser(toks)
    ast_root = parser.parse()

    # walk AST and replace DotChain nodes / Tr nodes with expanded AST pieces
    def expand_node(n):
        if isinstance(n, Number): return n
        if isinstance(n, Vec): return n
        if isinstance(n, DotChain):
            return expand_dotchain(n)
        if isinstance(n, UnaryOp):
            return UnaryOp(n.op, expand_node(n.operand))
        if isinstance(n, BinOp):
            return BinOp(n.op, expand_node(n.left), expand_node(n.right))
        # tuple nodes
        if isinstance(n, tuple):
            return n
        return n

    ast_expanded = expand_node(ast_root)
    # Now evaluate numeric value
    val = eval_ast(ast_expanded)
    return float(val)

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
                return f"({ _rw_Tr2(j,k) })"
            s_new = re.sub(r"Tr\(\(?F_(\d+)\s*·\s*F_(\d+)\)?\)", rep2, s)
            def rep3(m):
                nonlocal changed
                changed = True
                j,k,l = map(int, m.groups())
                return f"({ _rw_Tr3(j,k,l) })"
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
                return f"({ _rw_pFp(i, j, k) })"
            s_new = re.sub(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)", rep_pfp, s)
            def rep_pffp(m):
                nonlocal changed
                changed = True
                i, j, k, l = map(int, m.groups())
                return f"({ _rw_pFFp(i, j, k, l) })"
            s_new = re.sub(r"p_(\d+)\s*·\s*F_(\d+)\s*·\s*F_(\d+)\s*·\s*p_(\d+)", rep_pffp, s_new)
            s = s_new
        return s

    # For debug: produce a numeric-ready string by re-using the AST pipeline above
    try:
        # reuse eval_infix_numeric's tokenizer/parser/expander by calling it indirectly
        toks = re.findall(r"\S+", expr)
        # fallback: return the old style expansion
        expr_expanded = expr
        expr_expanded = _expand_traces(expr_expanded)
        expr_expanded = _expand_gi_blocks(expr_expanded)
        expr_f = _to_float_expr(expr_expanded, P, E)
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
    except Exception:
        # best-effort fallback
        return expr


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
                  log_examples:int=5) -> List[Tuple[str,str]]:
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
            for _ in range(2):  # a couple of random kinematic samples are usually enough
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
            for _ in range(3):
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
    N              = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # p_1 φ , p_2‑p_{n-1} γ , p_n φ
    NSAMPLES       = 50000
    MAX_SCRAMBLES  = 5 
    MIN_SCRAMBLES  = 0 # 0 means no scrambling, just expansion
    # --- New polynomial controls -------------------------------------------------
    MIN_TERMS      = 1  # =1 recovers legacy single-monomial behaviour
    MAX_TERMS      = 6  # choose >1 to enable polynomial generation
    SEED           = 42

    RAW = f"gi_{N}pt.csv"
    TOK = f"gi_{N}pt_tok.csv"

    t0 = time.perf_counter()
    # Oversample by +20% to compensate for duplicates
    target = NSAMPLES
    oversampled = int(round(target * 1.2))
    LOG = f"gen_data_{N}pt.log"
    pairs = build_dataset(N, num_samples=oversampled, max_scr=MAX_SCRAMBLES, min_scr=MIN_SCRAMBLES, seed=SEED,
                          use_denominators=True, validate=True,
                          min_terms=MIN_TERMS, max_terms=MAX_TERMS,
                          log_path=LOG, log_examples=5)
    t1 = time.perf_counter()
    # Deduplicate in-memory before writing; keep first occurrences
    before = len(pairs)
    pairs, removed = dedupe_pairs(pairs, keep="first")
    after = len(pairs)
    # Truncate to target size after dedupe
    if len(pairs) > target:
        pairs = pairs[:target]
    final = len(pairs)
    write_csv(pairs, RAW)
    tokenise_csv(RAW, TOK)
    t2 = time.perf_counter()

    print(f"{len(pairs)} pairs --> {RAW}")
    print(f"  generation : {(t1-t0):.2f}s")
    print(f"  oversample : requested {target}, generated {oversampled}")
    print(f"  dedupe     : removed {removed} duplicates ({before} -> {after})")
    print(f"  truncate   : final {final} rows written")
    print(f"  write+tok  : {(t2-t1):.2f}s")
    print(f"  log file   : {LOG}")
