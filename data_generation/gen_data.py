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
#               1…MAX_SCRAMBLES algebraic identities.
#
# Output: simple<TAB>scrambled   (plus tokenised file)

from __future__ import annotations
import random, re, time, json
from itertools import product
from typing import List, Tuple
import Tokenizer

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

def _rw_pFp(i,j,k):
    t1 = f"({dot(p(i), e(j))})*({dot(p(j), p(k))})"
    t2 = f"({dot(p(i), p(j))})*({dot(e(j), p(k))})"
    return f"{t1} - {t2}"

def _rw_pFFp(i,j,k,l):
    t1 = f"({dot(p(i), p(j))})*({dot(e(j), p(k))})*({dot(e(k), p(l))})"
    t2 = f"({dot(p(i), p(j))})*({dot(e(j), e(k))})*({dot(p(k), p(l))})"
    t3 = f"({dot(p(i), e(j))})*({dot(p(j), p(k))})*({dot(e(k), p(l))})"
    t4 = f"({dot(p(i), e(j))})*({dot(p(j), e(k))})*({dot(p(k), p(l))})"
    return f"{t1} - {t2} - {t3} + {t4}"

def _rw_Tr2(j,k):
    a = f"({dot(p(j), p(k))})*({dot(e(j), e(k))})"
    b = f"({dot(e(j), p(k))})*({dot(p(j), e(k))})"
    return f"2*({b} - {a})"

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
    expr = " ".join(terms).replace("  "," ")
    # move leading '+' if any
    return expr.lstrip("+ ")

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
# │  Scramblers (no symmetric‑dot)                                   │
# ╰──────────────────────────────────────────────────────────────────╯
def _mc_terms(idx:int,N:int) -> List[str]:
    return [f"-{dot(p(k),p(idx))}" for k in range(1,N+1) if k!=idx]

def scr_mul_by_one(expr:str,N:int)->str:
    i,j = random.sample(range(1,N+1),2)
    numerator = " ".join(_mc_terms(i,N))
    denominator = dot(p(i),p(j))
    return f"({expr})*(({numerator}))/({denominator})"

def scr_add_zero_gauge(expr:str,Ngamma:int,N:int)->str:
    i = random.randint(2,Ngamma+1)
    rhs = " ".join(f"-{dot(e(i),p(k))}" for k in range(1,N+1) if k!=i)
    term = f"{dot(e(i),p(i))} + ({rhs})"
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

def scramble(expr:str,Ngamma:int,N:int,max_scr:int)->str:
    n = random.randint(0, max_scr) if max_scr > 0 else 0
    out = expr
    for _ in range(n):
        out = random.choice(_SCRAMBLERS)(out,Ngamma,N)
    return out

# ╭──────────────────────────────────────────────────────────────────╮
# │  Dataset construction & I/O                                      │
# ╰──────────────────────────────────────────────────────────────────╯
def build_dataset(N:int, num_samples:int, max_scr:int=3) -> List[Tuple[str,str]]:
    Ngamma = N-2
    data=[]
    for _ in range(num_samples):
        gi = strict_gi_monomial(N)
        expd = "*".join(rewrite_gi(b) for b in gi.split("*"))
        data.append((gi, scramble(expd,Ngamma,N,max_scr)))
    return data

def write_txt(pairs:List[Tuple[str,str]],path:str)->None:
    with open(path,"w",encoding="utf-8") as f:
        for s,t in pairs: f.write(f"{s}\t{t}\n")

def tokenise_txt(inp:str,out:str,max_particles:int=8)->None:
    tok = Tokenizer.ScatteringAmplitudeTokenizer(max_particles=max_particles)
    with open(inp,encoding="utf-8") as fi, open(out,"w",encoding="utf-8") as fo:
        for line in fi:
            s,t = line.rstrip("\n").split("\t")
            fo.write(json.dumps(tok.encode_infix(s))+"\t"+json.dumps(tok.encode_infix(t))+"\n")

# ╭──────────────────────────────────────────────────────────────────╮
# │  Quick‑start driver                                              │
# ╰──────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    N              = 5       # p_1 φ , p_2‑p_{n-1} γ , p_n φ
    NSAMPLES       = 100
    MAX_SCRAMBLES  = 5

    RAW = f"gi_{N}pt.txt"
    TOK = f"gi_{N}pt_tok.txt"

    t0 = time.perf_counter()
    pairs = build_dataset(N, num_samples=NSAMPLES, max_scr=MAX_SCRAMBLES)
    t1 = time.perf_counter()
    write_txt(pairs, RAW)
    tokenise_txt(RAW, TOK)
    t2 = time.perf_counter()

    print(f"✓ {len(pairs)} pairs → {RAW}")
    print(f"  generation : {(t1-t0):.2f}s")
    print(f"  write+tok  : {(t2-t1):.2f}s")
