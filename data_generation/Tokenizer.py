import re
from typing import List, Dict, Tuple, Any, Optional


class ScatteringAmplitudeTokenizer:
    # Some sexy regexes to match tokens in the input expression
    # p_1, e_1, F_1 are the particle, polarisation and field strength tokens, respectively.
    # Tr is the trace operator.
    _token_re = re.compile(
        r'p_\d+|e_\d+|F_\d+|M_\d+'
        r'|Tr'
        r'|M' # mass, should probably not be used in expressions (favour p_1.p_1)
        r'|\d+'                     # one-or-more digits
        r'|[+\-*/^·().]'
    )

    # ──────────────────────────────────────────────────────────────────── #
    # Construction
    # ──────────────────────────────────────────────────────────────────── #
    def __init__(self, *, max_particles: int = 8, max_sequence_length: int = 2048):
        self.max_particles = max_particles
        self.max_sequence_length = max_sequence_length
        
        self.vocab_init = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "+": 4,
            "-": 5,
            "*": 6,
            "/": 7,
            "^": 8,
            "(": 9,
            ")": 10,
            "0:": 11,
            "1:": 12,
            "2:": 13,
            "3:": 14,
            "4:": 15,
            "5:": 16,
            "6:": 17,
            "7:": 18,
            "8:": 19,
            "9:": 20,
            "10:": 21,
            "·": 22, # dot operator.
            "Tr": 23,
            "u-": 24,  # unary minus
            "M": 25,  # mass. Probably best not used!
        }

        nxt = max(self.vocab_init.values()) + 1
        self.p_tokens = {f"p_{i}": nxt + i - 1 for i in range(1, max_particles + 1)}
        self.e_tokens = {f"e_{i}": nxt + max_particles + i - 1
                         for i in range(1, max_particles + 1)}
        self.f_tokens = {f"F_{i}": nxt + 2 * max_particles + i - 1
                         for i in range(1, max_particles + 1)}
        self.m_tokens = {f"M_{i}": nxt + 3 * max_particles + i - 1
                         for i in range(1, max_particles + 1)}
        self.vocab: Dict[str, int] = {
            **self.vocab_init, **self.p_tokens, **self.e_tokens, **self.f_tokens, **self.m_tokens
        }
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        print(f"Tokenizer vocabulary size: {len(self.vocab)}")

        # precedence & arity tables
        # airity is basically the number of objects the operator eats, i.e. +-* eats two, Tr eats one
        # precedence is the order of operations, i.e. Tr > ^ > * > / > + > -
        self._prec = {"+":1, "-":1, "*":2, "/":2, "·":2, "^":3, "u-":4, "Tr":5}
        self._arity = {op: 2 for op in ["+", "-", "*", "/", "·", "^"]}
        self._arity["Tr"] = 1
        # We also define a unary minus operator for when it appears at the start of an expression or after an operator
        self._arity["u-"] = 1

    # ──────────────────────────────────────────────────────────────────── #
    # Public helpers
    # ──────────────────────────────────────────────────────────────────── #
    def encode_infix(self, expr: str) -> List[int]:
        tokens = self._to_prefix(expr)
        result = []
        unknown_tokens = []
        
        for tok in tokens:
            if tok in self.vocab:
                result.append(self.vocab[tok])
            else:
                result.append(self.vocab["<UNK>"])
                unknown_tokens.append(tok)
        
        if unknown_tokens:
            raise ValueError(f"Unknown tokens found in expression '{expr}': {unknown_tokens}")
        
        return result

    def decode_prefix(self, ids: List[int]) -> str:
        return " ".join(self.id_to_token[i] for i in ids if i != self.vocab["<PAD>"])

    def decode_infix(self, ids: List[int]) -> str:
        toks = [self.id_to_token[i] for i in ids if i not in {self.vocab["<PAD>"], self.vocab["<BOS>"], self.vocab["<EOS>"]}]

        def _needs_parens(expr: str, parent_prec: int) -> bool:
            """Check if an expression needs parentheses based on its content"""
            # Simple heuristic: if it contains operators with lower precedence
            for op in ['+', '-']:
                if op in expr and self._prec[op] < parent_prec:
                    return True
            return False

        def _parse(idx: int, parent_prec: int = 0, is_right: bool = False) -> Tuple[str, int]:
            if idx >= len(toks):
                raise ValueError("Malformed prefix tokens: ran out of tokens.")

            tok = toks[idx]
            
            # operators --------------------------------------------------- #
            if tok == "Tr":                     # unary trace operator
                child, nxt = _parse(idx + 1, self._prec[tok])
                return f"Tr({child})", nxt
            elif tok == 'u-':                   # unary minus
                child, nxt = _parse(idx + 1, self._prec[tok])
                # Only add parentheses if child has lower precedence
                if _needs_parens(child, self._prec[tok]):
                    return f"-({child})", nxt
                else:
                    return f"-{child}", nxt
            elif tok in self._arity:            # binary operators
                left, nxt = _parse(idx + 1, self._prec[tok], False)
                right, nxt = _parse(nxt, self._prec[tok], True)
                
                # Determine if we need parentheses around the whole expression
                current_prec = self._prec[tok]
                needs_parens = (current_prec < parent_prec or 
                               (current_prec == parent_prec and is_right and tok in ['-', '/', '^']))
                
                # Format with appropriate spacing
                if tok in ['+', '-']:  # Binary + and - get spaces
                    expr = f"{left} {tok} {right}"
                else:  # All other operators (*, /, ^, ·) get no spaces
                    expr = f"{left}{tok}{right}"
                
                return f"({expr})" if needs_parens else expr, nxt

            # leaf (maybe part of a multi-digit constant) ---------------- #
            if tok.endswith(":") and tok[:-1].isdigit():
                digits = [tok[:-1]]
                j = idx + 1
                while (j < len(toks)
                       and toks[j].endswith(":")
                       and toks[j][:-1].isdigit()):
                    digits.append(toks[j][:-1])
                    j += 1
                return "".join(digits), j
            return tok, idx + 1

        expr, final_idx = _parse(0)
        if final_idx != len(toks):
            raise ValueError(f"Prefix stream not fully consumed: stopped at {final_idx}")
        return expr[1:-1] if expr.startswith("(") and expr.endswith(")") else expr

    # ──────────────────────────────────────────────────────────────────── #
    # internals: infix → prefix
    # ──────────────────────────────────────────────────────────────────── #
    def _tokenise(self, expr: str) -> List[str]:
        raw = self._token_re.findall(expr.replace(" ", ""))
        norm = []
        for t in raw:
            if t == '.':                      # ASCII "." → middle-dot
                norm.append('·')
            elif t.isdigit():                 # split integer into digit-tokens
                norm.extend(f"{d}:" for d in t)
            else:
                norm.append(t)
        return norm

    @staticmethod
    def _is_digit(tok: str) -> bool:
        return tok.endswith(":") and tok[:-1].isdigit()

    def _insert_implicit_mul(self, toks: List[str]) -> List[str]:
        res = []
        for i, t in enumerate(toks):
            if i:
                left, right = toks[i - 1], t
                left_val = left not in self._prec and left != "("
                right_val = right not in self._prec and right != ")"
                # skip digit–digit adjacency (part of the same number)
                if (left_val and (right_val or right in ("(", "Tr"))
                        and not (self._is_digit(left) and self._is_digit(right))):
                    res.append("*")
            res.append(t)
        return res

    def _detect_unary_minus(self, tokens: List[str]) -> List[str]:
        res = []
        for i, t in enumerate(tokens):
            if t == '-':
                #print(f"Detected unary minus at position {i} in tokens: {tokens}")
                if i == 0 or tokens[i - 1] in self._prec or tokens[i - 1] == '(':
                    #print(f"  → treating as unary minus")
                    res.append('u-')
                else:
                    res.append('-')
            else:
                res.append(t)
        #print(f"Tokens after unary minus detection: {res}")
        return res

    def _to_prefix(self, expr: str) -> List[str]:
        infix = self._insert_implicit_mul(self._tokenise(expr))
        infix = self._detect_unary_minus(infix)
        
        # Reverse the infix expression and swap parentheses
        infix = [")" if t == "(" else "(" if t == ")" else t for t in infix[::-1]]
        
        # shunting-yard on reversed stream
        out, stack = [], []
        for tok in infix:
            if tok in self._prec or tok in ("(", ")"):
                if tok == "(":
                    stack.append(tok)
                elif tok == ")":
                    while stack and stack[-1] != "(":
                        out.append(stack.pop())
                    if stack:
                        stack.pop()
                else:  # real operator
                    while (stack and stack[-1] != "("
                        and stack[-1] in self._prec
                        and self._prec[stack[-1]] > self._prec[tok]):
                        out.append(stack.pop())
                    stack.append(tok)
            else:
                out.append(tok)
        out.extend(reversed(stack))           # drain
        return out[::-1]                      # postfix → prefix

    def debug_tokenization(self, expr: str) -> Dict[str, Any]:
        """Debug method to see all tokenization steps"""
        raw_tokens = self._tokenise(expr)
        with_implicit_mul = self._insert_implicit_mul(raw_tokens)
        with_unary_minus = self._detect_unary_minus(with_implicit_mul)
        prefix_tokens = self._to_prefix(expr)
        
        unknown_tokens = [tok for tok in prefix_tokens if tok not in self.vocab]
        
        return {
            "original": expr,
            "raw_tokens": raw_tokens,
            "with_implicit_mul": with_implicit_mul,
            "with_unary_minus": with_unary_minus,
            "prefix_tokens": prefix_tokens,
            "unknown_tokens": unknown_tokens,
            "vocab_keys": list(self.vocab.keys())
        }


# ──────────────────────────────────────────────────────────────────── #
# Numerical equivalence helper                                        #
# ──────────────────────────────────────────────────────────────────── #
def numerically_equivalent(
    tokenizer: ScatteringAmplitudeTokenizer,
    a_tokens,
    b_tokens,
    N: int,
    *,
    samples: int = 3,
    M: float = 2.0,
    tol_abs: float = 1e-12,
    tol_rel: float = 1e-10,
    seed: Optional[int] = None,
    return_details: bool = False,
):
    """Compare two (tokenised) expressions for numerical equivalence.

    Parameters
    ----------
    tokenizer : ScatteringAmplitudeTokenizer
        Instance used to decode the token id streams back to infix.
    a_tokens , b_tokens : sequence[int] | str
        Either
          * a list/tuple of integer token ids, OR
          * a whitespace separated string of token *lexemes* (e.g. "p_1 · p_2 + p_3 · p_4"), OR
          * an infix expression itself (heuristically detected).
        In all cases the arguments are decoded to an infix expression that can be
        numerically evaluated.
    N : int
        Total number of external legs (needed to generate kinematics).
    samples : int, default 3
        How many random phase–space points to test over.
    M : float, default 2.0
        Mass of legs p_1 and p_N passed to the kinematics generator.
    tol_abs : float, default 1e-12
        Absolute tolerance threshold.
    tol_rel : float, default 1e-10
        Relative tolerance threshold (compared against max(|a|,|b|,1)).
    seed : int | None
        Optional base RNG seed for reproducibility; each sample uses seed+idx.
    return_details : bool, default False
        If True, also return a details dict with per-sample values & diffs.

    Returns
    -------
    bool | (bool, dict)
        True if all sampled evaluations agree within tolerances; otherwise False.
        If return_details=True returns (result, details_dict).

    Notes
    -----
    This function lazily imports `gen_data` and `kinematics` to avoid any
    potential circular import during module initialisation.
    """
    # Lazy, robust imports (works whether run as a module or script)
    import importlib, importlib.util, os, sys

    def _lazy_local_import(mod_name: str):
        try:
            return importlib.import_module(mod_name)
        except ModuleNotFoundError:  # fall back to same-directory file import
            here = os.path.dirname(__file__)
            candidate = os.path.join(here, f"{mod_name}.py")
            if not os.path.isfile(candidate):
                raise
            spec = importlib.util.spec_from_file_location(mod_name, candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                return module
            raise

    _gd = _lazy_local_import("gen_data")  # type: ignore
    _km = _lazy_local_import("kinematics")  # type: ignore

    def _coerce(expr_like):
        # If already a list/tuple of ints assume token ids
        if isinstance(expr_like, (list, tuple)) and all(isinstance(x, int) for x in expr_like):
            return tokenizer.decode_infix(list(expr_like))
        # If it's a string we try several interpretations
        if isinstance(expr_like, str):
            s = expr_like.strip()
            # Heuristic: if it contains "p_" or "e_" or operators typical of infix, treat directly
            if any(sym in s for sym in ["p_", "e_", "F_", "Tr", "·", "+", "-", "*", "/", "^"]):
                return s
            # Else treat as space separated tokens -> map to IDs -> decode
            toks = s.split()
            ids = []
            for t in toks:
                if t in tokenizer.vocab:
                    ids.append(tokenizer.vocab[t])
                else:
                    raise ValueError(f"Unknown token '{t}' in token sequence: {expr_like}")
            return tokenizer.decode_infix(ids)
        raise TypeError("Unsupported token input type; expected list[int] or str")

    expr_a = _coerce(a_tokens)
    expr_b = _coerce(b_tokens)

    # Determine the maximum particle / photon index referenced so we do not
    # generate insufficient kinematics (which would otherwise cause KeyError).
    import re as _re
    used_indices = [int(m.group(1)) for m in _re.finditer(r'[pPeEfF]_(\d+)', expr_a + " " + expr_b)]
    max_used = max(used_indices) if used_indices else N
    N_eff = max(N, max_used)
    if N_eff != N:
        # We silently upgrade N; record in details for transparency.
        # (Could raise ValueError instead; auto-upgrade is more convenient for ad-hoc tests.)
        pass

    details = {
        "expr_a": expr_a,
        "expr_b": expr_b,
        "N_requested": N,
        "N_effective": N_eff,
        "samples": [],  # list of dicts with values & diffs
        "tol_abs": tol_abs,
        "tol_rel": tol_rel,
    }

    ok = True
    for i in range(samples):
        sample_seed = None if seed is None else seed + i
        mom, pol = _km.generate_kinematics(N_eff, M=M, seed=sample_seed)
        val_a = _gd.eval_infix_numeric(expr_a, mom, pol)
        val_b = _gd.eval_infix_numeric(expr_b, mom, pol)
        diff = abs(val_a - val_b)
        scale = max(abs(val_a), abs(val_b), 1.0)
        rel = diff / scale
        passed = (diff <= tol_abs) or (rel <= tol_rel)
        if not passed:
            ok = False
        details["samples"].append({
            "index": i,
            "value_a": val_a,
            "value_b": val_b,
            "abs_diff": diff,
            "rel_diff": rel,
            "passed": passed,
            "seed": sample_seed,
        })
        if not ok:
            # Early exit if a failure is detected to save time
            break

    return (ok, details) if return_details else ok


if __name__ == "__main__":
    tests =   [
        "13p_2 · p_2",
        "(4p_1 · p_2) ^ 3:",
        "Tr(7F_1 · F_2) + 5p_3 · e_2",
        "-Tr((F_1 · F_1) ^ 2:) / (3p_1 · p_1)",
        "(9p_4 · p_4 - 8p_3 · p_3) / (2p_2 · p_2)",
        "64Tr(F_3 · F_3 · F_3) - 12p_6 · p_6",
        "(Tr(F_2 · F_3) + 4p_1 · e_1) ^ 2:",
        "-6(p_1 · p_2) / (e_1 · e_2)",
        "(Tr(F_1 · F_2) ^ 2:) / (5p_1 · p_1) + 7",
        "12(p_2 · p_3 - p_3 · p_4) * Tr(3F_1 · F_1)",
        "M_1*M_2*p_1 · p_2 + 3e_1 · e_2 - 4F_1 · F_2"
    ]


    tok = ScatteringAmplitudeTokenizer(max_particles=8)

    for expr in tests:
        vec   = tok.encode_infix(expr)
        pref  = tok.decode_prefix(vec)
        back  = tok.decode_infix(vec)

        print(f"Input:              {expr}")
        print(f"Vector:             {vec}")
        print(f"Polish:             {pref}")
        print(f"Decoded Vector:     {back}")
        
    # Uncomment the following lines to test a more complex expression with debug info
    """
    vec1 = tok.encode_infix("-(e_3 · p_2*e_4 · p_3*p_1 · p_3*p_1 · p_4) + e_3 · p_1*e_4 · p_3*p_1 · p_4*p_2 · p_3 - (p_1 · p_1*e_3 · p_2*e_4 · p_1*p_1 · p_3*p_3 · p_4)/(p_1 · p_4) - (e_3 · p_2*e_4 · p_1*(p_1 · p_3)^2*p_3 · p_4)/(p_1 · p_4) - e_3 · p_1*e_4 · p_1*p_2 · p_3*p_3 · p_4 + (e_3 · p_2*e_4 · p_1*p_1 · p_2*p_1 · p_3*p_2 · p_4*p_3 · p_4)/(p_1 · p_4)^2 + (e_3 · p_2*e_4 · p_1*p_1 · p_2*p_1 · p_3*(p_3 · p_4)^2)/(p_1 · p_4)^2") 
    print(f"Vector: {vec1}")   
    print(f"Extra test: {tok.decode_infix(vec1)}")

    # Debug a problematic expression
    problematic_expr = "M^2*p_1 · p_2 + 3e_1 · e_2 - 4F_1 · F_2"
    debug_info = tok.debug_tokenization(problematic_expr)
    
    print("=== DEBUG INFO ===")
    for key, value in debug_info.items():
        print(f"{key}: {value}")
    
    if debug_info["unknown_tokens"]:
        print(f"ERROR: Unknown tokens found: {debug_info['unknown_tokens']}")
    """
