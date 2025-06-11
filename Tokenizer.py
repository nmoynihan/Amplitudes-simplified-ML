import re
from typing import List, Dict, Tuple


class ScatteringAmplitudeTokenizer:
    # Some sexy regexes to match tokens in the input expression
    # p_1, e_1, F_1 are the particle, polarisation and field strength tokens, respectively.
    # Tr is the trace operator.
    _token_re = re.compile(
        r'p_\d+|e_\d+|F_\d+'
        r'|Tr'
        r'|M' # mass, should probably not be used in expressions!
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

        self.vocab: Dict[str, int] = {
            **self.vocab_init, **self.p_tokens, **self.e_tokens, **self.f_tokens
        }
        self.id_to_token = {i: t for t, i in self.vocab.items()}

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
        toks = [self.id_to_token[i] for i in ids if i != self.vocab["<PAD>"]]

        def _parse(idx: int) -> Tuple[str, int]:
            if idx >= len(toks):
                raise ValueError("Malformed prefix tokens: ran out of tokens.")

            tok = toks[idx]
            # operators --------------------------------------------------- #
            if tok == "Tr":                     # unary trace operator
                child, nxt = _parse(idx + 1)
                return f"Tr({child})", nxt
            elif tok == 'u-':                   # unary minus
                child, nxt = _parse(idx + 1)
                return f"-({child})", nxt
            elif tok in self._arity:            # binary operators
                left, nxt = _parse(idx + 1)
                right, nxt = _parse(nxt)
                return f"({left} {tok} {right})", nxt

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

    def debug_tokenization(self, expr: str) -> Dict[str, any]:
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
        "12(p_2 · p_3 - p_3 · p_4) * Tr(3F_1 · F_1)"
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
