"""
Tokenizer.py — Infix ↔ prefix tokeniser for scattering-amplitude expressions.

Vocabulary covers: p_i, e_i, F_i, M_i  (i = 1…max_particles),
arithmetic ops +−*/^, dot ·, Tr, unary minus, parentheses,
single-digit constants 0–9 (as colon-suffixed tokens), and special tokens
<PAD>, <UNK>, <BOS>, <EOS>.

The internal representation is *prefix* (Polish) notation so that the
token stream is a linearised tree — convenient for seq2seq training.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class ScatteringAmplitudeTokenizer:

    _token_re = re.compile(
        r"p_\d+|e_\d+|F_\d+|M_\d+"
        r"|Tr"
        r"|M"
        r"|\d+"
        r"|[+\-*/^·().]"
    )

    # ── Construction ─────────────────────────────────────────────────
    def __init__(self, *, max_particles: int = 8, max_sequence_length: int = 2048):
        self.max_particles = max_particles
        self.max_sequence_length = max_sequence_length

        self.vocab_init: Dict[str, int] = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
            "+": 4, "-": 5, "*": 6, "/": 7, "^": 8,
            "(": 9, ")": 10,
            "0:": 11, "1:": 12, "2:": 13, "3:": 14, "4:": 15,
            "5:": 16, "6:": 17, "7:": 18, "8:": 19, "9:": 20,
            "·": 21, "Tr": 22, "u-": 23, "M": 24,
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
            **self.vocab_init,
            **self.p_tokens, **self.e_tokens,
            **self.f_tokens, **self.m_tokens,
        }
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        log.debug("Tokenizer vocabulary size: %d", len(self.vocab))

        self._prec = {"+": 1, "-": 1, "*": 2, "/": 2, "·": 2, "^": 3, "u-": 4, "Tr": 5}
        self._arity = {op: 2 for op in ["+", "-", "*", "/", "·", "^"]}
        self._arity["Tr"] = 1
        self._arity["u-"] = 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ── Public API ───────────────────────────────────────────────────
    def encode_infix(self, expr: str) -> List[int]:
        """Infix string → list of integer token IDs (prefix order)."""
        tokens = self._to_prefix(expr)
        result = []
        unknown = []
        for tok in tokens:
            if tok in self.vocab:
                result.append(self.vocab[tok])
            else:
                result.append(self.vocab["<UNK>"])
                unknown.append(tok)
        if unknown:
            raise ValueError(f"Unknown tokens in '{expr}': {unknown}")
        if self.max_sequence_length is not None and len(result) > self.max_sequence_length:
            raise ValueError(
                f"Tokenized expression has length {len(result)}, "
                f"exceeding max_sequence_length={self.max_sequence_length}"
            )
        return result

    def decode_prefix(self, ids: List[int]) -> str:
        """Token IDs → space-separated prefix string (skipping <PAD>)."""
        return " ".join(
            self.id_to_token[i] for i in ids if i != self.vocab["<PAD>"]
        )

    def decode_infix(self, ids: List[int]) -> str:
        """Token IDs → infix string with minimal parentheses."""
        skip = {self.vocab["<PAD>"], self.vocab["<BOS>"], self.vocab["<EOS>"]}
        toks = [self.id_to_token[i] for i in ids if i not in skip]

        def _parse(idx: int, parent_prec: int = 0, is_right: bool = False) -> Tuple[str, int, int]:
            if idx >= len(toks):
                raise ValueError("Malformed prefix: ran out of tokens.")
            tok = toks[idx]

            # Trace (unary)
            if tok == "Tr":
                child, nxt, cprec = _parse(idx + 1, self._prec[tok])
                inner = child[1:-1] if child.startswith("(") and child.endswith(")") else child
                return f"Tr({inner})", nxt, self._prec[tok]

            # Unary minus
            if tok == "u-":
                child, nxt, cprec = _parse(idx + 1, self._prec[tok])
                if cprec < self._prec[tok]:
                    return f"-({child})", nxt, self._prec[tok]
                return f"-{child}", nxt, self._prec[tok]

            # Binary operators
            if tok in self._arity and self._arity[tok] == 2:
                l_str, nxt, l_prec = _parse(idx + 1, self._prec[tok], False)
                r_str, nxt, r_prec = _parse(nxt, self._prec[tok], True)
                cur_prec = self._prec[tok]

                l_needs = l_prec < cur_prec
                r_needs = r_prec < cur_prec
                if tok == "/":
                    l_needs = True
                    if r_prec <= cur_prec:
                        r_needs = True
                elif tok == "-":
                    if r_prec <= cur_prec:
                        r_needs = True
                elif tok == "^":
                    if r_prec <= cur_prec:
                        r_needs = True

                if l_needs:
                    l_str = f"({l_str})"
                if r_needs:
                    r_str = f"({r_str})"

                needs_parens = (
                    cur_prec < parent_prec
                    or (cur_prec == parent_prec and is_right and tok in ["-", "/", "^"])
                )
                if tok in ["+", "-"]:
                    expr_str = f"{l_str} {tok} {r_str}"
                else:
                    expr_str = f"{l_str}{tok}{r_str}"
                return (f"({expr_str})" if needs_parens else expr_str), nxt, cur_prec

            # Multi-digit constant
            if tok.endswith(":") and tok[:-1].isdigit():
                digits = [tok[:-1]]
                j = idx + 1
                while j < len(toks) and toks[j].endswith(":") and toks[j][:-1].isdigit():
                    digits.append(toks[j][:-1])
                    j += 1
                return "".join(digits), j, 1000

            # Leaf
            return tok, idx + 1, 1000

        expr_out, final_idx, _ = _parse(0)
        if final_idx != len(toks):
            raise ValueError(f"Prefix stream not fully consumed: stopped at {final_idx}")
        return _strip_matched_outer_parens(expr_out)

    # ── Debug ────────────────────────────────────────────────────────
    def debug_tokenization(self, expr: str) -> Dict[str, Any]:
        raw = self._tokenise(expr)
        with_mul = self._insert_implicit_mul(raw)
        with_uminus = self._detect_unary_minus(with_mul)
        prefix = self._to_prefix(expr)
        unknown = [t for t in prefix if t not in self.vocab]
        return {
            "original": expr,
            "raw_tokens": raw,
            "with_implicit_mul": with_mul,
            "with_unary_minus": with_uminus,
            "prefix_tokens": prefix,
            "unknown_tokens": unknown,
        }

    # ── Internals: infix → prefix ────────────────────────────────────
    def _tokenise(self, expr: str) -> List[str]:
        raw = self._token_re.findall(expr.replace(" ", ""))
        out: List[str] = []
        for t in raw:
            if t == ".":
                out.append("·")
            elif t.isdigit():
                out.extend(f"{d}:" for d in t)
            else:
                out.append(t)
        return out

    @staticmethod
    def _is_digit(tok: str) -> bool:
        return tok.endswith(":") and tok[:-1].isdigit()

    def _insert_implicit_mul(self, toks: List[str]) -> List[str]:
        res: List[str] = []
        for i, t in enumerate(toks):
            if i:
                left, right = toks[i - 1], t
                left_val = left not in self._prec and left != "("
                right_val = right not in self._prec and right != ")"
                if (left_val and (right_val or right in ("(", "Tr"))
                        and not (self._is_digit(left) and self._is_digit(right))):
                    res.append("*")
            res.append(t)
        return res

    def _detect_unary_minus(self, tokens: List[str]) -> List[str]:
        res: List[str] = []
        for i, t in enumerate(tokens):
            if t == "-" and (i == 0 or tokens[i - 1] in self._prec or tokens[i - 1] == "("):
                res.append("u-")
            else:
                res.append(t)
        return res

    def _to_prefix(self, expr: str) -> List[str]:
        infix = self._insert_implicit_mul(self._tokenise(expr))
        infix = self._detect_unary_minus(infix)
        # Reverse + swap parens, then shunting-yard → postfix of reversed = prefix
        infix = [")" if t == "(" else "(" if t == ")" else t for t in infix[::-1]]
        out: List[str] = []
        stack: List[str] = []
        for tok in infix:
            if tok == "(":
                stack.append(tok)
            elif tok == ")":
                while stack and stack[-1] != "(":
                    out.append(stack.pop())
                if stack:
                    stack.pop()
            elif tok in self._prec:
                while (stack and stack[-1] != "("
                       and stack[-1] in self._prec
                       and self._prec[stack[-1]] > self._prec[tok]):
                    out.append(stack.pop())
                stack.append(tok)
            else:
                out.append(tok)
        out.extend(reversed(stack))
        return out[::-1]


# ── Utilities ────────────────────────────────────────────────────────

def _strip_matched_outer_parens(s: str) -> str:
    """Remove outer parentheses only if they genuinely wrap the whole expression.

    '(a+b)*(c+d)' is left alone — the opening '(' closes before the end.
    '((a+b))' → '(a+b)' → 'a+b' (applied once per call).
    """
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return s  # the opening paren closes before the final character
    return s[1:-1]


# ── Numerical equivalence helper ─────────────────────────────────────

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
    pol_modes: Tuple[str, ...] = ("coulomb", "covariant"),
    return_details: bool = False,
):
    """Compare two expressions (token IDs or infix strings) numerically.

    Returns True if all sampled phase-space evaluations agree within tolerances.
    """
    import importlib
    import os
    import sys

    def _lazy(mod_name: str):
        try:
            return importlib.import_module(mod_name)
        except ModuleNotFoundError:
            here = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(here, f"{mod_name}.py")
            if not os.path.isfile(candidate):
                raise
            spec = importlib.util.spec_from_file_location(mod_name, candidate)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                return mod
            raise

    _gd = _lazy("gen_data")
    _km = _lazy("kinematics")

    def _coerce(expr_like):
        if isinstance(expr_like, (list, tuple)) and all(isinstance(x, int) for x in expr_like):
            return tokenizer.decode_infix(list(expr_like))
        if isinstance(expr_like, str):
            s = expr_like.strip()
            if any(sym in s for sym in ["p_", "e_", "F_", "Tr", "·", "+", "-", "*", "/", "^"]):
                return s
            toks = s.split()
            ids = []
            for t in toks:
                if t in tokenizer.vocab:
                    ids.append(tokenizer.vocab[t])
                else:
                    raise ValueError(f"Unknown token '{t}'")
            return tokenizer.decode_infix(ids)
        raise TypeError("Expected list[int] or str")

    expr_a = _coerce(a_tokens)
    expr_b = _coerce(b_tokens)

    used = [int(m.group(1)) for m in re.finditer(r"[pPeEfF]_(\d+)", expr_a + " " + expr_b)]
    N_eff = max(N, max(used) if used else N)

    details: Dict[str, Any] = {
        "expr_a": expr_a, "expr_b": expr_b,
        "N_effective": N_eff, "samples": [],
    }
    ok = True
    for mode_idx, pol_mode in enumerate(pol_modes):
        for i in range(samples):
            s = None if seed is None else seed + mode_idx * samples + i
            mom, pol = _km.generate_kinematics(N_eff, M=M, pol_mode=pol_mode, seed=s)
            va = _gd.eval_infix_numeric(expr_a, mom, pol)
            vb = _gd.eval_infix_numeric(expr_b, mom, pol)
            diff = abs(va - vb)
            passed = _gd.numeric_values_close(
                va,
                vb,
                tol_abs=tol_abs,
                tol_rel=tol_rel,
            )
            details["samples"].append(
                {"pol_mode": pol_mode, "va": va, "vb": vb, "diff": diff, "passed": passed}
            )
            if not passed:
                ok = False
                break
        if not ok:
            break

    return (ok, details) if return_details else ok


# ── Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    tests = [
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
        "M_1*M_2*p_1 · p_2 + 3e_1 · e_2 - 4F_1 · F_2",
    ]

    tok = ScatteringAmplitudeTokenizer(max_particles=8)
    for expr in tests:
        vec = tok.encode_infix(expr)
        pref = tok.decode_prefix(vec)
        back = tok.decode_infix(vec)
        print(f"Input:   {expr}")
        print(f"IDs:     {vec}")
        print(f"Prefix:  {pref}")
        print(f"Decoded: {back}")
        print()
