"""infer_amplitude.py — feed an amplitude into a trained model and show the predicted simplification.

Reads an amplitude (a `1, <amplitude>` CSV row, or a bare expression), feeds it as the *scrambled*
input to a trained encoder-decoder, decodes greedy + beam, and checks whether any prediction is
numerically equivalent to the input (using the gluon-correct data_gen_ym numerics, NOT the
scalar-QED Tokenizer.numerically_equivalent helper).

Run from the repo root, e.g.:
    venv/bin/python data_gen/data_gen_ym/infer_amplitude.py \
        --model models/ym_4pt_10k_smoke/best_model.pt \
        --csv data/data_ym/gluon4feyn.csv --N 4
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))      # .../data_gen/data_gen_ym
DATA_GEN = os.path.dirname(HERE)                         # .../data_gen
ROOT = os.path.dirname(DATA_GEN)                         # repo root
sys.path.insert(0, os.path.join(ROOT, "transformer"))   # transformer_functions
sys.path.insert(0, DATA_GEN)                             # Tokenizer + the data_gen_ym package

import torch
from transformer_functions import (
    TransformerRegressor, load_transformer_model, decode_with_model, clean_seq,
)
from Tokenizer import ScatteringAmplitudeTokenizer
from data_gen_ym.kinematics import generate_kinematics
from data_gen_ym.numerics import eval_infix_numeric

BOS, EOS, PAD = 2, 3, 0


def numerically_equivalent_gluon(expr_a, expr_b, N, seeds=(1, 2, 3, 4, 5),
                                 modes=("coulomb", "covariant"), tol=1e-6):
    """True if expr_a == expr_b at all sampled massless N-gluon kinematic points."""
    if not expr_b or not expr_b.strip():
        return False
    for s in seeds:
        for m in modes:
            mom, pol = generate_kinematics(N, 2.0, pol_mode=m, seed=s)
            try:
                va = eval_infix_numeric(expr_a, mom, pol)
                vb = eval_infix_numeric(expr_b, mom, pol)
            except Exception:
                return False
            if not (math.isfinite(va) and math.isfinite(vb)):
                return False
            if abs(va - vb) > tol * max(1.0, abs(va), abs(vb)):
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ym_4pt_10k_smoke/best_model.pt")
    ap.add_argument("--csv", default="data/data_ym/gluon4feyn.csv")
    ap.add_argument("--expr", default=None, help="Inline amplitude (overrides --csv).")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    if args.expr:
        amp = args.expr.strip()
    else:
        row = next(csv.reader(open(args.csv)))
        amp = (row[1] if len(row) > 1 else row[0]).strip()  # field after the leading "1,"

    tok = ScatteringAmplitudeTokenizer(max_particles=8)
    ids = tok.encode_infix(amp)
    src_ids = [BOS] + ids + [EOS]
    print(f"input amplitude: {len(amp)} chars, {len(ids)} tokens (src {len(src_ids)} w/ BOS/EOS)")

    loaded = load_transformer_model(TransformerRegressor, args.model, device="cpu")
    model = loaded["model"]
    model.to("cpu")
    model.device = "cpu"
    model.eval()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model: epoch {loaded['epoch']}, val_loss {loaded['val_loss']:.4f}, {nparams:,} params\n")

    src = torch.tensor([src_ids], dtype=torch.long)

    def to_text(seq):
        """Return (text, ntok, parseable). The model may emit an invalid prefix stream."""
        content = [t for t in clean_seq(seq) if t not in (BOS, EOS, PAD)]
        try:
            return tok.decode_infix(content), len(content), True
        except Exception:
            try:
                raw = tok.decode_prefix(content)
            except Exception:
                raw = str(content[:60])
            return f"<unparseable> {raw}", len(content), False

    candidates = []  # (label, text, ntok, parseable)
    with torch.no_grad():
        out, _ = decode_with_model(model, src, max_length=args.max_length, decoding_method="greedy")
        candidates.append(("greedy", *to_text(out[0].tolist())))

        _decoded, beams = decode_with_model(model, src, max_length=args.max_length,
                                            decoding_method="beam", beam_size=args.beam_size)
        for bi, seq in enumerate(beams[0]):
            candidates.append((f"beam{bi}", *to_text(seq)))

    print("=== predictions ===")
    best_equiv = None
    for label, s, n, ok in candidates:
        eq = numerically_equivalent_gluon(amp, s, args.N) if ok else False
        if eq and (best_equiv is None or n < best_equiv[2]):
            best_equiv = (label, s, n)
        disp = s if len(s) <= 220 else s[:220] + " …"
        print(f"[{label:7}] {'EQUIV ✓' if eq else 'not-equiv'}  ntok={n}\n    {disp or '(empty)'}\n")

    print("=== summary ===")
    print(f"input tokens: {len(ids)}")
    if best_equiv:
        rel = "shorter" if best_equiv[2] < len(ids) else "NOT shorter"
        print(f"shortest EQUIVALENT prediction: [{best_equiv[0]}] ntok={best_equiv[2]} ({rel} than input)")
        print("   ", best_equiv[1])
    else:
        print("no prediction was numerically equivalent to the input "
              "(expected for the smoke model on an out-of-distribution amplitude).")


if __name__ == "__main__":
    main()
