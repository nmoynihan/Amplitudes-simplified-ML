"""Replay the two successful scalar serializations with real greedy inference.

The scalar orientation is explicitly selected from the original investigation.
This module does not search for a decomposition or load a compact target.
"""
from pathlib import Path
import csv
import json
import time

import sympy as sp
from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from data_gen.data_gen_gravity.core import expand_expression
from .common import numeric_check, read_source, require, sha256, write_json
from .symbolic_utils import safe, symbolic

SCALAR_PAIRS = (
    ("((p_2 · p_4)*(e_4 · p_5)-(p_2 · e_4)*(p_4 · p_5))",
     "((p_3 · p_5)*(e_5 · p_4)-(p_3 · e_5)*(p_5 · p_4))"),
    ("((p_1 · p_4)*(e_4 · p_5)-(p_1 · e_4)*(p_4 · p_5))",
     "((p_1 · p_5)*(e_5 · p_4)-(p_1 · e_5)*(p_5 · p_4))"),
)
DENOMINATOR = "((p_1 · p_2)*(p_1 · p_3)*(p_1 · p_4)*(p_1 · p_5)*(p_2 · p_4)*(p_3 · p_5))"


def prepare_inputs(output_dir: Path):
    """Check the selected scalar inputs against newly derived groups."""
    tok = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
    previous = json.loads((output_dir / "grouped_inputs.json").read_text())
    require(len(previous) == 2, "Expected exactly two coefficient-derived groups.")
    choices = []
    for index, ((a, b), prior) in enumerate(zip(SCALAR_PAIRS, previous), 1):
        scalar = f"({a}*{a}*{b}*{b})/{DENOMINATOR}"
        text = safe(expand_expression(scalar, full=True))
        ids = tok.encode_infix(text)
        exact_input = symbolic(prior["input"])
        require("F_" not in text and 1 not in ids, "Invalid scalar-only model input.")
        require(sp.cancel(symbolic(text) - exact_input) == 0,
                "Source is outside this seed-specific workflow: scalar core differs.")
        require(sp.cancel(symbolic(tok.decode_infix(ids)) - exact_input) == 0,
                "Model input changed in the tokenizer roundtrip.")
        choices.append({"name": f"group_{index}_variant_1", "group": index,
                        "input": text, "input_tokens": len(ids), "ids": ids,
                        "weight": prior["weight"]})
    write_json(output_dir / "prepared_model_inputs.json", choices)
    with (output_dir / "model_inputs.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows((r["name"], r["input"]) for r in choices)
    return choices


def main(source_path: Path, checkpoint_path: Path, output_dir: Path,
         *, threads: int = 4, max_length: int = 256):
    import torch
    from data_testing import evaluate_model as ev

    choices = prepare_inputs(output_dir)
    tok = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
    torch.set_num_threads(threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ev.TransformerRegressor(**dict(checkpoint["model_args"], device="cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    source_capacity = ev.positional_encoding_capacity(model, "src_pos_encoding")
    target_capacity = ev.positional_encoding_capacity(model, "tgt_pos_encoding")
    require(2 <= max_length <= target_capacity,
            f"max-length must be between 2 and the checkpoint capacity {target_capacity}.")
    for row in choices:
        require(len(row["ids"]) + 2 <= source_capacity,
                f"{row['name']} exceeds the checkpoint source capacity {source_capacity}.")
        require(all(0 <= token < model.vocab_size for token in [2, *row["ids"], 3]),
                "Input tokens are outside the checkpoint vocabulary.")

    results, selected = [], []
    for choice in choices:
        started = time.monotonic()
        with torch.inference_mode():
            outputs, _ = ev.decode_with_model(
                model, torch.tensor([[2, *choice["ids"], 3]]),
                max_length=max_length, decoding_method="greedy")
        raw = outputs[0].tolist()
        content = ev.strip_special_tokens(raw)
        ok, prediction, error = ev.safe_decode_infix(tok, content)
        try:
            exact = bool(ok and sp.cancel(symbolic(prediction) - symbolic(choice["input"])) == 0)
        except Exception as exc:
            exact, error = False, f"{type(exc).__name__}: {exc}"
        result = {"name": choice["name"], "group": choice["group"], "mode": "greedy",
                  "input_tokens": choice["input_tokens"], "output_tokens": len(content),
                  "prediction": prediction, "parsed": ok, "error": error,
                  "exact_equal": exact, "eos": 3 in raw, "raw_ids": raw,
                  "weight": choice["weight"], "seconds": time.monotonic() - started}
        results.append(result)
        write_json(output_dir / "serialization_results.json", results)
        print(json.dumps({k: result[k] for k in
                          ("name", "input_tokens", "output_tokens", "exact_equal", "eos")}), flush=True)
        require(exact and result["eos"] and 1 not in content,
                f"{choice['name']}: model output was not an exact EOS-terminated prediction; "
                "see serialization_results.json. No full amplitude was accepted.")
        selected.append(dict(result, input=choice["input"]))

    write_json(output_dir / "selected_groups.json", selected)
    reconstructed = " + ".join(f"({r['weight']})*({r['prediction']})" for r in selected)
    original = read_source(source_path)
    zeros = {sp.Symbol(s): 0 for s in ("e4p1", "e5p1", "e4e5")}
    require(sp.cancel((symbolic(original) - symbolic(reconstructed)).subs(zeros)) == 0,
            "Weighted reconstruction does not equal the source in the common reference.")
    equal, error = numeric_check(original, reconstructed)
    require(equal, "Weighted reconstruction failed the 80 same-helicity numerical checks.")
    require(sp.cancel(symbolic(reconstructed) - symbolic(tok.decode_infix(tok.encode_infix(reconstructed)))) == 0,
            "Weighted reconstruction changed in the tokenizer roundtrip.")
    (output_dir / "two_call_reconstruction.txt").write_text(reconstructed + "\n", encoding="utf-8")
    with (output_dir / "successful_model_inputs.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows((r["name"], r["input"]) for r in selected)
    summary = {"neural_groups": 2, "greedy_search_calls": len(results),
               "calls_to_reproduce_selected_pipeline": len(results),
               "checkpoint_sha256": sha256(checkpoint_path), "source_sha256": sha256(source_path),
               "exact_component_equalities": True, "scope": "positive-positive helicity",
               "whole_exact_common_reference_residual": "0", "numeric_checks": 80,
               "numeric_equal": equal, "maximum_relative_error": error,
               "full_tokens_before_cleanup": len(tok.encode_infix(reconstructed)),
               "selected": [{k: r[k] for k in ("name", "input_tokens", "output_tokens", "prediction")}
                            for r in selected]}
    write_json(output_dir / "two_call_summary.json", summary)
    return summary
