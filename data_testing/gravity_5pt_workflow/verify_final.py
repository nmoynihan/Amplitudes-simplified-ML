"""Reconstruct and verify 67/53/99-token forms from fresh actual predictions."""
from pathlib import Path
import csv
import json

import sympy as sp
from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from .symbolic_utils import symbolic, parse, serialize
from .common import read_source, require, numeric_check


def main(source_path: Path, output_dir: Path):
    tok=ScatteringAmplitudeTokenizer(max_particles=8,max_sequence_length=None)
    source=read_source(source_path)
    original=symbolic(source)
    selected=json.loads((output_dir/"selected_groups.json").read_text())
    require(len(selected)==2 and all(r["exact_equal"] for r in selected), 'Verification failed: len(selected)==2 and all(r["exact_equal"] for r in selected)')
    for row in selected:
        require(row["eos"] and sp.cancel(symbolic(row["prediction"])-symbolic(row["input"]))==0,
                "A saved model prediction is not an exact EOS-terminated scalar-core identity.")
    raw=(output_dir/"two_call_reconstruction.txt").read_text().strip()
    raw_sym=symbolic(raw)
    weighted=" + ".join(f"({row['weight']})*({row['prediction']})" for row in selected)
    require(sp.cancel(raw_sym-symbolic(weighted))==0,
            "Saved reconstruction differs from the current weighted model predictions.")
    zeros={sp.Symbol(s):0 for s in ["e4p1","e5p1","e4e5"]}
    # Reuse the same ACTUAL two model outputs with weights derived from the
    # unrestricted CSV identity. The trace block is kept entirely analytic.
    dot=lambda a,b:f"(p_{a}·p_{b})"
    h,k,u,v,t,q=[dot(a,b) for a,b in [(1,4),(1,5),(2,4),(3,5),(4,5),(2,3)]]
    D="("+"*".join(dot(a,b) for a,b in [(1,2),(1,3),(1,4),(1,5),(2,4),(3,5)])+")"
    trace="Tr(F_4·F_5)"
    trace_part=f"({trace})^2*({q}*{t}-3*{h}*{k})/(24*{q}*{t}^3)"
    generic=(f"({trace_part})-({D}/({u}*{v}*{t}^4))*({selected[0]['prediction']})"
             f"+({D}*({t}-{q})/(2*{h}*{k}*{q}*{t}^4))*({selected[1]['prediction']})")
    require(sp.cancel(symbolic(generic)-original)==0, 'Verification failed: sp.cancel(symbolic(generic)-original)==0')
    # Compact F monomials below are extracted from the actual model predictions.
    base_products=[]
    for row in selected:
        numerator=sp.fraction(sp.cancel(parse(row["prediction"])))[0]
        powers=numerator.as_powers_dict()
        require(all(power==2 for power in powers.values()), 'Verification failed: all(power==2 for power in powers.values())')
        base_products.append(serialize(sp.prod(powers.keys())))
    Y,X=base_products
    # Cancel the actual second prediction against its external weight, treating
    # F contractions as atoms. Extract square momentum factors from its scalar
    # denominator to put them inside the squared numerator product.
    model_num_x=sp.fraction(sp.cancel(parse(selected[1]["prediction"])))[0]
    coefficient_x=sp.factor(parse(selected[1]["weight"])*parse(selected[1]["prediction"])/model_num_x)
    n,d=sp.fraction(sp.factor(coefficient_x*sp.Symbol("p4p5")**4))
    square_factor=sp.Integer(1)
    for base,exponent in d.as_powers_dict().items():
        if isinstance(base,sp.Symbol) and exponent.is_Integer and exponent>0:
            square_factor*=base**(int(exponent)//2)
    remaining_denominator=sp.cancel(d/square_factor**2)
    # Keep pure power terms first to avoid ambiguous adjacent digit leaves in
    # this legacy prefix tokenizer. Every final serialization is checked below.
    terms=sorted(sp.Add.make_args(n),key=lambda term:(not isinstance(term,sp.Pow),str(term)))
    numerator_text=""
    for term in terms:
        negative=term.could_extract_minus_sign()
        numerator_text+=("-" if negative else "+")+serialize(-term if negative else term)
    numerator_text=numerator_text.lstrip("+")
    square_text=serialize(square_factor)
    coeff_text=serialize(remaining_denominator)
    grouped=(f"((({X})/({square_text}))^2*({numerator_text})/({coeff_text})"
             f"-({Y})^2/({u}*{v}))/({t}^4)")
    trace_grouped=(f"(({trace})^2*({numerator_text})/({serialize(4*remaining_denominator)})"
                   f"-({Y})^2/({u}*{v}))/({t}^4)")
    expressions={"model_plus_cleanup_67":grouped,
                 "model_plus_helicity_cleanup_53":trace_grouped}
    generic_short=(f"(({trace})^2*{t}*({q}*{t}-3*{h}*{k})/(24*{q})"
                   f"+({X})^2*({t}-{q})/(2*{h}*{k}*{q})"
                   f"-({Y})^2/({u}*{v}))/({t}^4)")
    require(sp.cancel(symbolic(generic_short)-original)==0, 'Verification failed: sp.cancel(symbolic(generic_short)-original)==0')
    expressions["model_plus_cleanup_general"]=generic_short
    records={}
    for name,text in expressions.items():
        ids=tok.encode_infix(text)
        expanded=symbolic(text)
        roundtrip=sp.cancel(expanded-symbolic(tok.decode_infix(ids)))
        common=sp.cancel((expanded-original).subs(zeros))
        require(roundtrip==common==0, 'Verification failed: roundtrip==common==0')
        delta_model=sp.cancel(expanded-raw_sym)
        delta_original=sp.cancel(expanded-original)
        if name=="model_plus_cleanup_67":require(delta_model==0, 'Verification failed: delta_model==0')
        if name=="model_plus_cleanup_general":require(delta_original==0, 'Verification failed: delta_original==0')
        equal,error=numeric_check(source,text)
        require(equal, 'Verification failed: equal')
        (output_dir/(name+".txt")).write_text(text+"\n")
        with (output_dir/(name+".csv")).open("w") as handle:
            csv.writer(handle).writerow([name,text])
        records[name]={"tokens":len(ids),"roundtrip_exact_residual":str(roundtrip),
                       "generic_equal_to_two_call_output":delta_model==0,
                       "generic_equal_to_original":delta_original==0,
                       "exact_common_reference_residual":str(common),
                       "numeric_checks":80,"numeric_equal":equal,"maximum_relative_error":error,
                       "scope":"independent dot-product rational identity" if delta_original==0 else "positive-positive helicity",
                       "origin":"actual neural predictions plus explicitly symbolic algebra"}
    summary={"original_tokens":len(tok.encode_infix(source)),"historical_nine_call_tokens":789,
             "two_call_tokens_before_cleanup":len(tok.encode_infix(raw)),"results":records,
             "historical_exploratory_neural_calls":4,
             "selected_pipeline_neural_calls":2,
             "historical_context":"The original investigation had two failed scalar serializations before the two selected successes; those attempts are not rerun here.",
             "claim":"Grouping and all final cleanup are analytic; model generates the compact F expressions for the two scalar cores.",
             "final_expressions_generated_from_fresh_two_model_outputs":True,
             "earlier_compact_answer_loaded":False,
             "no_retraining":True}
    (output_dir/"final_verification.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

    return summary
