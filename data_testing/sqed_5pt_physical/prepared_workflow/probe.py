"""Bounded preparation experiment with the unchanged SQED five-point model."""
from pathlib import Path
import csv
import json
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_and_validate import ROOT, MODEL, symbolic, on_shell, expressions, gd
from data_testing import evaluate_model as ev


def prepare():
    cases=[]
    for k in (2,3,4):
        i,j=[x for x in (2,3,4) if x!=k]
        target=f'(p_1 · F_{i} · F_{j} · p_1)*(p_1 · F_{k} · p_5)/((p_1 · p_2)*(p_1 · p_3)*(p_1 · p_4)*(p_{k} · p_5))'
        expanded=gd.simplify_to_lowest_terms(gd.full_expand_expression(target))
        assert on_shell(symbolic(target)-symbolic(expanded))==0
        cases.append(dict(name=f'component_{k}',simple=target,scrambled=expanded))
    full=expressions()
    for name,expr in [('whole_projected',full['simple']),('whole_projected_unit',f'({full["simple"]})/(-2)')]:
        expanded=gd.simplify_to_lowest_terms(gd.full_expand_expression(expr))
        assert on_shell(symbolic(expr)-symbolic(expanded))==0
        cases.append(dict(name=name,simple=expr,scrambled=expanded))
    return cases


def main():
    torch.set_num_threads(4)
    model=ev.load_transformer_model(ev.TransformerRegressor,str(MODEL),device='cpu')['model'];model.eval()
    tok=ev.ScatteringAmplitudeTokenizer(max_particles=8,max_sequence_length=None)
    cases=prepare()
    (HERE/'cases.json').write_text(json.dumps(cases,indent=2))
    results=[]
    for case in cases:
        ids=tok.encode_infix(case['scrambled'])
        assert on_shell(symbolic(tok.decode_infix(ids))-symbolic(case['simple']))==0
        src=torch.tensor([[2]+ids+[3]])
        for mode in ('greedy','beam'):
            t=time.monotonic()
            with torch.inference_mode():
                output,beams=ev.decode_with_model(model,src,max_length=256,decoding_method=mode,beam_size=20)
            seqs=[output[0].tolist()] if beams is None else beams[0]
            records=[]
            for raw in seqs:
                content=ev.strip_special_tokens(raw)
                ok,pred,error=ev.safe_decode_infix(tok,content)
                exact=False
                if ok:
                    try: exact=on_shell(symbolic(pred)-symbolic(case['simple']))==0
                    except Exception as exc: error=str(exc)
                records.append(dict(prediction=pred,exact=bool(exact),error=error,raw_ids=raw,tokens=len(content)))
            rec=dict(name=case['name'],mode=mode,input_tokens=len(ids),seconds=time.monotonic()-t,candidates=records)
            results.append(rec)
            (HERE/'probe_results.json').write_text(json.dumps(results,indent=2))
            print(case['name'],mode,len(ids),[(r['tokens'],r['exact']) for r in records],flush=True)
            if any(r['exact'] for r in records): break


if __name__=='__main__':main()
