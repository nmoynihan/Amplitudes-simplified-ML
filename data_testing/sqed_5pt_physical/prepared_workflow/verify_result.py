"""Check the actual saved evaluator prediction against the original amplitude."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_and_validate import (ROOT, MODEL, symbolic, on_shell, gd,
                                generate_kinematics, direct_diagrams,
                                ScatteringAmplitudeTokenizer)


def read(path):
    with path.open() as f:return next(csv.DictReader(f))


def main():
    original=read(HERE.parent/'sqed_5pt_physical.csv')['scrambled']
    prepared=read(HERE/'sqed_5pt_prepared.csv')['scrambled']
    results=ROOT/'data_testing/outputs/sqed_5pt_prepared_beam20_beam_results.csv'
    row=read(results)
    prediction=row['top1_prediction_expr']
    assert row['top1_completed']=='1' and row['top1_syntax_valid']=='1'
    assert on_shell(symbolic(prediction)-symbolic(original))==0
    assert on_shell(symbolic(prediction)-symbolic(prepared))==0
    tok=ScatteringAmplitudeTokenizer(max_particles=8,max_sequence_length=5000)
    assert on_shell(symbolic(tok.decode_infix(tok.encode_infix(prediction)))-symbolic(original))==0
    worst=0.; count=0
    for mass in (0.7,2.,5.):
        for seed in range(20):
            p,e=generate_kinematics(5,M=mass,seed=23000+seed)
            rng=np.random.default_rng(41000+seed)
            # Arbitrary directions in each transverse polarization plane.
            for k in range(3):
                direction=p[k+1,1:]/np.linalg.norm(p[k+1,1:])
                other=np.cross(direction,e[k,1:])
                angle=rng.uniform(0,2*np.pi)
                e[k,1:]=np.cos(angle)*e[k,1:]+np.sin(angle)*other
            for shifted in (False,True):
                ee=e+rng.uniform(-0.5,0.5,size=(3,1))*p[1:4] if shifted else e
                values=[gd.eval_infix_numeric(expr,p,ee,strict=True) for expr in (original,prepared,prediction)]
                values.append(direct_diagrams(p,ee,mass))
                assert all(np.isfinite(values))
                err=max(abs(v-values[0])/max(1.,abs(v),abs(values[0])) for v in values)
                assert err<1e-9,err
                worst=max(worst,err);count+=1
    with MODEL.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(checkpoint=str(MODEL),checkpoint_sha256=digest,
                source_results=str(results),exact_prediction_minus_original=0,
                exact_prediction_minus_prepared=0,exact_tokenizer_round_trip=True,
                masses=[0.7,2.,5.],momentum_points=60,polarization_and_gauge_checks=count,
                max_normalized_numerical_error=worst,
                token_counts={name:len(tok.encode_infix(expr)) for name,expr in
                              [('original',original),('prepared',prepared),('prediction',prediction)]},
                selection=row['selection_reason'],
                original_top1_correct=row['original_top1_num_eq_scrambled'],
                retained_candidates=row['candidate_sequences_generated'],
                equivalent_candidates=row['candidate_num_eq_scrambled_count'],
                prediction=prediction)
    (HERE/'verified_result.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'simplified_amplitude.txt').write_text(prediction+'\n')
    with (HERE/'simplified_amplitude.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','expression']);writer.writerow(['sqed_5pt',prediction])
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
