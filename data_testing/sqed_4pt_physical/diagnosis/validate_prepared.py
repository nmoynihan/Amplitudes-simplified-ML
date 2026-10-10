"""Validate successful prepared input/prediction against the physical diagrams."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import build_and_validate as physics
from data_gen import gen_data as gd
from data_gen.kinematics import generate_kinematics, mdot


def main():
    success = json.loads((HERE/'full_search_summary.json').read_text())['success']
    original = physics.expressions()
    cases = dict(reference=original['simple'],original=original['scrambled'],
                 prepared=success['input'],prediction=success['prediction'])
    reference = physics.symbolic(cases['reference'])
    for name,expr in cases.items():
        assert physics.on_shell(physics.symbolic(expr)-reference)==0, name
        assert all(physics.ward(physics.symbolic(expr),leg)==0 for leg in (2,3)), name
    maxima = dict(diagrams=0.,gauge_shift=0.,photon_exchange=0.)
    def check(kind,a,b):
        residual=abs(a-b)/max(1.,abs(a),abs(b))
        maxima[kind]=max(maxima[kind],residual)
        assert np.isfinite(a) and np.isfinite(b) and residual<1e-9
    count=0
    for mass in (0.7,2.,5.):
        for mode in ('coulomb','covariant'):
            for seed in range(20):
                p,e=generate_kinematics(4,M=mass,pol_mode='coulomb',seed=10100+seed)
                rng=np.random.default_rng(11100+seed)
                for k in range(2):
                    cross=np.cross(p[k+1,1:]/np.linalg.norm(p[k+1,1:]),e[k,1:])
                    angle=rng.uniform(0.,2*np.pi)
                    e[k]=np.cos(angle)*e[k]+np.sin(angle)*np.r_[0.,cross]
                if mode=='covariant': e+=rng.uniform(-0.5,0.5,(2,1))*p[1:3]
                exact=physics.direct_diagrams(p,e,mass)
                for name,expr in cases.items():
                    value=gd.eval_infix_numeric(expr,p,e,strict=True)
                    check('diagrams',value,exact)
                    check('gauge_shift',gd.eval_infix_numeric(expr,p,e+np.array([0.31,-0.19])[:,None]*p[1:3],strict=True),value)
                    check('photon_exchange',gd.eval_infix_numeric(expr,p[[0,2,1,3]],e[[1,0]],strict=True),value)
                count+=1
    row=dict(simple=original['simple'],scrambled=success['input'])
    csv_path=HERE.parent/'sqed_4pt_physical_prepared.csv'
    if csv_path.exists():
        with csv_path.open(newline='',encoding='utf-8') as handle:
            assert list(csv.DictReader(handle))==[row]
    (HERE/'prepared_amplitude.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    report=dict(exact_input_equality=True,exact_model_prediction_equality=True,
                exact_ward_identities=True,numerical_points=count,masses=[0.7,2.,5.],
                polarisations='random transverse rotations plus covariant gauge shifts',
                max_normalized_errors=maxima,
                preparation='p4=-p1-p2-p3 and e2.p2=e3.p3=0; distribute scalar terms',
                unchanged_model=True,model_input_contains_field_strengths=False,
                external_reconstruction_factors=False,input_tokens=success['input_tokens'],
                output_tokens=success['prediction_tokens'],model_prediction=success['prediction'])
    (HERE/'prepared_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
