"""Bounded, exact-verified search of normalized SQED scalar-expression inputs.

Only the expanded scalar-dot input goes to the unchanged checkpoint.
References are used after inference to verify candidates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import sympy as sp
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from data_testing import evaluate_model as ev
from data_gen import gen_data as gd

spec = importlib.util.spec_from_file_location('physical_validation', HERE.parents[1]/'build_and_validate.py')
physics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics)

MODEL = Path('/Users/kymani/Documents/SQED_KYMANI/SQED/4PT_Model/best_model.pt')
CORES = {
    'unit_core': dict(target='(p_1 · F_2 · F_3 · p_1)/((p_1 · p_2)*(p_1 · p_3))', external='-2'),
    'numerator_core': dict(target='p_1 · F_2 · F_3 · p_1', external='-2/((p_1 · p_2)*(p_1 · p_3))'),
    'trace_core': dict(target='(F_2 · F_3)', external=None),
    'cross_endpoint_core': dict(target='p_1 · F_2 · F_3 · p_4', external=None),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', nargs='+', default=['unit_core','numerator_core'])
    parser.add_argument('--variants', type=int, default=40)
    parser.add_argument('--beam-every', type=int, default=5)
    parser.add_argument('--beam-all', action='store_true')
    parser.add_argument('--output-prefix', default='')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tokenizer = ev.ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
    checkpoint = torch.load(MODEL, map_location='cpu', weights_only=True)
    model = ev.TransformerRegressor(**dict(checkpoint['model_args'], device='cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    records, successes = [], {}
    begun = time.monotonic()
    for name in args.cores:
        core = CORES[name]
        target = core['target']
        target_sym = physics.symbolic(target)
        expanded = gd.full_expand_expression(gd.expand_simple_expression(target))
        seen = set()
        for variant in range(args.variants):
            seed = 739100+variant
            random.seed(seed)
            expression = expanded if variant == 0 else gd.scramble(expanded,Ngamma=2,N=4,min_scr=1,max_scr=4,full_expand=True)
            if expression in seen:
                continue
            seen.add(expression)
            assert 'F_' not in expression
            assert physics.on_shell(physics.symbolic(expression)-target_sym) == 0
            tokens = tokenizer.encode_infix(expression)
            assert 1 not in tokens and len(tokens)+2 <= 5000
            source = torch.tensor([[2]+tokens+[3]])
            modes = ['greedy']
            if args.beam_all or variant % args.beam_every == 0:
                modes.append('beam')
            row = dict(core=name,variant=variant,seed=seed,input=expression,input_tokens=len(tokens),trials=[])
            selected = None
            for mode in modes:
                start = time.monotonic()
                with torch.inference_mode():
                    _, _, evidence = ev.decode_with_model(model, source, max_length=256,decoding_method=mode,beam_size=20,return_diagnostics=True)
                for candidate in evidence[0]['candidates']:
                    raw = candidate['raw_token_ids']
                    inspected = ev.inspect_sequence(raw,tokenizer)
                    prediction = inspected['expr']
                    exact = False
                    error = None
                    if inspected['completed'] and inspected['decode_ok']:
                        try:
                            exact = physics.on_shell(physics.symbolic(prediction)-target_sym) == 0
                        except Exception as exc:
                            error = f'{type(exc).__name__}: {exc}'
                    shorter = len(inspected['tokens']) < len(tokens)
                    trial = dict(mode=mode,decoder=candidate,inspection=inspected,exact_on_shell=bool(exact),shorter=shorter,verification_error=error,inference_elapsed=time.monotonic()-start)
                    row['trials'].append(trial)
                    if exact and shorter and inspected['completed'] and candidate['eos_emitted']:
                        selected = trial
                        break
                if selected:
                    break
            row['selected'] = selected
            records.append(row)
            (HERE/f'{args.output_prefix}results.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
            print(json.dumps(dict(core=name,variant=variant,tokens=len(tokens),modes=modes,success=selected is not None,elapsed=time.monotonic()-begun)),flush=True)
            if selected:
                prediction = selected['inspection']['expr']
                success = dict(input=expression,prediction=prediction,variant=variant,seed=seed,mode=selected['mode'],rank=selected['decoder']['rank'])
                if core['external']:
                    reconstruction = f"({core['external']})*({prediction})"
                    success['reconstructed_physical_amplitude'] = reconstruction
                    success['reconstruction_exact_on_shell'] = physics.on_shell(physics.symbolic(reconstruction)-physics.symbolic(physics.expressions()['simple'])) == 0
                    assert success['reconstruction_exact_on_shell']
                successes[name] = success
                break
    summary = dict(cores=args.cores,successful_cores=successes,model_calls=sum(len(set(t['mode'] for t in r['trials'])) for r in records),unique_inputs=len(records),elapsed_seconds=time.monotonic()-begun,reference_supplied_to_inference=False,checkpoint_updated=False,scope='external scalar normalization and reconstruction')
    (HERE/f'{args.output_prefix}summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
