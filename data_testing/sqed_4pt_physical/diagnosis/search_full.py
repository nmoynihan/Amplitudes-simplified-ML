"""Bounded exact search over equivalent complete SQED4 scalar inputs."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
import build_and_validate as physics
from data_gen import gen_data as gd
from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from transformer.transformer_functions import TransformerRegressor, decode_with_model


def ordered_terms(expression, order):
    split = re.split(r' ([+-]) ', expression)
    first = split[0]
    terms = [(-1 if first.startswith('-') else 1, first[1:] if first.startswith('-') else first)]
    terms += [(1 if split[i] == '+' else -1, split[i+1]) for i in range(1,len(split),2)]
    arranged = [terms[i] for i in order]
    result = ('-' if arranged[0][0] < 0 else '')+arranged[0][1]
    return result+''.join((' - ' if sign < 0 else ' + ')+term for sign,term in arranged[1:])


def main():
    torch.set_num_threads(1)
    model_hash = hashlib.file_digest(physics.MODEL.open('rb'),'sha256').hexdigest()
    saved = torch.load(physics.MODEL, map_location='cpu', weights_only=True)
    model = TransformerRegressor(**dict(saved['model_args'],device='cpu'))
    model.load_state_dict(saved['model_state_dict']); model.eval()
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8,max_sequence_length=5000)
    original = physics.expressions()
    reference = physics.symbolic(original['simple'])
    expanded = gd.expand_simple_expression(original['simple'])
    full = gd.full_expand_expression(expanded)
    eliminated = ('2*(e_2 · e_3)'
                  ' - 2*(e_2 · p_1)*(e_3 · p_1)/(p_1 · p_2)'
                  ' - 2*(e_2 · p_1)*(e_3 · p_2)/(p_1 · p_2)'
                  ' - 2*(e_3 · p_1)*(e_2 · p_1)/(p_1 · p_3)'
                  ' - 2*(e_3 · p_1)*(e_2 · p_3)/(p_1 · p_3)')
    cases = [('f_expansion_grouped',expanded),('f_expansion_full',full),
             ('momentum_eliminated',eliminated)]
    for number, order in enumerate(itertools.permutations(range(4))):
        cases.append((f'full_term_order_{number:02d}',ordered_terms(full,order)))
    for seed in range(8):
        random.seed(3000+seed)
        cases.append((f'training_scramble_{seed}',gd.scramble(full,2,4,min_scr=1,max_scr=4,full_expand=True)))
    # Equivalent commuting dot orientation often alters training serialization.
    def orient(match):
        a,b=match.groups()
        return f'{b} · {a}' if a.startswith('p_') and b.startswith('e_') else match.group(0)
    cases += [(name+'_e_first',re.sub(r'((?:p|e)_\d+) · ((?:p|e)_\d+)',orient,expr))
              for name,expr in cases[:3]]
    records=[]; seen=set(); started=time.monotonic(); success=None
    for name,expression in cases:
        ids=tokenizer.encode_infix(expression)
        if tuple(ids) in seen: continue
        seen.add(tuple(ids))
        assert 'F_' not in expression and physics.on_shell(physics.symbolic(expression)-reference)==0
        source=torch.tensor([[2,*ids,3]])
        record=dict(name=name,input=expression,input_content_tokens=len(ids),exact_input_equivalence=True,trials=[])
        for mode in ('greedy','beam'):
            with torch.inference_mode():
                _,_,diagnostics=decode_with_model(model,source,max_length=256,decoding_method=mode,
                                                  beam_size=20,return_diagnostics=True)
            for candidate in diagnostics[0]['candidates']:
                raw=candidate['raw_token_ids']
                completed=bool(candidate['completed'])
                output=None; error=None; equal=False
                try:
                    if not completed or raw[0] != 2 or 3 not in raw[1:]:
                        raise ValueError('No authoritative completed sequence')
                    content=raw[1:raw.index(3)]
                    if any(v in (0,1,2,3) for v in content): raise ValueError('Invalid special token')
                    output=tokenizer.decode_infix(content)
                    equal=physics.on_shell(physics.symbolic(output)-reference)==0
                except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError) as exc:
                    error=str(exc)
                trial=dict(mode=mode,rank=candidate['rank'],raw_token_ids=raw,
                           eos_emitted=candidate['eos_emitted'],completed=completed,
                           prediction=output,error=error,exact_equal=bool(equal))
                record['trials'].append(trial)
                if equal:
                    success=dict(case=name,input=expression,prediction=output,mode=mode,rank=candidate['rank'],
                                 input_tokens=len(ids),prediction_tokens=len(content),scope='complete physical amplitude')
                    break
            if success: break
        records.append(record)
        (HERE/'full_search_results.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(dict(case=name,tokens=len(ids),success=bool(success))),flush=True)
        if success: break
    summary=dict(attempted_inputs=len(records),retained_candidates_checked=sum(len(r['trials']) for r in records),
                 success=success,checkpoint_sha256=model_hash,elapsed_seconds=time.monotonic()-started,
                 weights_changed=False,compact_target_sent_to_inference=False,
                 scope='Bounded search of exact equivalent scalar-only complete-amplitude serializations')
    (HERE/'full_search_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
