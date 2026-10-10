"""Gauge-project the original scalar CSV; no compact target is used in preparation."""
from pathlib import Path
import csv
import json
import re
import sys

import sympy as sp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_and_validate import symbolic, on_shell, gd, ScatteringAmplitudeTokenizer


def atom(a,b):
    return sp.Symbol('_'.join(sorted((a,b))))


def project(source):
    expression=symbolic(source)
    replacements={}
    for symbol in expression.free_symbols:
        a,b=symbol.name.split('_')
        if a.startswith('e') and b.startswith('p'):
            i,j=int(a[1:]),int(b[1:])
            replacements[symbol]=symbol-atom(a,'p1')*atom(f'p{i}',b)/atom('p1',f'p{i}')
        elif a.startswith('e') and b.startswith('e'):
            i,j=int(a[1:]),int(b[1:])
            ai,aj=atom('p1',f'p{i}'),atom('p1',f'p{j}')
            xi,xj=atom(a,'p1'),atom(b,'p1')
            replacements[symbol]=(symbol-xi*atom(b,f'p{i}')/ai-xj*atom(a,f'p{j}')/aj
                                  +xi*xj*atom(f'p{i}',f'p{j}')/(ai*aj))
    projected=expression.subs(replacements,simultaneous=True)
    # Serialize through the generator's established scalar-product convention.
    text=sp.sstr(projected)
    for symbol in sorted(projected.free_symbols,key=lambda s:-len(s.name)):
        a,b=symbol.name.split('_')
        dot=f'({a[0]}_{a[1:]} · {b[0]}_{b[1:]})'
        text=re.sub(r'\b'+re.escape(symbol.name)+r'\b',dot,text)
    text=gd.simplify_to_lowest_terms(gd.full_expand_expression(text.replace('**','^')))
    assert 'F_' not in text
    assert on_shell(symbolic(text)-expression)==0
    return text


def main():
    source_path=HERE.parent/'sqed_5pt_physical.csv'
    with source_path.open() as f: row=next(csv.DictReader(f))
    prepared=project(row['scrambled'])
    tok=ScatteringAmplitudeTokenizer(max_particles=8,max_sequence_length=5000)
    assert on_shell(symbolic(tok.decode_infix(tok.encode_infix(prepared)))-symbolic(row['scrambled']))==0
    output=HERE/'sqed_5pt_prepared.csv'
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['simple','scrambled']);writer.writeheader()
        writer.writerow(dict(simple=row['simple'],scrambled=prepared))
    report=dict(source=str(source_path),output=str(output),
                preparation='Substitute e_i -> e_i-p_i*(p1.e_i)/(p1.p_i) in the original scalar input; expand and cancel scalar factors.',
                compact_target_used_for_preparation=False,
                exact_original_minus_prepared=0,
                input_tokens=len(tok.encode_infix(row['scrambled'])),
                prepared_tokens=len(tok.encode_infix(prepared)),
                exact_tokenizer_round_trip=True)
    (HERE/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
