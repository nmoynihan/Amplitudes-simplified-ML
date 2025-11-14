#!/usr/bin/env python3
"""Check numeric evaluation of pairs in fail1.csv using gen_data.eval_infix_numeric

This script finds `fail1.csv` in the repo, imports `gen_data.eval_infix_numeric`
and `kinematics.generate_kinematics`, then evaluates each (simple,scrambled)
pair and prints numeric comparisons.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import importlib.util
import math

# Add data_generation to path for consistent imports
sys.path.insert(0, str(Path(__file__).parent.parent / "data_generation"))


def find_file(name: str) -> Path | None:
    p = Path.cwd()
    for f in p.rglob(name):
        return f
    return None


def load_module_from_path(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def max_p_index(expr: str) -> int:
    # crude scan for p_<n> occurrences
    import re
    m = re.findall(r"p_(\d+)", expr)
    if not m:
        return 0
    return max(map(int, m))


def compare_pair(eval_fn, kin_fn, simple: str, scrambled: str, row_idx: int):
    N = max(max_p_index(simple), max_p_index(scrambled))
    if N < 2:
        print(f"Row {row_idx}: cannot detect N (found <2). Skipping")
        return
    # use deterministic seed per row
    seed = 1000 + row_idx
    mom, pol = kin_fn(N, M=2.0, seed=seed)
    try:
        va = eval_fn(simple, mom, pol)
        vb = eval_fn(scrambled, mom, pol)
    except Exception as e:
        print(f"Row {row_idx}: evaluation raised exception: {e}")
        try:
            if hasattr(eval_fn, 'to_numeric_string'):
                print("  simple numeric str:")
                print(eval_fn.to_numeric_string(simple, mom, pol))
                print("  scrambled numeric str:")
                print(eval_fn.to_numeric_string(scrambled, mom, pol))
                # show AST node types present to help debug disallowed nodes
                import ast
                for label, s in (('simple', eval_fn.to_numeric_string(simple, mom, pol)), ('scrambled', eval_fn.to_numeric_string(scrambled, mom, pol))):
                    try:
                        tree = ast.parse(s, mode='eval')
                        types = set(type(n).__name__ for n in ast.walk(tree))
                        print(f"    AST node types for {label}: {sorted(types)}")
                    except Exception as e2:
                        print(f"    AST parse failed for {label}: {e2}")
                    # Try to run the module's safe evaluator to capture traceback
                    try:
                        import traceback
                        import data_generation.gen_data as gen
                        try:
                            gen._safe_eval_float(s)
                        except Exception:
                            print("    gen._safe_eval_float traceback:")
                            traceback.print_exc()
                    except Exception:
                        pass
                    # check which AST node types lack visitor methods in gen._SafeEval
                    try:
                        import data_generation.gen_data as gen
                        SE = gen._SafeEval
                        missing = set()
                        tree = ast.parse(eval_fn.to_numeric_string(simple, mom, pol), mode='eval')
                        for node in ast.walk(tree):
                            vname = 'visit_' + type(node).__name__
                            if not hasattr(SE, vname):
                                missing.add(type(node).__name__)
                        if missing:
                            print(f"    gen._SafeEval missing handlers for: {sorted(missing)}")
                    except Exception:
                        pass
        except Exception:
            pass
        return
    absd = abs(va - vb)
    reld = absd / (abs(va) + 1e-30)
    ok = (absd < 1e-9) or (reld < 1e-9)
    print(f"Row {row_idx}: N={N}  ok={ok}")
    print(f"  simple    => {va:.12e}")
    print(f"  scrambled => {vb:.12e}")
    print(f"  abs diff  = {absd:.3e}   rel diff = {reld:.3e}")
    if not ok:
        print("  ---- Detailed (numeric strings after expansion):")
        try:
            if hasattr(eval_fn, 'to_numeric_string'):
                print("    simple numeric str:")
                print(eval_fn.to_numeric_string(simple, mom, pol))
                print("    scrambled numeric str:")
                print(eval_fn.to_numeric_string(scrambled, mom, pol))
        except Exception:
            pass


def main():
    csv_path = find_file('fail1.csv')
    if csv_path is None:
        print('fail1.csv not found in repo tree. Please run from repo root or place fail1.csv here.')
        sys.exit(2)

    # Try direct imports first (more reliable), then fall back to dynamic loading
    try:
        import gen_data as gen
        import kinematics as kin
        kin_fn = kin.generate_kinematics
        eval_fn = gen.eval_infix_numeric
        if hasattr(gen, 'to_numeric_string'):
            setattr(eval_fn, 'to_numeric_string', gen.to_numeric_string)
    except ImportError:
        # Fall back to dynamic module loading
        gen_path = find_file('gen_data.py')
        if gen_path is None:
            print('gen_data.py not found in repo tree. Cannot continue.')
            sys.exit(2)

        kin_path = find_file('kinematics.py')
        if kin_path is None:
            print('kinematics.py not found; trying to import from gen_data if available')

        gen = load_module_from_path(gen_path, 'gen_data_local')
        if kin_path is not None:
            kin = load_module_from_path(kin_path, 'kinematics_local')
            kin_fn = getattr(kin, 'generate_kinematics')
        else:
            # fall back to gen_data's internal kinematics if present
            kin_fn = getattr(gen, 'generate_kinematics', None)
            if kin_fn is None:
                print('No kinematics generator found. Exiting.')
                sys.exit(3)

        eval_fn = getattr(gen, 'eval_infix_numeric', None)
        if eval_fn is None:
            print('gen_data.py does not export eval_infix_numeric. Exiting.')
            sys.exit(4)

        # Provide optional helper to show numeric-ready string if available
        # Attach to eval_fn object if gen exported to_numeric_string
        if hasattr(gen, 'to_numeric_string'):
            setattr(eval_fn, 'to_numeric_string', gen.to_numeric_string)

    with csv_path.open('r', newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for i, row in enumerate(reader, start=1):
            if not row or len(row) < 2:
                continue
            simple, scrambled = row[0].strip(), row[1].strip()
            print('\n=== Row %d ===' % i)
            print('simple   :', simple)
            print('scrambled:', scrambled)
            compare_pair(eval_fn, kin_fn, simple, scrambled, i)


if __name__ == '__main__':
    main()
