import ast

src = open("gen_data.py").read()
tree = ast.parse(src)

# name -> source segment (verbatim)
_lines = src.splitlines()
def _segment(node):
    # include decorators (get_source_segment drops them)
    start = node.lineno
    decos = getattr(node, "decorator_list", [])
    if decos:
        start = min(start, min(d.lineno for d in decos))
    return "\n".join(_lines[start - 1:node.end_lineno])

seg = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        seg[node.name] = _segment(node)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                seg[t.id] = _segment(node)

# ---- module name lists (constants relocated; dead/step dropped) ----
NOTATION = [
    "DOT","dot","p","e","F","Tr","_vec","photon_legs","scalar_legs",
    "_RE_pp","_RE_pFchainp","_RE_TrN","_RE_DOT","_TOKEN_RE",
    "_format_signed_sum","_format_poly","_split_top_level","_strip_matched_outer_parens",
    # shared tuning constants (used by expr_model fns + generate defaults)
    "UNIT_PROBABILITY","OLD_STYLE_PROBABILITY","SPURIOUS_REPEAT_PROBABILITY",
    "DENOM_REPEAT_PROBABILITY","SCALAR_POWER_PROBABILITY",
    "SCALAR_COEFF_POOL","TERM_COEFF_POOL",
    "N4_BLOCK_WEIGHTS","OLD_STYLE_N4_BLOCK_WEIGHTS","GENERAL_BLOCK_WEIGHTS",
    "DEFAULT_MIN_TERMS","DEFAULT_MAX_TERMS","DEFAULT_USE_DENOMINATORS",
    "DEFAULT_MAX_ATTEMPTS_FACTOR",
]
ALGEBRA = [
    "_canon_pp","_canon_TrN","_factor_sort_key","canonicalise_gi_product","canonicalise_denominator",
    "_RatTerm","_is_zero_coeff","_format_number","_format_factor","_node_to_factor","_negate_terms",
    "_mul_terms","_divide_terms","_rational_terms","_format_rational_term","full_expand_expression",
    "simplify_to_lowest_terms",
    "_Num","_Vec","_DotChain","_BinOp","_UnaryOp","_DOT_TAG","_VEC_TAG","_Parser","_tokenize",
    "_mk_dot","_vn","_ast_add","_ast_sub","_ast_mul","_expand_dotchain","_expand_ast",
    "_prec","_vec_name","_ast_to_infix",
]
NUMERICS = ["DEFAULT_VALIDATION_POL_MODES","eval_infix_numeric","_validate_pair"]
SCRAMBLE = [
    "SCRAMBLE_MULTIPLY_ONE","SCRAMBLE_WARD","SCRAMBLE_MOMENTUM","SCRAMBLE_COMMUTE_DOT","SCRAMBLE_RATIO",
    "SCRAMBLE_MASS_SHELL_ZERO","SCRAMBLE_PARTIAL_FRACTION","SCRAMBLE_WARD_ALL","SCRAMBLE_POLARISATION_ZERO",
    "SCRAMBLE_TERM_REORDER","DEFAULT_SCRAMBLES","DEFAULT_MIN_SCR","DEFAULT_MAX_SCR","DEFAULT_MAX_SCRAMBLED_LEN",
    "scr_mul_by_one","scr_ward_substitute","scr_momentum_substitute","scr_commute_dot","scr_mul_by_ratio",
    "_mass_shell_zero_relation","_random_context_factor","scr_add_mass_shell_zero",
    "_find_denom_blocks","_find_paren_block_ending_at","scr_partial_fraction",
    "scr_ward_substitute_all","scr_add_polarisation_zero","_split_signed_terms","_join_signed_terms",
    "scr_term_reorder","_SCRAMBLER_BY_NAME","normalise_scramble_names","_active_scramblers","scramble",
]
EXPR_MODEL = [
    "BlockSpec","MonomialSpec",
    "_rw_pFchainp","_rw_TrN","rewrite_gi","expand_simple_term",
    "_chain_endpoints","_singleF_block","_doubleF_block","_tripleF_block","_tr2_block","_tr3_block",
    "_tr4_block","_scalar_pp_factor","_block_mass_dimension","_all_physical_poles",
    "_required_denominator_count","_weighted_choice","_block_choice_weights","_generate_gi_monomial_spec",
    "_photon_pole_choices","_explicit_scalar_pp_counts","_chain_expansion_spurious_pp_counts",
    "_physical_denominator_factors","_term_signature","_generate_term","_has_supported_physical_poles",
    "manifest_mass_dimension","_build_base_expression",
]
GENERATE = [
    "DEFAULT_N_PARTICLES","DEFAULT_SAMPLES","DEFAULT_SEED","DEFAULT_MASS","NSAMPS",
    "DEFAULT_RAW_OUT_TEMPLATE","DEFAULT_TOK_OUT_TEMPLATE","DEFAULT_LOG_OUT_TEMPLATE",
    "DEFAULT_VALIDATE","DEFAULT_TOKENISE","DEFAULT_FULL_EXPAND_SCRAMBLED","DEFAULT_OVERSAMPLE_FACTOR",
    "DEFAULT_DATASET_KIND","DEFAULT_MAX_TOKENS","DEFAULT_BATCH_SIZE","DEFAULT_JOBS","DEFAULT_PROGRESS",
    "DEFAULT_TOKENIZER_MAX_PARTICLES","BatchJob",
    "_make_tokenizer","_within_token_budget","build_dataset","_batch_sizes","_progress",
    "_worker_build_dataset","build_dataset_batched","_resolve_jobs","dedupe_pairs","write_csv",
    "tokenise_csv",
]

# lineno for ordering
lineno = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        lineno[node.name] = node.lineno
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                lineno[t.id] = node.lineno

HEADERS = {
"notation.py": "from __future__ import annotations\nimport re\nfrom typing import Sequence\n",
"algebra.py": "from __future__ import annotations\nimport re\nfrom dataclasses import dataclass\nfrom typing import Sequence\n\nimport sympy as sp\n\nfrom notation import *\n",
"numerics.py": "from __future__ import annotations\nimport math\nfrom typing import Sequence\n\nfrom notation import *\nfrom algebra import *\nfrom kinematics import generate_kinematics, mdot\n",
"scramble.py": "from __future__ import annotations\nimport random\nimport re\nfrom typing import Sequence\n\nfrom notation import *\nfrom algebra import *\n",
"expr_model.py": "from __future__ import annotations\nimport random\nimport re\nfrom dataclasses import dataclass\nfrom typing import Sequence\n\nfrom notation import *\nfrom algebra import *\n",
"generate.py": "from __future__ import annotations\nimport argparse\nimport csv\nimport json\nimport multiprocessing as mp\nimport os\nimport random\nimport time\nfrom dataclasses import dataclass\nfrom typing import Iterable, Sequence\n\nfrom notation import *\nfrom algebra import *\nfrom kinematics import generate_kinematics, mdot\nfrom numerics import *\nfrom scramble import *\nfrom expr_model import *\n",
}

MODULES = {
"notation.py": NOTATION, "algebra.py": ALGEBRA, "numerics.py": NUMERICS,
"scramble.py": SCRAMBLE, "expr_model.py": EXPR_MODEL, "generate.py": GENERATE,
}

import os
os.makedirs("data_gen_ym", exist_ok=True)
open("data_gen_ym/__init__.py","w").close()

for fname, names in MODULES.items():
    missing = [n for n in names if n not in seg]
    if missing:
        raise SystemExit(f"{fname}: missing segments {missing}")
    names_sorted = sorted(names, key=lambda n: lineno[n])
    parts = [f'"""{fname[:-3]} — extracted from gen_data.py (scaffold, verbatim)."""']
    parts.append(HEADERS[fname].rstrip("\n"))
    body = "\n\n\n".join(seg[n] for n in names_sorted)
    parts.append(body)
    all_list = "__all__ = [\n" + "".join(f"    {n!r},\n" for n in names) + "]"
    parts.append(all_list)
    open(f"data_gen_ym/{fname}","w").write("\n\n".join(parts) + "\n")
    print(f"wrote data_gen_ym/{fname}  ({len(names)} names)")

# idempotent: drop step-mode routing in the worker
_g = open("data_gen_ym/generate.py").read()
_g = _g.replace(
    'builder = build_step_dataset if job.dataset_kind == "step" else build_dataset',
    'builder = build_dataset  # step mode dropped in YM rewrite')
open("data_gen_ym/generate.py", "w").write(_g)

# main block into generate.py (append verbatim)
main_node = [n for n in tree.body if isinstance(n, ast.If)][-1]
main_src = ast.get_source_segment(src, main_node)
with open("data_gen_ym/generate.py","a") as fh:
    fh.write("\n\n" + main_src + "\n")
print("appended __main__ to generate.py")
