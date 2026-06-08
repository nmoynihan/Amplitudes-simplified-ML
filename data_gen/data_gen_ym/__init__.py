"""Colour-ordered all-gluon amplitude data generator.

Produces (simple, scrambled) training pairs for colour-ordered n-gluon amplitudes
(fixed ordering 1…n, no colour factors). All N external legs are massless gluons,
each carrying a polarisation e_i and field strength F_i.

Pipeline (per sample, in generate.build_dataset):
    expr_model._build_base_expression   build a gauge-invariant numerator (F-blocks)
                                         over physical (adjacent p_i·p_{i+1}) + F-cancellable
                                         (chain-endpoint) denominator poles
      → numerics._validate_pair          simple ≡ expanded     (massless Σp=0 kinematics)
      → scramble.scramble                apply algebra-preserving moves (ward / momentum / …)
      → numerics._validate_pair          expanded ≡ scrambled
      → algebra.simplify_to_lowest_terms
      → numerics._validate_pair          simple ≡ simplified
      → token budget → accept
    then: dedupe_pairs → write_csv → tokenise_csv

Modules (leaf → root):
    notation     symbols p/e/F/Tr/dot, gluon_legs, regexes, formatting
    kinematics   generate_kinematics, mdot           (numpy; all-N massless, Σp=0)
    algebra      AST parse / expand / canonicalise / simplify_to_lowest_terms
    numerics     eval_infix_numeric, _validate_pair  (algebra + kinematics)
    scramble     scramble, scr_*, _SCRAMBLER_BY_NAME
    expr_model   blocks, pole pool, rewrite_gi, _generate_term, _build_base_expression
    generate     build_dataset loop, batching, CSV, tokenise, CLI

Run:
    ./run_ym.sh 5 --samples 1000          # convenience wrapper (sets cwd + venv)
    python -m data_gen_ym.generate 5 --samples 1000   # from the data_gen/ directory
"""
