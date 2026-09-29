# Four-gluon exact physics audit

This package ports the three computations used for the scientific audit dated
21 September 2026. They validate the scalar-contraction input independently of
the model. The [historical report](reference/report.md) explains the physical
conventions, scope, and interpretation. See the [workflow README](../README.md)
for input regeneration and the optional model-dependent stages.

From the repository root, using an environment with SymPy 1.14.0:

```bash
python -m data_testing.ym_4pt_workflow.audit \
  --input data_testing/ym_4pt_workflow/inputs/gluon4feyn1234_model_ready.csv \
  --output-dir /tmp/ym_4pt_audit
```

The checked-in input is the audited 4,473-byte file with SHA256
`f73c1e6b8459876394697b26d4f2585b6f51ed0eee52fa8412dd5ba9fbc6ca0c`.
The command recalculates all results; it does not read the historical result
files. It requires neither a model checkpoint nor Wolfram Language.

| Module | Independent calculation | Output |
| --- | --- | --- |
| `symbolic` | Exact scalar-contraction vertex equality, degrees, Ward identities, cyclic/reflection/decoupling/BCJ relations, poles, residues, current conservation | `symbolic_results.json` |
| `vertices` | Explicit four-vectors at three rational angles; 48 linear states, 48 circular states, generic gauge-shifted polarizations, 12 Ward replacements | `vertices_results.json` |
| `parke_taylor` | Explicit momentum spinors and all 16 helicity assignments as exact functions of scattering angle | `parke_taylor_results.json` |

Each module can also be run separately with the same options, for example
`python -m data_testing.ym_4pt_workflow.audit.symbolic --input INPUT.csv --output-dir OUTPUT`.
Imports perform no file reads, audit runs, or output writes. All identities use
explicit checks that raise `AuditFailure` (a `ValueError` subclass), including
when Python is run with `-O`. Parse and filesystem errors also terminate the run.
A failed stage does not write its result file; earlier successful stage files
can exist, so successful process completion is required for a complete audit.

The Python interface is:

```python
from pathlib import Path
from data_testing.ym_4pt_workflow.audit import run_all

results = run_all(Path("input.csv"), Path("output/audit"))
# results has keys "symbolic", "vertices", "parke_taylor".
```

Each module additionally provides `run(source: Path, output_dir: Path) -> dict`.
The symbolic audit retains its own scalar parser and algebra. The explicit
vertex audit retains a separate parser and four-vector implementation; the
Parke–Taylor audit reuses the four-vector parser and polarization conventions,
then independently constructs and checks spinor formulas, as the original did.
The explicit-vector parser now accepts only arithmetic syntax instead of
passing arbitrary input to `sympify`; its exact parsed expression and all
historical results have been checked for agreement.

Fast malformed-input and failure checks, including optimized Python:

```bash
python -O -m unittest data_testing.ym_4pt_workflow.audit.test_audit -v
```

All fresh JSON files record the input path and SHA256. The historical
Parke–Taylor file did not contain these fields; the portable version adds them.
Scientific expressions can have different string orderings in other SymPy
versions; SymPy 1.14.0 reproduces the historical strings exactly.
