# Historical audit evidence

These files retain the scientific evidence produced on 21 September 2026.
They are reference data, not outputs of a fresh run of the packaged workflow.

- [report.md](report.md): original scientific findings, with local links and
  reproduction instructions updated for this portable package.
- [symbolic_results.json](symbolic_results.json) and
  [vertices_results.json](vertices_results.json): original calculation results;
  only `source` is changed to the relative path of the bundled input.
- [parke_taylor_results.json](parke_taylor_results.json): byte-identical original
  evidence, which did not record the input path or hash. The new runner adds both.
- [provenance.json](provenance.json): original relative filenames, original file
  hashes, and hashes of adapted copies where applicable.

The original audit was saved under
`Repo/amplitude-audit-2026-09-21/`, with scripts named `audit_symbolic.py`,
`independent/check_vertices.py`, and `independent/check_parke_taylor.py`.
Those scripts are ported to [symbolic.py](../symbolic.py),
[vertices.py](../vertices.py), and [parke_taylor.py](../parke_taylor.py).

The original input location was
`/Users/kymani/Documents/Big_YM_Data_Sets/gpt_Test_Amplitude_ym_4pt_colour_ordered/gluon4feyn1234_model_ready.csv`.
Its byte-identical [bundled copy](../../inputs/gluon4feyn1234_model_ready.csv)
has SHA256
`f73c1e6b8459876394697b26d4f2585b6f51ed0eee52fa8412dd5ba9fbc6ca0c`.
The source path adaptation does not change this input hash or any scientific
result. Historical JSON source paths are relative to this reference directory;
fresh JSON records the input path passed to the runner.

Reproduce the evidence with the [audit instructions](../README.md).
