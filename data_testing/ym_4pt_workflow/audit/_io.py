"""Input/output only; the audits keep their own amplitude algebra and parsers."""

import argparse
import csv
import io
import json
from pathlib import Path


class AuditFailure(ValueError):
    """An exact identity, kinematic assumption, or input check failed."""


def require(condition, message):
    """Check an invariant even when Python runs with optimization enabled."""
    if not condition:
        raise AuditFailure(str(message))


def read_source(source):
    source = Path(source)
    data = source.read_bytes()
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    require(len(rows) == 1 and len(rows[0]) == 2,
            "Expected exactly one headerless CSV record with ID and amplitude columns")
    return source, data, rows


def write_result(output_dir, filename, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def command_line(run, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, type=Path, help="One-row scalar-contraction CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for audit JSON evidence")
    args = parser.parse_args()
    run(args.input, args.output_dir)
    print(f"Exact audit passed. Results: {args.output_dir.resolve()}")
