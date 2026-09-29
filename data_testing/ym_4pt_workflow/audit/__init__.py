"""Portable exact checks from the 21 September 2026 four-gluon audit."""

from pathlib import Path


def run_all(source: Path, output_dir: Path) -> dict:
    """Run every independent audit and write the three machine-readable reports.

    Failure raises ``AuditFailure`` (a ``ValueError`` subclass); no identity
    depends on Python's optional ``assert`` statements. No model is imported.
    """
    from . import parke_taylor, symbolic, vertices

    return {
        "symbolic": symbolic.run(source, output_dir),
        "vertices": vertices.run(source, output_dir),
        "parke_taylor": parke_taylor.run(source, output_dir),
    }


__all__ = ["run_all"]
