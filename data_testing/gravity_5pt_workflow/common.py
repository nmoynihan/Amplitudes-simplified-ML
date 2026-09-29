"""Input handling and explicit scientific checks shared by the workflow."""
from pathlib import Path
import csv
import gzip
import hashlib
import json

SEEDS = tuple(range(401, 421))
REFERENCE_MODES = ("first", "last", "random")


def require(condition, message):
    """Keep validation enabled even when Python is invoked with -O."""
    if not condition:
        raise ValueError(message)


def read_source(path: Path) -> str:
    """Read exactly one headerless id,expression CSV row, optionally gzipped."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    require(len(rows) == 1 and len(rows[0]) == 2 and bool(rows[0][1].strip()),
            "Expected one headerless CSV row with exactly two columns: id,expression.")
    require(rows[0][1].strip().lower() not in {"expression", "scrambled", "simple"},
            "The source must be a headerless id,expression CSV.")
    return rows[0][1].strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def numeric_check(left: str, right: str):
    """80 checks in the 4D massless ++ sector, not a general-helicity proof."""
    from data_gen.data_gen_gravity.core import numerically_equivalent
    return numerically_equivalent(left, right, "3s2h", seeds=SEEDS,
                                  reference_modes=REFERENCE_MODES, gauge_shift=True)
