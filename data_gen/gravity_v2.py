"""Version 2 gravity release. Exact algebra, frozen families, accepted-row quotas.

Run ``python -m data_gen.gravity_v2 --help``. The legacy entry point is unchanged.
All polynomial coefficients are fractions.Fraction; floats occur only in checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from fractions import Fraction
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

from .Tokenizer import ScatteringAmplitudeTokenizer
from .data_gen_gravity.core import BENCHMARKS, PROCESS_SPECS
from .data_gen_gravity.ordered_benchmarks import reconstruction_terms

VERSION = "gravity-v2.0.1"
ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
DIAGNOSIS = ROOT / "gravity-5pt-diagnosis-2026-09-22"
ORIGINAL = Path("/Users/kymani/Documents/Big_YM_Data_Sets/Cyclic_Gravity")
CATEGORIES = ("legacy_single_f", "rational_distinct", "rational_repeated",
              "trace_mixed", "mixed_family", "4s1h_replay")
FAMILIES = ("XXXX", "TXX", "TT", "QXX", "QQ", "TQ")
DENOMINATORS = dict(zip(FAMILIES, (6, 4, 2, 5, 4, 3)))
COEFFICIENTS = tuple(Fraction(n, d) for n in (1, 2, 3, 4, 6, -1, -2, -3, -4, -6)
                     for d in (1, 2, 3, 6)) + (Fraction(-7, 12), Fraction(11, 24))
DEFAULT_DENOMINATORS = tuple(range(1, 13)) + (24,)
TRANSFORMS = ("full_expansion", "term_collection", "common_factor_cancellation",
              "common_denominator", "partial_fraction", "momentum_conservation",
              "transversality", "dot_commutation", "term_order")
METADATA_FIELDS = ("row_id", "split", "process", "helicity_domain", "assigned_category",
                   "feature_tags", "compact_origin", "family_id", "generation_seed",
                   "transformation_path", "coefficients", "pole_multiplicities",
                   "contraction_families", "stage", "simple_tokens", "scrambled_tokens",
                   "validation", "simple", "scrambled")


def d(a, b):
    return ("d", *sorted((a, b)))


def monomial(items):
    counts = Counter()
    for atom, power in items:
        counts[atom] += power
    return tuple(sorted((a, n) for a, n in counts.items() if n))


def add(*polys):
    out = defaultdict(Fraction)
    for poly in polys:
        for m, c in poly.items():
            out[m] += c
    return {m: c for m, c in out.items() if c}


def scale(poly, c):
    return {m: c * v for m, v in poly.items() if c * v}


def mul(left, right):
    out = defaultdict(Fraction)
    for a, c in left.items():
        for b, e in right.items():
            out[monomial(a + b)] += c * e
    return {m: c for m, c in out.items() if c}


def atom_poly(atom):
    return {((atom, 1),): Fraction(1)}


def atom_string(atom):
    kind, *legs = atom
    if kind == "d":
        return f"(p_{legs[0]} · p_{legs[1]})"
    if kind == "ep":
        return f"(e_{legs[0]} · p_{legs[1]})"
    if kind == "ee":
        return f"(e_{legs[0]} · e_{legs[1]})"
    if kind == "X":
        i, a, b = legs
        return f"(p_{a} · F_{i} · p_{b})"
    if kind == "Q":
        i, j, a, b = legs
        return f"(p_{a} · F_{i} · F_{j} · p_{b})"
    if kind == "T":
        return "Tr(" + " · ".join(f"F_{i}" for i in legs) + ")"
    raise ValueError(f"Unsupported atom {atom!r}")


def product_string(atoms, coefficient=1):
    """A numeric leaf always precedes a nonconstant factor in prefix order."""
    factors = [atom_string(a) for a in atoms]
    if coefficient != 1 or not factors:
        factors.insert(0, str(coefficient))
    return "*".join(factors)


def term_string(m, coefficient):
    coefficient = Fraction(coefficient)
    numer = [a for a, n in m for _ in range(max(0, n))]
    denom = [a for a, n in m for _ in range(max(0, -n))]
    if not numer:
        raise ValueError("unsafe_constant_monomial")
    top = product_string(numer, abs(coefficient.numerator))
    bottom = product_string(denom, coefficient.denominator)
    text = f"({top})/({bottom})" if denom or coefficient.denominator != 1 else f"({top})"
    return ("-" if coefficient < 0 else "") + text


def serialize(poly, *, order=None):
    terms = list(sorted(poly.items())) if order is None else order
    if not terms:
        raise ValueError("zero_origin")
    return " + ".join(term_string(m, c) for m, c in terms).replace("+ -", "- ")


def expansion(atom):
    """F_i = p_i tensor e_i - e_i tensor p_i, with the repository metric."""
    kind, *legs = atom
    if kind not in ("X", "T", "Q"):
        return atom_poly(atom)
    if kind == "X":
        i, a, b = legs
        return add(mul(atom_poly(d(a, i)), atom_poly(("ep", i, b))),
                   scale(mul(atom_poly(("ep", i, a)), atom_poly(d(i, b))), -1))
    if kind == "T":
        i, j = legs
        return add(scale(mul(atom_poly(("ep", i, j)), atom_poly(("ep", j, i))), 2),
                   scale(mul(atom_poly(("ee", *sorted((i, j)))), atom_poly(d(i, j))), -2))
    i, j, a, b = legs
    return add(mul(mul(atom_poly(d(a, i)), atom_poly(("ep", i, j))), atom_poly(("ep", j, b))),
               scale(mul(mul(atom_poly(d(a, i)), atom_poly(("ee", *sorted((i, j))))), atom_poly(d(j, b))), -1),
               scale(mul(mul(atom_poly(("ep", i, a)), atom_poly(d(i, j))), atom_poly(("ep", j, b))), -1),
               mul(mul(atom_poly(("ep", i, a)), atom_poly(("ep", j, i))), atom_poly(d(j, b))))


def expand(poly, *, only_one=False):
    out = {}
    for m, c in poly.items():
        term = {(): c}
        expanded_one = False
        for atom, n in m:
            if n < 0:
                term = mul(term, {((atom, n),): Fraction(1)})
                continue
            for _ in range(n):
                use = atom[0] in ("X", "T", "Q") and (not only_one or not expanded_one)
                term = mul(term, expansion(atom) if use else atom_poly(atom))
                expanded_one |= use
        out = add(out, term)
    # Endpoint degeneracies are disallowed when constructing origins; this also
    # removes explicitly transverse/on-shell terms introduced by transformations.
    return {m: c for m, c in out.items()
            if not any(a[0] in ("d", "ep") and a[1] == a[2] and n > 0 for a, n in m)}


def x_atom(i, rng):
    a, b = sorted(rng.sample([j for j in range(1, 6) if j != i], 2))
    return ("X", i, a, b)


def q_atom(rng):
    return ("Q", 4, 5, rng.choice((1, 2, 3, 5)), rng.choice((1, 2, 3, 4)))


def numerator(family, rng):
    t = ("T", 4, 5)
    if family == "XX":
        return [x_atom(5, rng), x_atom(5, rng)]
    if family == "XXXX":
        return [x_atom(4, rng), x_atom(4, rng), x_atom(5, rng), x_atom(5, rng)]
    if family == "TXX":
        return [t, x_atom(4, rng), x_atom(5, rng)]
    if family == "TT":
        return [t, t]
    if family == "QXX":
        return [q_atom(rng), x_atom(4, rng), x_atom(5, rng)]
    if family == "QQ":
        return [q_atom(rng), q_atom(rng)]
    if family == "TQ":
        return [t, q_atom(rng)]
    raise ValueError(family)


def coefficient_pool(numerator_max=12, denominators=DEFAULT_DENOMINATORS):
    """Exact, configurable nonzero rational support; no decimal intermediates."""
    if numerator_max < 1 or not denominators or any(type(d) is not int or d < 1 for d in denominators):
        raise ValueError("coefficient numerator bound and denominators must be positive integers")
    return tuple(sorted({Fraction(n, d) for n in range(-numerator_max, numerator_max + 1)
                         if n for d in denominators}))


def stage_for_row(index, direct_fraction=Fraction(3, 4)):
    """Deterministic schedule: ceil(N*f) direct targets in every N-row prefix."""
    ratio = Fraction(direct_fraction)
    if not 0 <= ratio <= 1 or index < 0:
        raise ValueError("direct fraction must be in [0, 1] and row index nonnegative")
    def ceiling(n):
        return (n * ratio.numerator + ratio.denominator - 1) // ratio.denominator
    return "direct" if ceiling(index + 1) > ceiling(index) else "intermediate"


def family_split(family, split_seed):
    """Assign a coefficient/relabeling family before any descendant is made."""
    digest = hashlib.sha256(f"{split_seed}/{family}".encode()).hexdigest()
    return "test" if int(digest[:8], 16) % 5 == 0 else "train"


def sample_origin(category, rng, serial, *, coefficients=None):
    if category not in CATEGORIES:
        raise ValueError(f"unknown category: {category}")
    support = coefficient_pool() if coefficients is None else coefficients
    preferred = tuple(c for c in COEFFICIENTS if c in support)
    process = "4s1h" if category == "4s1h_replay" else "3s2h"
    # Multi-term replay origins ensure adequate diversity even after all scalar
    # relabelings are identified; zero/degenerate origins are rejected later.
    count = rng.choice((1, 2, 3)) if process == "3s2h" else rng.choice((2, 3))
    if category == "mixed_family":
        count = rng.choice((2, 3))
        families = ["XXXX", FAMILIES[1 + serial % 5]] + ([rng.choice(FAMILIES)] if count == 3 else [])
    elif category == "trace_mixed":
        families = [FAMILIES[1 + (serial + k) % 5] for k in range(count)]
    else:
        families = ["XX" if process == "4s1h" else "XXXX"] * count
    out, coeffs, poles = {}, [], []
    for k, family in enumerate(families):
        atoms = numerator(family, rng)
        extra = rng.randrange(5) == 0
        all_poles = [d(*p) for p in itertools.combinations(range(1, 6), 2)]
        extra_atom = rng.choice(all_poles) if extra else None
        pool = [p for p in all_poles if p != extra_atom]
        multiplicity = (4 if family == "XX" else DENOMINATORS[family]) + extra
        repeated = category in ("rational_repeated", "trace_mixed", "mixed_family")
        if repeated and multiplicity > 1:
            pole = rng.choice(pool)
            power = min(multiplicity, 2 + serial % 3)
            denominators = [pole] * power
            while len(denominators) < multiplicity:
                options = [p for p in pool if denominators.count(p) < 4]
                denominators.append(rng.choice(options))
        else:
            denominators = rng.sample(pool, multiplicity)
        if extra_atom:
            atoms.append(extra_atom)
        if category in ("legacy_single_f", "4s1h_replay"):
            c = Fraction(rng.choice((-1, 1)))
        elif (serial + k) % 3 == 0 and preferred:
            c = preferred[(serial // 3 + k) % len(preferred)]
        else:
            c = rng.choice(support)
        m = monomial([(a, 1) for a in atoms] + [(a, -1) for a in denominators])
        out = add(out, {m: c})
        coeffs.append(str(c))
        poles.append({atom_string(a): n for a, n in Counter(denominators).items()})
    if len(out) != count:
        raise ValueError("origin_terms_cancel_or_collect")
    if category not in ("legacy_single_f", "4s1h_replay") and all(abs(Fraction(c)) == 1 for c in coeffs):
        raise ValueError("missing_nonunit_coefficient")
    return process, out, families, coeffs, poles


def common_denominator(poly, cancel_atom=None):
    """Collect rational terms over their exact monomial LCM, then optionally
    expose a common factor inside the *sum* in its numerator.
    """
    lower = {}
    for m in poly:
        for a, n in m:
            lower[a] = min(lower.get(a, 0), n)
    if cancel_atom:
        lower[cancel_atom] = lower.get(cancel_atom, 0) - 1
    denominator = [a for a, n in sorted(lower.items()) for _ in range(-n)]
    # Integer LCM retains exact coefficients and prevents adjacent numeric leaves.
    from math import lcm
    divisor = lcm(*(c.denominator for c in poly.values()))
    top = {}
    for m, c in poly.items():
        shifted = monomial(m + tuple((a, -n) for a, n in lower.items()))
        top[shifted] = c * divisor
    return f"({serialize(top)})/({product_string(denominator, divisor)})"


def source_variant(expanded, mode, rng):
    """All transformations preserve rational identities, or explicitly use the
    massless momentum/transversality constraints checked by the validator.
    """
    items = list(sorted(expanded.items()))
    if mode == "full_expansion":
        return serialize(expanded)
    if mode == "term_collection":
        # Independent term coefficients, including A+A -> 2*A and mixed signs.
        order = []
        for k, (m, c) in enumerate(items):
            if k % 3 == 0:
                order.extend(((m, c / 2), (m, c / 2)))
            elif k % 3 == 1:
                order.extend(((m, c * 2), (m, -c)))
            else:
                order.append((m, c))
        rng.shuffle(order)
        return serialize(expanded, order=order)
    if mode == "common_factor_cancellation":
        existing = [a for m in expanded for a, n in m if a[0] == "d" and n < 0]
        return common_denominator(expanded, rng.choice(existing))
    if mode == "common_denominator":
        return common_denominator(expanded)
    if mode == "partial_fraction":
        # First combine over the LCM, then distribute and cancel each numerator
        # monomial separately: N/(D*x*y) -> N1/(D*x) + N2/(D*y).
        # The independent parser performs the exact rational decomposition.
        from .data_gen_gravity.v2_validation import parse_laurent
        combined = common_denominator(expanded)
        decomposed = parse_laurent(combined)
        if decomposed != expanded:
            raise ValueError("partial_fraction_identity")
        order = list(reversed(sorted(decomposed.items())))
        return serialize(decomposed, order=order)
    if mode == "momentum_conservation":
        choices = [(m, c, a, n) for m, c in items for a, n in m
                   if a[0] == "ep" and n > 0]
        m, c, atom, n = rng.choice(choices)
        i, a = atom[1:]
        rest = monomial(m + ((atom, -1),))
        replacement = {monomial(rest + ((("ep", i, b), 1),)): -c
                       for b in range(1, 6) if b != a and b != i}
        return serialize(add(expanded, {m: -c}, replacement))
    if mode == "transversality":
        choices = [(m, c, a, n) for m, c in items for a, n in m
                   if a[0] == "ep" and n > 0]
        m, c, atom, _ = rng.choice(choices)
        zero = monomial(m + ((atom, -1), (("ep", atom[1], atom[1]), 1)))
        return serialize(add(expanded, {zero: c}))
    if mode == "dot_commutation":
        text = serialize(expanded)
        return re.sub(r"\(([pe]_\d+) · ([pe]_\d+)\)", lambda m: f"({m[2]} · {m[1]})", text)
    if mode == "term_order":
        rng.shuffle(items)
        return serialize(expanded, order=items)
    raise ValueError(mode)


def exact_onshell(poly):
    """Eliminate ep(i,5) using sum(p)=0 and remove ep(i,i), d(i,i).
    Generated momentum substitutions use only ep; no helicity identity is used.
    """
    out = {}
    for m, c in poly.items():
        term = {(): c}
        for a, n in m:
            if a[0] in ("d", "ep") and a[1] == a[2]:
                if n < 0:
                    raise ValueError("singular_symbolic_denominator")
                term = {}
                break
            if a[0] == "ep" and a[2] == 5 and a[1] != 5 and n > 0:
                replacement = {((("ep", a[1], j), 1),): Fraction(-1)
                               for j in range(1, 5) if j != a[1]}
                for _ in range(n):
                    term = mul(term, replacement)
            elif a[0] == "ep" and a[1] == 5 and a[2] == 4 and n > 0:
                replacement = {((("ep", 5, j), 1),): Fraction(-1) for j in (1, 2, 3)}
                for _ in range(n):
                    term = mul(term, replacement)
            else:
                term = mul(term, {((a, n),): Fraction(1)})
        out = add(out, term)
    return out


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze_inputs(output):
    files = [ORIGINAL / "Test_Amplitude/gravity5unified12345_seed.csv",
             DIAGNOSIS / "physics/compact_factored.txt", DIAGNOSIS / "physics/compact_same_helicity.txt"]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required holdout reference missing: " + ", ".join(missing))
    files.extend(sorted((PROJECT / "data/gravity").glob("*benchmark*")))
    files.extend(sorted((PROJECT / "data_gen/data_gen_gravity").glob("*benchmark*")))
    files = [p for p in files if p.is_file()]
    result = {str(p): sha(p) for p in files}
    (output / "frozen_inputs.json").write_text(json.dumps(result, indent=2) + "\n")
    references = {p: [BENCHMARKS[p], *reconstruction_terms(p)] for p in PROCESS_SPECS}
    with files[0].open(newline="") as f:
        references["3s2h"].append(next(csv.reader(f))[1])
    references["3s2h"].extend(p.read_text().strip() for p in files[1:3])
    return result, references


def json_text(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def gzip_writer(path, fields):
    # Stable gzip headers make byte-for-byte seeded regeneration possible.
    raw = open(path, "wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    return raw, text, writer


def quotas(train_rows, test_rows):
    if train_rows <= 0 or test_rows <= 0 or train_rows % 40 or test_rows % 10:
        raise ValueError("Training rows must be divisible by 40; test rows by 10")
    return {"train": {c: train_rows // (2 if c == "4s1h_replay" else 10) for c in CATEGORIES},
            "test": {c: test_rows // (2 if c == "4s1h_replay" else 10) for c in CATEGORIES}}


def feature_tags(origin, scrambled):
    """Overlapping features inferred from accepted expressions, not intentions."""
    tags = set()
    if any(c not in (1, -1) for c in origin.values()):
        tags.add("coefficient")
    if any(a[0] in ("T", "Q") for m in origin for a, n in m if n > 0):
        tags.add("trace_or_mixed")
    if re.search(r"e_4\s*·\s*e_5|e_5\s*·\s*e_4", scrambled):
        tags.add("polarization_contraction")
    if any(a[0] == "d" and n < -1 for m in origin for a, n in m):
        tags.add("repeated_pole")
    if {"coefficient", "polarization_contraction", "repeated_pole"} <= tags:
        tags.add("combined_features")
    return sorted(tags)


def update_accepted_coverage(coverage, metadata, token_ids, tokenizer):
    """Call only after all rejection gates and the aligned row writes succeed."""
    for col in ("simple", "scrambled"):
        coverage[f"{col}_digits"].update(tokenizer.id_to_token[i] for i in token_ids[col]
                                         if tokenizer.id_to_token[i].endswith(":"))
        stripped = re.sub(r"[peF]_\d+", "", metadata[col])
        coverage[f"{col}_numeric_literals"].update(re.findall(r"(?<![\w])\d+(?![\w])", stripped))
    for field in ("process", "assigned_category", "stage"):
        coverage[field][metadata[field]] += 1
    coverage["features"].update(json.loads(metadata["feature_tags"]))
    coverage["origin_coefficients"].update(json.loads(metadata["coefficients"]))
    coverage["origin_contraction_families"].update(json.loads(metadata["contraction_families"]))
    coverage["origin_max_pole_multiplicity"].update(str(max(p.values())) for p in json.loads(metadata["pole_multiplicities"]))
    coverage["transformation"].update(json.loads(metadata["transformation_path"])[1:])


def verify_publication_alignment(paths, expected, tokenizer):
    """Read closed files before writing a manifest; reject partial/misaligned data.

    Kept separate from generation so tiny temporary fixtures test this gate
    without invoking any sampling loop or production defaults.
    """
    from .data_gen_gravity.v2_validation import parse_laurent
    counts = {}
    for split, categories in expected.items():
        observed, rows = Counter(), 0
        with ExitStack() as stack:
            readers = []
            for kind in ("raw", "tok", "metadata"):
                handle = stack.enter_context(gzip.open(paths[f"{split}_{kind}"], "rt", newline=""))
                reader = csv.DictReader(handle)
                if reader.fieldnames != list(METADATA_FIELDS if kind == "metadata" else ("simple", "scrambled")):
                    raise ValueError(f"publication_column_mismatch: {split}/{kind}")
                readers.append(reader)
            for index, triplet in enumerate(itertools.zip_longest(*readers)):
                if any(row is None for row in triplet):
                    raise ValueError(f"publication_row_alignment: {split}/{index}")
                raw, tokens, metadata = triplet
                if metadata["row_id"] != f"{split}-{index:06d}" or metadata["split"] != split:
                    raise ValueError(f"publication_row_identity: {split}/{index}")
                if metadata["assigned_category"] not in categories:
                    raise ValueError(f"publication_unknown_category: {split}/{index}")
                for col in ("simple", "scrambled"):
                    ids = json.loads(tokens[col])
                    if (raw[col] != metadata[col] or ids != tokenizer.encode_infix(raw[col])
                            or len(ids) != int(metadata[f"{col}_tokens"])
                            or any(type(i) is not int for i in ids)):
                        raise ValueError(f"publication_content_alignment: {split}/{index}/{col}")
                    if len(ids) + 2 > 5000:
                        raise ValueError("publication_checkpoint_capacity")
                    if parse_laurent(raw[col]) != parse_laurent(tokenizer.decode_infix(ids)):
                        raise ValueError(f"publication_token_semantics: {split}/{index}/{col}")
                observed[metadata["assigned_category"]] += 1
                rows += 1
        if dict(observed) != categories:
            raise ValueError(f"publication_count_mismatch: {split}: {dict(observed)} != {categories}")
        counts[split] = rows
    return counts


def generate(args):
    from .data_gen_gravity.v2_validation import (parse_laurent, PairValidator,
                                                canonical_family, FrozenFamilyGuard)
    wanted = quotas(args.train_rows, args.test_rows)
    coefficients = coefficient_pool(getattr(args, "coefficient_numerator_max", 12),
                                    getattr(args, "coefficient_denominators", DEFAULT_DENOMINATORS))
    if all(abs(c) == 1 for c in coefficients):
        raise ValueError("coefficient support must contain a nonunit value for enriched quotas")
    direct_fraction = Fraction(getattr(args, "direct_fraction", "3/4"))
    stage_for_row(0, direct_fraction)
    if not 1 <= args.max_tokens <= 4998:
        raise ValueError("max_tokens must be in [1, 4998] including room for BOS/EOS")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("gravity_v2_*.csv.gz")):
        raise FileExistsError("Release output files already exist; choose a new directory")
    frozen, references = freeze_inputs(output)
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=args.max_tokens)
    guard = FrozenFamilyGuard(references, seed=args.fingerprint_seed)
    validator = PairValidator(seed=args.validation_seed, momentum_samples=3)
    rejected, accepted = Counter(), {s: Counter() for s in wanted}
    seen_families, seen_sources, expressions = {}, {}, {"train": set(), "test": set()}
    assignments, coverage = {}, {s: defaultdict(Counter) for s in wanted}
    lengths = {s: {"simple": [], "scrambled": []} for s in wanted}
    writers, handles, paths = {}, [], {}
    for split in wanted:
        for kind in ("raw", "tok", "metadata"):
            path = output / f"gravity_v2_{split}_{kind}.csv.gz"
            raw, handle, writer = gzip_writer(path, METADATA_FIELDS if kind == "metadata" else ("simple", "scrambled"))
            paths[f"{split}_{kind}"] = path
            handles.extend((raw, handle))
            writers[split, kind] = writer
    start = time.time()
    heldout_origins = {p: [] for p in PROCESS_SPECS}
    for split in ("test", "train"):
        if split == "train":
            # Reserve every test origin, all species relabelings and projective
            # numerical matches before creating any training descendants.
            guard.add_references(heldout_origins)
        for category in CATEGORIES:
            rng = random.Random(f"{args.seed}/{split}/{category}")
            attempts = 0
            while accepted[split][category] < wanted[split][category]:
                attempts += 1
                if attempts > max(10000, wanted[split][category] * 100):
                    raise RuntimeError(f"Quota stalled: {split}/{category}: {rejected}")
                origin_seed = rng.getrandbits(63)
                local = random.Random(origin_seed)
                try:
                    process, origin, families, coeffs, poles = sample_origin(category, local, attempts,
                                                                           coefficients=coefficients)
                    compact = serialize(origin)
                    family = canonical_family(compact, process, ignore_coefficients=True)
                    assigned = family_split(family, args.split_seed)
                    if assigned != split:
                        rejected["family_assigned_other_split"] += 1
                        continue
                    if family in seen_families:
                        rejected["duplicate_family"] += 1
                        continue
                    if guard.is_reserved(compact, process):
                        rejected["frozen_family"] += 1
                        continue
                    expanded = expand(origin)
                    if not expanded:
                        raise ValueError("zero_origin")
                    variants = 1 if split == "test" else 4
                    pending = []
                    for variant in range(variants):
                        row_number = accepted[split][category] + variant
                        direct = stage_for_row(row_number, direct_fraction) == "direct"
                        target = origin if direct else expand(origin, only_one=True)
                        simple = serialize(target)
                        mode = TRANSFORMS[(row_number // 4 + variant) % len(TRANSFORMS)]
                        if category == "mixed_family":
                            mode = ("term_collection", "common_factor_cancellation", "common_denominator", "partial_fraction")[row_number % 4]
                        scrambled = source_variant(expanded, mode, local)
                        # Strict exact independent parsing after actual encoding is
                        # mandatory: unsafe numeric leaves cannot pass this gate.
                        texts = {"simple": simple, "scrambled": scrambled}
                        token_ids, parsed = {}, {}
                        for col, text in texts.items():
                            token_ids[col] = tokenizer.encode_infix(text)
                            if len(token_ids[col]) + 2 > 5000:
                                raise ValueError("checkpoint_position_capacity")
                            parsed[col] = parse_laurent(text, expand_f=False)
                            decoded = tokenizer.decode_infix(token_ids[col])
                            if parsed[col] != parse_laurent(decoded, expand_f=False):
                                raise ValueError("token_roundtrip_changed_expression")
                        if len(token_ids["simple"]) + 8 > len(token_ids["scrambled"]):
                            raise ValueError("target_not_meaningfully_shorter")
                        exact_target = expand(parsed["simple"])
                        exact_source = expand(parsed["scrambled"])
                        if mode in ("momentum_conservation", "transversality"):
                            exact_target, exact_source = exact_onshell(exact_target), exact_onshell(exact_source)
                        if exact_target != exact_source:
                            raise ValueError("exact_pair_mismatch")
                        result = validator.validate(simple, scrambled, process)
                        if not result["ok"]:
                            raise ValueError("validation_" + result["reason"])
                        source_hash = hashlib.sha256(scrambled.encode()).hexdigest()
                        if source_hash in seen_sources or any(p["source_hash"] == source_hash for p in pending):
                            raise ValueError("duplicate_serialized_input")
                        other = "train" if split == "test" else "test"
                        if any(hashlib.sha256(text.encode()).hexdigest() in expressions[other] for text in texts.values()):
                            raise ValueError("cross_split_expression")
                        tags = feature_tags(origin, scrambled)
                        actual_stage = "direct" if parsed["simple"] == origin else "intermediate"
                        if actual_stage != ("direct" if direct else "intermediate"):
                            raise ValueError("target_stage_mismatch")
                        metadata = dict(row_id="", split=split, process=process, helicity_domain="general_polarization",
                                        assigned_category=category, feature_tags=json_text(sorted(tags)), compact_origin=compact,
                                        family_id=family, generation_seed=origin_seed, transformation_path=json_text(["expand_F", mode]),
                                        coefficients=json_text(coeffs), pole_multiplicities=json_text(poles), contraction_families=json_text(families),
                                        stage=actual_stage, simple_tokens=len(token_ids["simple"]),
                                        scrambled_tokens=len(token_ids["scrambled"]), validation=json_text({**result, "exact": True, "token_roundtrip": True}), **texts)
                        pending.append(dict(metadata=metadata, tokens=token_ids, source_hash=source_hash))
                    seen_families[family] = split
                    if split == "test":
                        heldout_origins[process].append(compact)
                    assignments[family] = dict(split=split, category=category, process=process, variants=len(pending), origin_seed=origin_seed)
                    for pending_row in pending:
                        metadata, token_ids = pending_row["metadata"], pending_row["tokens"]
                        metadata["row_id"] = f"{split}-{sum(accepted[split].values()):06d}"
                        writers[split, "raw"].writerow({c: metadata[c] for c in ("simple", "scrambled")})
                        writers[split, "tok"].writerow({c: json_text(token_ids[c]) for c in ("simple", "scrambled")})
                        writers[split, "metadata"].writerow(metadata)
                        seen_sources[pending_row["source_hash"]] = metadata["simple"]
                        for col in ("simple", "scrambled"):
                            expressions[split].add(hashlib.sha256(metadata[col].encode()).hexdigest())
                            lengths[split][col].append(len(token_ids[col]))
                        update_accepted_coverage(coverage[split], metadata, token_ids, tokenizer)
                        accepted[split][category] += 1
                    total = sum(sum(c.values()) for c in accepted.values())
                    if total % 1000 == 200 or args.train_rows < 1000:
                        print(json.dumps(dict(rows=total, split=split, category=category, elapsed=round(time.time()-start, 1), rejected=sum(rejected.values()))), flush=True)
                except (ValueError, ZeroDivisionError) as error:
                    reason = str(error)
                    if reason.startswith("Tokenized expression"):
                        reason = "token_cap"
                    rejected[reason[:160]] += 1
            print(f"Completed {split}/{category}: {accepted[split][category]}", flush=True)
    for handle in reversed(handles):
        handle.close()
    if dict(accepted["train"]) != wanted["train"] or dict(accepted["test"]) != wanted["test"]:
        raise AssertionError("accepted quota mismatch")
    published_counts = verify_publication_alignment(paths, wanted, tokenizer)
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise AssertionError(f"Frozen input changed: {path}")
    (output / "family_assignments.json").write_text(json.dumps(assignments, sort_keys=True, indent=2) + "\n")
    import numpy as np
    report = {s: {**{k: dict(v) for k, v in coverage[s].items()},
                  "unique_origins": sum(a["split"] == s for a in assignments.values()),
                  "lengths": {c: dict(min=min(v), max=max(v), mean=float(np.mean(v)),
                                      percentiles=dict(zip(("p10", "p50", "p90", "p99"), map(float, np.percentile(v, [10, 50, 90, 99])))))
                              for c, v in lengths[s].items()}} for s in wanted}
    (output / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    sources = [Path(__file__), PROJECT / "data_gen/Tokenizer.py", PROJECT / "data_gen/ordered_gravity_gen.py",
               *sorted((PROJECT / "data_gen/data_gen_gravity").glob("*.py"))]
    manifest = dict(schema_version=VERSION, generator_version=VERSION, settings=vars(args), source_revision=revision,
                    source_files={str(p.relative_to(PROJECT)): sha(p) for p in sources},
                    tokenizer=dict(format="legacy-prefix-safe-rational-v1", checkpoint_compatible=True,
                                   max_particles=8, vocabulary=tokenizer.vocab, definition_sha256=sha(PROJECT / "data_gen/Tokenizer.py")),
                    requested_counts=wanted, accepted_counts={s: dict(v) for s, v in accepted.items()},
                    total_train_rows=args.train_rows, total_test_rows=args.test_rows, total_rows=args.train_rows+args.test_rows,
                    rejections=dict(rejected), files={k: dict(path=p.name, sha256=sha(p), rows=args.train_rows if k.startswith("train") else args.test_rows) for k, p in paths.items()},
                    frozen_hashes=frozen, family_assignments_sha256=sha(output / "family_assignments.json"),
                    max_variants_per_origin=4, family_split="sha256(split_seed/family_id) mod 5: 0=test, otherwise train; assigned before source variants",
                    direct_fraction=str(direct_fraction), publication_alignment_counts=published_counts,
                    holdout_fingerprint_domains=["general_and_positive", "positive_helicity_only_conservative"],
                    numerical_tolerances=dict(rtol=2e-8, atol=2e-9), exact_identity_scope="general polarization; momentum conservation and transversality where tagged",
                    generation_seconds=time.time()-start, verification="pending independent exported-file audit")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(output), counts=manifest["accepted_counts"], elapsed=manifest["generation_seconds"])), flush=True)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-rows", type=int, default=100000, help="Training rows ONLY; excludes --test-rows")
    parser.add_argument("--test-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--split-seed", type=int, default=730021)
    parser.add_argument("--validation-seed", type=int, default=930103)
    parser.add_argument("--fingerprint-seed", type=int, default=510031)
    parser.add_argument("--audit-seed", type=int, default=830017)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--coefficient-numerator-max", type=int, default=12,
                        help="Sample nonzero numerators from -N..N, before exact reduction")
    parser.add_argument("--coefficient-denominators", type=int, nargs="+", default=list(DEFAULT_DENOMINATORS),
                        help="Positive integer denominator support before exact reduction")
    parser.add_argument("--direct-fraction", default="3/4",
                        help="Exact fraction of direct-to-compact pairs; remainder are intermediate targets")
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
