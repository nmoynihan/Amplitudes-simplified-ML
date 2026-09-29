"""Independently audit every exported gravity-v2 row, without modifying inputs.

The audit streams aligned raw/token/metadata files and writes audit.json only
when every release gate succeeds. Momentum and fingerprint seeds differ from
construction. Fingerprints complement structural checks; they are not proofs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from fractions import Fraction
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import numpy as np

from .Tokenizer import ScatteringAmplitudeTokenizer
from . import gen_data as sqed
from .data_gen_gravity.core import BENCHMARKS, PROCESS_SPECS, parse_expression
from .data_gen_gravity.ordered_benchmarks import reconstruction_terms
from .data_gen_gravity.v2_validation import (
    FrozenFamilyGuard, PairValidator, canonical_family, check_process,
    expand_laurent, parse_laurent, relabel_laurent, species_mappings,
)

CATEGORIES = ("legacy_single_f", "rational_distinct", "rational_repeated", "trace_mixed", "mixed_family", "4s1h_replay")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _multiply(left, right):
    result = defaultdict(Fraction)
    for lm, lc in left.items():
        for rm, rc in right.items():
            powers = defaultdict(int)
            for atom, power in lm + rm:
                powers[atom] += power
            monomial = tuple(sorted((a, p) for a, p in powers.items() if p))
            result[monomial] += lc * rc
    return {m: c for m, c in result.items() if c}


def canonical_onshell(poly):
    """Independent exact reduction by e_i.p_i=0, p_i.p_i=0 and sum(p)=0.

    Eliminate e_4.p_5 and e_5.p_4, leaving independent transverse dot variables.
    This uses no helicity relation and does not invoke the generator algebra.
    """
    result = defaultdict(Fraction)
    for monomial, coefficient in poly.items():
        term = {(): coefficient}
        for atom, exponent in monomial:
            kind, *legs = atom
            if kind in ("d", "ep") and legs[0] == legs[1]:
                require(exponent > 0, "singular on-shell denominator")
                term = {}
                break
            if kind == "ep" and ((legs[0] != 5 and legs[1] == 5) or (legs[0] == 5 and legs[1] == 4)) and exponent > 0:
                eliminated = 5 if legs[0] != 5 else 4
                replacement = {((("ep", legs[0], j), 1),): Fraction(-1)
                               for j in range(1, 6) if j not in (legs[0], eliminated)}
                for _ in range(exponent):
                    term = _multiply(term, replacement)
            else:
                term = _multiply(term, {((atom, exponent),): Fraction(1)})
        for m, c in term.items():
            result[m] += c
    return {m: c for m, c in result.items() if c}


def projective_matches(values, references, *, rtol=2e-8, atol=2e-9):
    """Conservative scale comparison in general and positive-helicity domains.

    A positive-helicity-only equivalence also reserves the family, while pair
    correctness is still independently required for general polarizations.
    """
    if not len(references):
        return np.zeros(0, dtype=bool)
    refs = np.asarray(references)
    values = np.asarray(values)

    def compare(candidate, grid):
        pivots = np.argmax(np.abs(grid), axis=1)
        pivot_values = grid[np.arange(len(grid)), pivots]
        valid = np.isfinite(grid).all(axis=1) & (np.abs(pivot_values) >= 1e-14)
        ratios = np.divide(candidate[pivots], pivot_values, out=np.zeros(len(grid), complex), where=valid)
        expected = ratios[:, None] * grid
        delta = np.abs(candidate[None, :] - expected)
        scale = np.maximum(np.abs(candidate)[None, :], np.abs(expected))
        matched = np.all((delta <= atol) | (delta <= rtol * scale), axis=1)
        return valid & (np.abs(candidate[pivots]) >= 1e-14) & matched

    matched = compare(values, refs)
    if len(values) % 5 == 0:
        positive = [j for j in range(len(values)) if j % 5 != 4]
        matched |= compare(values[positive], refs[:, positive])
    return matched


def explicit_pole_multiplicities(expression):
    """Count repeated p.p factors in the actual serialized denominator subtrees.

    This deliberately precedes cancellation; it measures visible syntax rather
    than physical singularities. Algebraically reduced counts are separate.
    """
    result = Counter()

    def factors(node):
        if isinstance(node, sqed._DotChain):
            parts = node.parts
            if len(parts) == 2 and all(isinstance(p, sqed._Vec) and p.tag == "p" for p in parts):
                return Counter({tuple(sorted(p.idx for p in parts)): 1})
            return Counter()
        if isinstance(node, sqed._BinOp) and node.op == "*":
            return factors(node.left) + factors(node.right)
        if isinstance(node, sqed._BinOp) and node.op == "**" and isinstance(node.right, sqed._Num):
            n = int(node.right.value)
            return Counter({k: n * v for k, v in factors(node.left).items()})
        if isinstance(node, sqed._UnaryOp):
            return factors(node.operand)
        return Counter()

    def visit(node):
        if isinstance(node, sqed._BinOp):
            if node.op == "/":
                result.update(str(n) for n in factors(node.right).values())
            visit(node.left)
            visit(node.right)
        elif isinstance(node, sqed._UnaryOp):
            visit(node.operand)

    visit(parse_expression(expression))
    return result


def frozen_references(frozen):
    import csv
    references = {process: [BENCHMARKS[process], *reconstruction_terms(process)] for process in PROCESS_SPECS}
    amplitude = next((Path(p) for p in frozen if Path(p).name == "gravity5unified12345_seed.csv"), None)
    require(amplitude is not None, "frozen user amplitude missing")
    with amplitude.open(newline="") as handle:
        references["3s2h"].append(next(csv.reader(handle))[1])
    for filename in ("compact_factored.txt", "compact_same_helicity.txt"):
        path = next((Path(p) for p in frozen if Path(p).name == filename), None)
        require(path is not None, f"frozen reference missing: {filename}")
        references["3s2h"].append(path.read_text().strip())
    return references


def audit(release_dir, *, audit_seed=830017):
    import csv
    csv.field_size_limit(10_000_000)
    directory = Path(release_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    settings = manifest["settings"]
    require(audit_seed not in (settings["seed"], settings["validation_seed"], settings["fingerprint_seed"]), "audit seed must be independent")
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=settings["max_tokens"])
    require(manifest["tokenizer"]["vocabulary"] == tokenizer.vocab, "tokenizer vocabulary changed")
    require(sha(Path(__file__).with_name("Tokenizer.py")) == manifest["tokenizer"]["definition_sha256"], "tokenizer definition changed")
    assignments = json.loads((directory / "family_assignments.json").read_text())
    require(sha(directory / "family_assignments.json") == manifest["family_assignments_sha256"], "family assignment hash changed")
    frozen = manifest["frozen_hashes"]
    for path, digest in frozen.items():
        require(sha(path) == digest, f"frozen reference changed: {path}")
    original_records = None
    for baseline in (directory / "baseline/frozen_source_hashes.json", directory.parent / "baseline/frozen_source_hashes.json"):
        if baseline.is_file():
            original_records = json.loads(baseline.read_text())
            break
    # Older releases used an extra diagnostic baseline. A fresh future release
    # already has an explicit immutable-reference manifest and does not require
    # an unrelated prior experiment directory to be auditable.
    if original_records is None:
        original_records = [{"path": path, "sha256": digest} for path, digest in frozen.items()]
    for record in original_records:
        require(sha(record["path"]) == record["sha256"], f"original source changed: {record['path']}")
    guard = FrozenFamilyGuard(frozen_references(frozen), seed=audit_seed + 37)
    validator = PairValidator(seed=audit_seed, momentum_samples=3)
    frozen_fingerprints = {p: np.asarray(fps) for p, fps in guard.fingerprints.items()}
    test_fingerprints = defaultdict(list)
    file_records = {}
    for key, record in manifest["files"].items():
        path = directory / record["path"]
        digest = sha(path)
        require(digest == record["sha256"], f"exported file hash changed: {path}")
        file_records[key] = {"path": str(path), "sha256": digest, "rows": 0}
    origin_cache, family_counts, sources = {}, Counter(), set()
    expression_hashes = {s: set() for s in ("train", "test")}
    algebra_hashes = {s: set() for s in ("train", "test")}
    split_families = {s: set() for s in ("train", "test")}
    counts = {s: defaultdict(Counter) for s in ("train", "test")}
    lengths = {s: {c: [] for c in ("simple", "scrambled")} for s in ("train", "test")}
    worst_error, resamples, started, audited = 0.0, Counter(), time.time(), 0
    for split in ("test", "train"):
        other = "train" if split == "test" else "test"
        with ExitStack() as stack:
            readers = []
            for kind in ("raw", "tok", "metadata"):
                handle = stack.enter_context(gzip.open(file_records[f"{split}_{kind}"]["path"], "rt", newline=""))
                reader = csv.DictReader(handle)
                if kind != "metadata":
                    require(reader.fieldnames == ["simple", "scrambled"], f"wrong {kind} columns")
                readers.append(reader)
            for index, triplet in enumerate(itertools.zip_longest(*readers)):
                require(all(row is not None for row in triplet), f"unaligned {split} file counts at {index}")
                raw, token_row, metadata = triplet
                where = f"{split}:{index}"
                for kind in ("raw", "tok", "metadata"):
                    file_records[f"{split}_{kind}"]["rows"] += 1
                require(metadata["split"] == split and metadata["row_id"] == f"{split}-{index:06d}", f"row identity mismatch at {where}")
                process, category = metadata["process"], metadata["assigned_category"]
                require(category in CATEGORIES, f"unknown category at {where}")
                require(process == ("4s1h" if category == "4s1h_replay" else "3s2h"), f"category/process mismatch at {where}")
                require(metadata["helicity_domain"] == "general_polarization", f"unexpected helicity assumptions at {where}")
                family, origin = metadata["family_id"], metadata["compact_origin"]
                if origin not in origin_cache:
                    origin_poly = parse_laurent(origin)
                    computed = canonical_family(origin_poly, process, ignore_coefficients=True)
                    require(computed == family, f"family signature changed at {where}")
                    assigned = "test" if int(hashlib.sha256(f"{settings['split_seed']}/{family}".encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
                    require(assigned == split, f"family was assigned to wrong split at {where}")
                    check_process(origin_poly, process)
                    require(canonical_family(origin_poly, process) not in guard.skeleton_signatures[process], f"frozen structural family leaked at {where}")
                    values = guard.fingerprint(origin_poly, process)
                    require(not np.any(projective_matches(values, frozen_fingerprints[process])), f"frozen numerical family leaked at {where}")
                    if split == "test":
                        for mapping in species_mappings(process):
                            test_fingerprints[process].append(guard.fingerprint(relabel_laurent(origin_poly, mapping), process))
                    else:
                        require(not np.any(projective_matches(values, test_fingerprints[process])), f"cross-split numerical family overlap at {where}")
                    origin_expanded = expand_laurent(origin_poly)
                    origin_reduced = {m: c for m, c in origin_expanded.items() if not any(a[0] in ("d", "ep") and a[1] == a[2] and n > 0 for a, n in m)}
                    origin_cache[origin] = (process, family, split, origin_poly, origin_reduced)
                    actual_families = []
                    for monomial in origin_poly:
                        fs = Counter({"X": 0, "T": 0, "Q": 0})
                        for atom, power in monomial:
                            if atom[0] in fs:
                                fs[atom[0]] += power
                            if atom[0] == "d" and power < 0:
                                require(-power <= 4, f"compact denominator multiplicity exceeds four at {where}")
                        shape = (fs["X"], fs["T"], fs["Q"])
                        family_name = {(4, 0, 0): "XXXX", (2, 1, 0): "TXX", (0, 2, 0): "TT", (2, 0, 1): "QXX", (0, 0, 2): "QQ", (0, 1, 1): "TQ", (2, 0, 0): "XX"}.get(shape)
                        require(family_name is not None, f"unsupported compact family at {where}: {shape}")
                        actual_families.append(family_name)
                    require(Counter(actual_families) == Counter(json.loads(metadata["contraction_families"])), f"compact family metadata differs at {where}")
                    require(Counter(map(str, origin_poly.values())) == Counter(json.loads(metadata["coefficients"])), f"exact coefficient metadata differs at {where}")
                    if category in ("legacy_single_f", "4s1h_replay"):
                        require(all(c in (-1, 1) for c in origin_poly.values()), f"legacy/replay coefficient policy differs at {where}")
                    else:
                        require(any(c not in (-1, 1) for c in origin_poly.values()), f"enriched category lost coefficients at {where}")
                    has_repeated = any(a[0] == "d" and n < -1 for m in origin_poly for a, n in m)
                    if category in ("legacy_single_f", "rational_distinct", "4s1h_replay"):
                        require(not has_repeated, f"distinct-pole category has repeated poles at {where}")
                    if category == "rational_repeated":
                        require(has_repeated, f"repeated-pole category lost repetitions at {where}")
                    if category == "mixed_family":
                        require(len(set(actual_families)) > 1, f"mixed category lost family mixture at {where}")
                require(origin_cache[origin][:3] == (process, family, split), f"origin changes identity at {where}")
                require(family not in split_families[other], f"cross-split family overlap at {where}")
                split_families[split].add(family)
                family_counts[family] += 1
                require(family_counts[family] <= (1 if split == "test" else 5), f"origin variant cap exceeded at {where}")
                assignment = assignments.get(family)
                require(assignment is not None and assignment["split"] == split and assignment["category"] == category
                        and assignment["process"] == process and int(metadata["generation_seed"]) == assignment["origin_seed"], f"family assignment metadata mismatch at {where}")
                parsed, actual_ids = {}, {}
                for column in ("simple", "scrambled"):
                    require(raw[column] == metadata[column], f"raw/metadata text mismatch at {where}/{column}")
                    ids = json.loads(token_row[column])
                    require(all(type(i) is int for i in ids), f"noninteger token at {where}/{column}")
                    require(ids == tokenizer.encode_infix(raw[column]), f"raw/token mismatch at {where}/{column}")
                    require(len(ids) + 2 <= 5000 and len(ids) <= settings["max_tokens"], f"token capacity exceeded at {where}/{column}")
                    require(len(ids) == int(metadata[f"{column}_tokens"]), f"length metadata mismatch at {where}/{column}")
                    decoded = tokenizer.decode_infix(ids)
                    parsed[column] = parse_laurent(raw[column])
                    require(parsed[column] == parse_laurent(decoded), f"token semantic mismatch at {where}/{column}")
                    check_process(parsed[column], process)
                    exact_key = hashlib.sha256(repr(sorted(parsed[column].items())).encode()).hexdigest()
                    require(exact_key not in algebra_hashes[other], f"cross-split exact expression overlap at {where}")
                    algebra_hashes[split].add(exact_key)
                    token_key = hashlib.sha256(bytes(ids)).hexdigest()
                    require(token_key not in expression_hashes[other], f"cross-split token expression overlap at {where}")
                    expression_hashes[split].add(token_key)
                    if column == "scrambled":
                        require(token_key not in sources, f"duplicate serialized source at {where}")
                        sources.add(token_key)
                    actual_ids[column] = ids
                    lengths[split][column].append(len(ids))
                    stats = counts[split]
                    stats[f"{column}_digit_tokens"].update(tokenizer.id_to_token[i] for i in ids if tokenizer.id_to_token[i].endswith(":"))
                    stripped = re.sub(r"[peF]_\d+", "", raw[column])
                    stats[f"{column}_numeric_literals"].update(re.findall(r"\d+", stripped))
                    stats[f"{column}_rational_coefficients"].update(str(c) for c in parsed[column].values())
                    stats[f"{column}_reduced_pole_multiplicities"].update(str(-power) for m in parsed[column] for atom, power in m if atom[0] == "d" and power < 0)
                    stats[f"{column}_explicit_denominator_multiplicities"].update(explicit_pole_multiplicities(raw[column]))
                    ee = len(re.findall(r"e_4\s*·\s*e_5|e_5\s*·\s*e_4", raw[column]))
                    stats[f"{column}_polarization_contraction_rows"]["present" if ee else "absent"] += 1
                    stats[f"{column}_polarization_contraction_occurrences"]["e4_e5"] += ee
                    stats[f"{column}_contraction_atoms"].update(atom[0] for m in parsed[column] for atom, power in m if power > 0)
                require((parsed["simple"] == origin_cache[origin][3]) == (metadata["stage"] == "direct"), f"learning-stage classification differs at {where}")
                inferred_tags = set()
                if any(c not in (1, -1) for c in origin_cache[origin][3].values()):
                    inferred_tags.add("coefficient")
                if any(a[0] in ("T", "Q") for m in origin_cache[origin][3] for a, n in m):
                    inferred_tags.add("trace_or_mixed")
                if re.search(r"e_4\s*·\s*e_5|e_5\s*·\s*e_4", raw["scrambled"]):
                    inferred_tags.add("polarization_contraction")
                if any(a[0] == "d" and n < -1 for m in origin_cache[origin][3] for a, n in m):
                    inferred_tags.add("repeated_pole")
                if {"coefficient", "polarization_contraction", "repeated_pole"} <= inferred_tags:
                    inferred_tags.add("combined_features")
                require(inferred_tags == set(json.loads(metadata["feature_tags"])), f"feature tags differ from actual text at {where}")
                require(len(actual_ids["simple"]) + 8 <= len(actual_ids["scrambled"]), f"target lacks meaningful reduction at {where}")
                mode = json.loads(metadata["transformation_path"])[-1]
                expanded = {c: expand_laurent(p) for c, p in parsed.items()}
                # Prune masslessness/transversality first; eliminate momentum
                # variables only when required by a conservation transformation.
                reduced = {c: {m: v for m, v in poly.items() if not any(a[0] in ("d", "ep") and a[1] == a[2] and n > 0 for a, n in m)}
                           for c, poly in expanded.items()}
                require(reduced["simple"] == origin_cache[origin][4], f"target does not match its stated compact origin at {where}")
                exact_ok = reduced["simple"] == reduced["scrambled"]
                if not exact_ok:
                    exact_ok = canonical_onshell(reduced["simple"]) == canonical_onshell(reduced["scrambled"])
                require(exact_ok, f"exact pair identity failed at {where}")
                numerical = validator.validate(parsed["simple"], parsed["scrambled"], process)
                require(numerical["ok"], f"independent numerical gate failed at {where}: {numerical}")
                worst_error = max(worst_error, numerical["worst_relative_error"])
                resamples[str(numerical["resamples"])] += 1
                stored_validation = json.loads(metadata["validation"])
                require(stored_validation["ok"] and stored_validation["exact"] and stored_validation["token_roundtrip"], f"generation validation missing at {where}")
                for field in ("process", "assigned_category", "stage"):
                    counts[split][field][metadata[field]] += 1
                counts[split]["stages_by_process"][f"{process}/{metadata['stage']}"] += 1
                counts[split]["transformation"][mode] += 1
                counts[split]["feature_tags"].update(json.loads(metadata["feature_tags"]))
                counts[split]["origin_contraction_families"].update(json.loads(metadata["contraction_families"]))
                counts[split]["origin_coefficients"].update(json.loads(metadata["coefficients"]))
                audited += 1
                if audited % 1000 == 0:
                    print(json.dumps({"audited": audited, "elapsed_seconds": round(time.time() - started, 1)}), flush=True)
        requested = manifest["total_train_rows" if split == "train" else "total_test_rows"]
        require(file_records[f"{split}_raw"]["rows"] == requested, f"wrong final {split} row count")
        expected_quotas = {c: requested // (2 if c == "4s1h_replay" else 10) for c in CATEGORIES}
        require(dict(counts[split]["assigned_category"]) == expected_quotas == manifest["requested_counts"][split] == manifest["accepted_counts"][split], f"accepted category quotas differ for {split}")
        require(counts[split]["process"] == {"3s2h": requested // 2, "4s1h": requested // 2}, f"process quotas differ for {split}")
        for process in PROCESS_SPECS:
            relevant = [q for c, q in expected_quotas.items() if (c == "4s1h_replay") == (process == "4s1h")]
            fraction = Fraction(manifest.get("direct_fraction", settings.get("direct_fraction", "3/4")))
            require(0 <= fraction <= 1, "invalid direct-fraction setting")
            direct = sum((q*fraction.numerator + fraction.denominator-1)//fraction.denominator for q in relevant)
            intermediate = sum(relevant)-direct
            require(counts[split]["stages_by_process"][f"{process}/direct"] == direct
                    and counts[split]["stages_by_process"][f"{process}/intermediate"] == intermediate, f"stage mix differs for {split}/{process}")
            if requested in (100000, 200) and fraction == Fraction(3, 4):
                require(direct == 3 * requested // 8 and intermediate == requested // 8, "release 75/25 stage quota failed")
        require(counts[split]["simple_digit_tokens"] and counts[split]["scrambled_digit_tokens"], f"numeric literals lost in {split}")
        require(counts[split]["scrambled_polarization_contraction_rows"]["present"] > 0, f"polarization contractions lost in {split}")
        require(all(counts[split]["origin_contraction_families"][name] for name in ("XXXX", "TXX", "TT", "QXX", "QQ", "TQ", "XX")), f"compact family coverage missing in {split}")
        require(all(counts[split]["simple_reduced_pole_multiplicities"][str(n)] for n in (1, 2, 3, 4)), f"target pole multiplicity coverage missing in {split}")
        for key in ("simple", "scrambled"):
            require(all(counts[split][f"{key}_numeric_literals"][n] for n in ("2", "3", "4", "6")), f"required serialized numeric literals missing in {split}/{key}")
    require(set(assignments) == set(family_counts), "unused or missing family assignments")
    require(all(assignments[f]["variants"] == n for f, n in family_counts.items()), "family variant counts changed")
    report = {"status": "passed", "audit_seed": audit_seed, "fingerprint_seed": audit_seed + 37,
              "independent_momentum_samples": 3, "evaluations_per_pair": 15, "general_transverse_samples_per_pair": 3,
              "numerical_tolerances": {"rtol": validator.rtol, "atol": validator.atol}, "worst_relative_error": worst_error,
              "resample_counts": dict(resamples), "total_rows": audited, "files": file_records,
              "cross_split_family_overlap": 0, "cross_split_expression_overlap": 0, "cross_split_numerical_family_overlap": 0,
              "frozen_family_leaks": 0, "original_artifacts_preserved": original_records,
              "unique_origins": {s: len(split_families[s]) for s in split_families},
              "maximum_variants_per_origin": max(family_counts.values()), "family_variant_histogram": dict(Counter(family_counts.values())),
              "coverage": {s: {k: dict(v) for k, v in stats.items()} for s, stats in counts.items()},
              "lengths": {s: {c: {"min": min(v), "max": max(v), "mean": float(np.mean(v)),
                                           "percentiles": dict(zip(("p10", "p50", "p90", "p99"), map(float, np.percentile(v, (10, 50, 90, 99)))))}
                              for c, v in columns.items()} for s, columns in lengths.items()},
              "elapsed_seconds": time.time() - started,
              "holdout_fingerprint_domains": ["general_and_positive", "positive_helicity_only_conservative"],
              "fingerprint_limit": "Finite numerical fingerprints are complementary checks, not proof of arbitrary algebraic inequivalence."}
    (directory / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "rows": audited, "report": str(directory / 'audit.json'), "seconds": report["elapsed_seconds"]}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--audit-seed", type=int, default=830017)
    args = parser.parse_args(argv)
    audit(args.release_dir, audit_seed=args.audit_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
