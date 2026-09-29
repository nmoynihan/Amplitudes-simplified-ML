#!/usr/bin/env python3
"""Reproducible family partitions, checkpoint fitting, and gravity v2 evaluation.

The published 100,000/200 CSVs are immutable inputs. Validation is a saved list
of training-pool indices, never an additional dataset or the release test set.
Training only starts through the explicit ``train`` subcommand.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "transformer"))
DEFAULT_RELEASE = WORKSPACE / "gravity-data-v2"
ORIGINAL = Path("/Users/kymani/Documents/Big_YM_Data_Sets/Cyclic_Gravity")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def tokenizer():
    from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
    return ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)


def tokenizer_identity():
    tok = tokenizer()
    definition = json.dumps(tok.vocab, sort_keys=True, separators=(",", ":"))
    return {"vocabulary_size": tok.vocab_size,
            "vocabulary_sha256": hashlib.sha256(definition.encode()).hexdigest(),
            "definition_file_sha256": sha256(PROJECT / "data_gen/Tokenizer.py"),
            "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}}


def make_partition(metadata_rows, *, seed=20260923, validation_fraction=0.1):
    """Stratify families, choosing a deterministic nearest row-count prefix.

    Family membership overrides strata. A family spanning multiple categories is
    assigned to its sorted joint stratum, so it can never cross fitting/validation.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    families = defaultdict(list)
    for i, row in enumerate(metadata_rows):
        if not row.get("family_id"):
            raise ValueError(f"Metadata row {i + 1} has no family_id")
        families[row["family_id"]].append(i)
    strata = defaultdict(list)
    for family, indices in families.items():
        stratum = tuple(sorted({(metadata_rows[i]["process"],
                                 metadata_rows[i].get("assigned_category", "unknown"))
                                for i in indices}))
        strata[stratum].append(family)
    validation_families = set()
    for stratum, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda fid: hashlib.sha256(
            f"{seed}:{fid}".encode()).digest())
        wanted = sum(len(families[fid]) for fid in ordered) * validation_fraction
        cumulative = [0]
        for fid in ordered:
            cumulative.append(cumulative[-1] + len(families[fid]))
        # Reserve both fitting and validation families when the stratum permits.
        candidates = range(1, len(ordered)) if len(ordered) > 1 else range(1)
        count = min(candidates, key=lambda i: (abs(cumulative[i] - wanted), i))
        validation_families.update(ordered[:count])
    fitting = [i for i, row in enumerate(metadata_rows)
               if row["family_id"] not in validation_families]
    validation = [i for i, row in enumerate(metadata_rows)
                  if row["family_id"] in validation_families]
    if not fitting or not validation:
        raise ValueError("Need at least two independent families for a usable split")
    return {"schema_version": "gravity-v2-experiment-1", "seed": seed,
            "requested_validation_fraction": validation_fraction,
            "index_base": 0, "fitting_indices": fitting,
            "validation_indices": validation,
            "fitting_rows": len(fitting), "validation_rows": len(validation),
            "fitting_families": len(families) - len(validation_families),
            "validation_families": len(validation_families),
            "family_overlap": 0,
            "validation_family_ids": sorted(validation_families),
            "effective_validation_fraction": len(validation) / len(metadata_rows)}


def partition_command(args):
    release = Path(args.release)
    metadata = release / "gravity_v2_train_metadata.csv.gz"
    token_file = release / "gravity_v2_train_tok.csv.gz"
    rows = list(read_csv(metadata))
    if len(rows) != 100_000:
        raise ValueError(f"Expected the complete 100000-row training pool, got {len(rows)}")
    result = make_partition(rows, seed=args.seed,
                            validation_fraction=args.validation_fraction)
    result.update(metadata_path=str(metadata.resolve()),
                  metadata_sha256=sha256(metadata), token_file_sha256=sha256(token_file),
                  tokenizer=tokenizer_identity(),
                  test_usage="No test data read or used for model selection")
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("fitting_rows", "validation_rows",
                                           "fitting_families", "validation_families")}))


def load_checkpoint(path, device="cpu"):
    import torch
    from transformer_functions import TransformerRegressor
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_args = dict(checkpoint["model_args"], device=device)
    if model_args["vocab_size"] < tokenizer().vocab_size:
        raise ValueError("Checkpoint has fewer vocabulary slots than this tokenizer")
    model = TransformerRegressor(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def capacity_report(model_args, source_max, target_max, decoder_ceiling):
    capacity = int(model_args.get("max_seq_len", 5000))
    if max(source_max, target_max) > 4096:
        raise ValueError("Release exceeds the 4096-content-token limit")
    if max(source_max, target_max) + 2 > capacity:
        raise ValueError("Content plus BOS/EOS exceeds checkpoint positional capacity")
    if decoder_ceiling < target_max + 2:
        raise ValueError("Decoder ceiling must cover every target plus BOS/EOS")
    if decoder_ceiling > capacity:
        raise ValueError("Decoder ceiling exceeds checkpoint positional capacity")
    return {"source_content_max": source_max, "target_content_max": target_max,
            "source_with_bos_eos_max": source_max + 2,
            "target_with_bos_eos_max": target_max + 2,
            "checkpoint_positions": capacity, "decoder_ceiling": decoder_ceiling,
            "checkpoint_vocabulary_slots": model_args["vocab_size"],
            "tokenizer_vocabulary_size": tokenizer().vocab_size,
            "token_ids_preserved": True, "passes": True}


def train_command(args):
    """Fresh optimizer fine-tuning (or training from the same architecture)."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from data_import import TransformerDataset, BucketBatchSampler, dynamic_pad_collate
    from transformer_functions import train_model
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    release = Path(args.release)
    partition = json.loads(Path(args.partition).read_text())
    tok_path = release / "gravity_v2_train_tok.csv.gz"
    meta_path = release / "gravity_v2_train_metadata.csv.gz"
    if sha256(tok_path) != partition["token_file_sha256"] or sha256(meta_path) != partition["metadata_sha256"]:
        raise ValueError("Training files differ from the frozen partition")
    if tokenizer_identity()["vocabulary_sha256"] != partition["tokenizer"]["vocabulary_sha256"]:
        raise ValueError("Tokenizer vocabulary changed after partition creation")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if args.from_scratch:
        from transformer_functions import TransformerRegressor
        model = TransformerRegressor(**dict(checkpoint["model_args"], device=device)).to(device)
    dataset = TransformerDataset(str(tok_path), max_length=None, dynamic_padding=True)
    if len(dataset) != 100_000:
        raise ValueError("Expected 100000 training pool rows")
    source_max = max(map(len, dataset.scrambled_sequences)) - 2
    target_max = max(map(len, dataset.simple_sequences)) - 2
    capacity = capacity_report(checkpoint["model_args"], source_max, target_max,
                               args.max_output_tokens)
    valid_ids = set(tokenizer().id_to_token)
    for sequence in dataset.simple_sequences + dataset.scrambled_sequences:
        if any(token not in valid_ids or token in (0, 1, 2, 3) for token in sequence[1:-1]):
            raise ValueError("Invalid or special ID inside training content")
    fitting_indices, val_indices = partition["fitting_indices"], partition["validation_indices"]
    if set(fitting_indices) & set(val_indices) or sorted(fitting_indices + val_indices) != list(range(len(dataset))):
        raise ValueError("Partition is not a disjoint cover of the training pool")
    metadata = list(read_csv(meta_path))
    if {metadata[i]["family_id"] for i in fitting_indices} & {metadata[i]["family_id"] for i in val_indices}:
        raise ValueError("Family leakage in saved partition")
    fitting, validation = Subset(dataset, fitting_indices), Subset(dataset, val_indices)
    sampler = BucketBatchSampler([dataset.sequence_length(i) for i in fitting_indices],
                                 args.batch_size, seed=args.seed)
    train_loader = DataLoader(fitting, batch_sampler=sampler, collate_fn=dynamic_pad_collate,
                              num_workers=args.num_workers)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dynamic_pad_collate, num_workers=args.num_workers)
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"settings": vars(args), "capacity": capacity,
                  "experiment_code_sha256": sha256(__file__), "torch_version": torch.__version__,
                  "tokenizer": tokenizer_identity(), "initial_checkpoint_sha256": sha256(args.checkpoint),
                  "partition_sha256": sha256(args.partition),
                  "fitting_rows": len(fitting), "validation_rows": len(validation),
                  "optimizer": "fresh AdamW; original checkpoint optimizer is not resumed",
                  "test_used_for_selection": False, "dry_run": args.dry_run}
    write_json(run_dir / "experiment.json", provenance)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "fitting_rows": len(fitting),
                          "validation_rows": len(validation), "capacity": capacity}))
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                          eta_min=args.learning_rate * 0.01)
    # train_model joins models/<run_name>; an absolute run_name preserves the requested output dir.
    train_model(model, optimizer, criterion, train_loader, val_loader, epochs=args.epochs,
                run_name=str(run_dir), early_stopping_patience=args.patience,
                scheduler=scheduler, grad_clip=1.0,
                gradient_accumulation_steps=args.gradient_accumulation_steps)


def oracle(seed=151, samples=3):
    from data_testing import evaluate_model as ev
    ev.NUMERIC_BACKEND = "gravity"
    ev.N_PARTICLES = 5
    ev.NUMERIC_EQUIV_SEED = seed
    ev.NUMERIC_EQUIV_SAMPLES = samples
    return ev, ev.precompute_kinematics()


def compare_expressions(prediction, source, process, ev, points):
    try:
        ev.validate_gravity_expression(prediction, process)
        ev.validate_gravity_expression(source, process)
    except Exception as exc:
        return {"equivalent": False, "status": "parse_error", "error": f"{type(exc).__name__}: {exc}"}
    max_scaled = 0.0
    abs_tol, rel_tol = ev.resolve_numeric_tolerances()
    mismatch = False
    errors = []
    checked = 0
    queue = list(points[process])
    # Replace singular/nonfinite points deterministically, retaining tolerances.
    attempts = 0
    while checked < len(points[process]) and attempts < 4 * len(points[process]):
        if not queue:
            old_seed = ev.NUMERIC_EQUIV_SEED
            ev.NUMERIC_EQUIV_SEED = old_seed + 100_003 + attempts
            queue.extend(ev.precompute_kinematics()[process])
            ev.NUMERIC_EQUIV_SEED = old_seed
        point = queue.pop(0)
        attempts += 1
        try:
            left = ev.eval_numeric_expr(prediction, point, None, gravity_process=process)
            right = ev.eval_numeric_expr(source, point, None, gravity_process=process)
            if not all(math.isfinite(complex(value).real) and math.isfinite(complex(value).imag)
                       for value in (left, right)):
                raise ArithmeticError("Nonfinite kinematic value")
        except (ArithmeticError, FloatingPointError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        scale = max(abs(left), abs(right))
        difference = abs(left - right)
        max_scaled = max(max_scaled, difference / max(1.0, scale))
        mismatch |= not ev.numeric_values_close(left, right, tol_abs=abs_tol, tol_rel=rel_tol)
        checked += 1
    complete = checked == len(points[process])
    return {"equivalent": complete and not mismatch,
            "status": "singular_point_error" if not complete else "mismatch" if mismatch else "equivalent",
            "checks": checked, "singular_points_resampled": len(errors),
            "singular_errors": errors, "maximum_scaled_difference": max_scaled}


def score_tokens(raw_ids, source, process, tok, ev, points):
    ids = [int(i) for i in raw_ids]
    if ids and ids[0] == 2:
        ids = ids[1:]
    eos = 3 in ids
    content = ids[:ids.index(3)] if eos else ids
    result = {"raw_ids": raw_ids, "content_tokens": len(content), "eos": eos,
              "valid": False, "equivalent": False}
    if any(i in (0, 1, 2) or i not in tok.id_to_token for i in content):
        return result | {"status": "parse_error", "error": "Invalid/special content token"}
    try:
        expr = tok.decode_infix(content)
        ev.validate_gravity_expression(expr, process)
    except Exception as exc:
        return result | {"status": "parse_error", "error": f"{type(exc).__name__}: {exc}"}
    result.update(valid=True, expression=expr)
    result.update(compare_expressions(expr, source, process, ev, points))
    if not eos:
        result.update(equivalent=False, status="truncated_output")
    return result


def decode_case(model, case, *, ceiling, method, beam_size, device, ev, points):
    import torch
    from transformer_functions import decode_with_model
    tok = tokenizer()
    source_ids = tok.encode_infix(case["scrambled"])
    source = torch.tensor([[2] + source_ids + [3]], dtype=torch.long, device=device)
    start = time.monotonic()
    with torch.inference_mode():
        output, beams = decode_with_model(model, source, max_length=ceiling,
                                         decoding_method=method, beam_size=beam_size)
    top = score_tokens(output[0].tolist(), case["scrambled"], case["process"], tok, ev, points)
    candidates = [score_tokens(ids, case["scrambled"], case["process"], tok, ev, points)
                  for ids in beams[0]] if beams else [top]
    top.update(source_tokens=len(source_ids), source_reduction=1 - top["content_tokens"] / len(source_ids),
               target_tokens=len(tok.encode_infix(case["simple"])) if case.get("simple") else None)
    return {"name": case.get("name"), "process": case["process"], "top1": top,
            "candidates": candidates, "oracle_at_k": any(row["equivalent"] for row in candidates),
            "seconds": time.monotonic() - start}


def numerical_settings(ev, points):
    return {"seed": ev.NUMERIC_EQUIV_SEED, "samples": ev.NUMERIC_EQUIV_SAMPLES,
            "checks_per_process": {key: len(value) for key, value in points.items()},
            "reference_modes": list(ev.resolve_gravity_reference_modes()),
            "gauge_shift": ev.GRAVITY_GAUGE_SHIFT, "tolerances_abs_rel": list(ev.resolve_numeric_tolerances()),
            "helicity_domain": "same_positive_helicity",
            "limitation": "Numerical agreement in this sampler does not prove general-polarization equality"}


def frozen_baseline_command(args):
    import torch
    torch.set_num_threads(args.cpu_threads)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    ev, points = oracle(args.numeric_seed, args.numeric_samples)
    with Path(args.amplitude).open(newline="") as stream:
        amplitude = next(csv.reader(stream))[1].strip()
    tok = tokenizer()
    reference_dir = WORKSPACE / "gravity-5pt-diagnosis-2026-09-22/physics"
    references = {name: (reference_dir / f"{name}.txt").read_text().strip()
                  for name in ("compact_factored", "compact_same_helicity")}
    capacity = capacity_report(checkpoint["model_args"], len(tok.encode_infix(amplitude)),
                               max(len(tok.encode_infix(expr)) for expr in references.values()),
                               args.max_output_tokens)
    checkpoint_role = ("original_checkpoint" if Path(args.checkpoint).resolve() ==
                       (ORIGINAL / "Model/best_model.pt").resolve() else "supplied_checkpoint")
    report = {"status": "measured_frozen_amplitude", "checkpoint_role": checkpoint_role,
              "checkpoint": str(Path(args.checkpoint).resolve()),
              "torch_version": torch.__version__, "experiment_code_sha256": sha256(__file__),
              "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
              "amplitude_sha256": sha256(args.amplitude), "tokenizer": tokenizer_identity(),
              "capacity": capacity, "numerical_settings": numerical_settings(ev, points),
              "decode": {"max_length": args.max_output_tokens, "beam_size": args.beam_size,
                         "device": args.device, "cpu_threads": args.cpu_threads},
              "references": {name: compare_expressions(expr, amplitude, "3s2h", ev, points)
                             for name, expr in references.items()}, "inference": []}
    for method in ("greedy", "beam"):
        result = decode_case(model, {"name": "frozen_user_amplitude", "process": "3s2h", "scrambled": amplitude},
                             ceiling=args.max_output_tokens, method=method, beam_size=args.beam_size,
                             device=args.device, ev=ev, points=points)
        report["inference"].append(dict(method=method, **result))
        write_json(args.output, report)
        print(json.dumps({"method": method, "valid": result["top1"]["valid"],
                          "equivalent": result["top1"]["equivalent"],
                          "oracle_at_k": result["oracle_at_k"], "seconds": result["seconds"]}), flush=True)


def summarise(results):
    count = len(results)
    if not count:
        return {"count": 0}
    return {"count": count,
            "valid_fraction": sum(row["top1"]["valid"] for row in results) / count,
            "numerical_equivalence_fraction": sum(row["top1"]["equivalent"] for row in results) / count,
            "oracle_at_k_fraction": sum(row["oracle_at_k"] for row in results) / count,
            "mean_output_content_tokens": sum(row["top1"]["content_tokens"] for row in results) / count,
            "mean_source_reduction": sum(row["top1"]["source_reduction"] for row in results) / count,
            "equivalent_and_shorter_fraction": sum(row["top1"]["equivalent"] and row["top1"]["source_reduction"] > 0 for row in results) / count,
            "statuses": dict(Counter(row["top1"]["status"] for row in results))}


def evaluate_command(args):
    import torch
    torch.set_num_threads(args.cpu_threads)
    raw = list(read_csv(args.raw))
    metadata = list(read_csv(args.metadata))
    if not raw or len(raw) != len(metadata):
        raise ValueError("Raw and metadata files must have the same nonzero row count")
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    tok = tokenizer()
    capacity = capacity_report(checkpoint["model_args"], max(len(tok.encode_infix(r["scrambled"])) for r in raw),
                               max(len(tok.encode_infix(r["simple"])) for r in raw), args.max_output_tokens)
    ev, points = oracle(args.numeric_seed, args.numeric_samples)
    results = []
    strata = defaultdict(list)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    selected_count = min(len(raw), args.limit) if args.limit is not None else len(raw)
    report = {"status": "incomplete", "file_rows": len(raw), "selected_rows": selected_count,
              "torch_version": torch.__version__, "experiment_code_sha256": sha256(__file__),
              "selection": "all" if args.limit is None else f"first {selected_count} rows; diagnostic only",
              "checkpoint_sha256": sha256(args.checkpoint),
              "raw_sha256": sha256(args.raw), "metadata_sha256": sha256(args.metadata),
              "tokenizer": tokenizer_identity(), "capacity": capacity,
              "decode": {"method": args.method, "beam_size": args.beam_size,
                         "max_length": args.max_output_tokens, "device": args.device},
              "numerical_settings": numerical_settings(ev, points), "results": results}
    for i, (row, meta) in enumerate(zip(raw, metadata, strict=True)):
        if i >= selected_count:
            break
        for column in ("simple", "scrambled"):
            if meta.get(column) and meta[column] != row[column]:
                raise ValueError(f"Metadata alignment mismatch at row {i + 1}: {column}")
        result = decode_case(model, dict(row, process=meta["process"], name=f"row_{i + 1}"),
                             ceiling=args.max_output_tokens, method=args.method, beam_size=args.beam_size,
                             device=args.device, ev=ev, points=points)
        result.update(row_1based=i + 1, family_id=meta.get("family_id"),
                      assigned_category=meta.get("assigned_category"))
        results.append(result)
        keys = ["all", "process/" + meta["process"], "category/" + meta.get("assigned_category", "legacy")]
        try:
            features = json.loads(meta.get("feature_tags", "[]"))
        except json.JSONDecodeError:
            features = [x for x in re.split(r"[|,;]", meta.get("feature_tags", "")) if x]
        if isinstance(features, dict):
            features = [key for key, value in features.items() if value]
        keys.extend("feature/" + feature for feature in features)
        feature_set = set(features)
        for combination in (("coefficient", "repeated_pole"),
                            ("coefficient", "polarization_contraction"),
                            ("polarization_contraction", "repeated_pole"),
                            ("coefficient", "polarization_contraction", "repeated_pole")):
            if set(combination) <= feature_set:
                keys.append("feature_combination/" + "+".join(combination))
        numeric = any(11 <= token <= 20 for token in tok.encode_infix(row["scrambled"]))
        ee = bool(re.search(r"e_[45]\s*·\s*e_[45]", row["scrambled"]))
        if numeric:
            keys.append("measured_source/numeric_literals")
        if ee:
            keys.append("measured_source/polarization_contraction")
        if numeric and ee:
            keys.append("measured_source/numeric_and_polarization")
        for key in set(keys):
            strata[key].append(result)
        report["summary"] = {key: summarise(value) for key, value in sorted(strata.items())}
        report["completed_rows"] = len(results)
        write_json(args.output, report)
        print(json.dumps({"row": i + 1, "total": len(raw), "status": result["top1"]["status"]}), flush=True)
    report["status"] = "measured_complete" if selected_count == len(raw) else "measured_diagnostic_subset"
    write_json(args.output, report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    partition = sub.add_parser("partition", help="Write training-family fitting/validation indices only")
    partition.add_argument("--release", default=str(DEFAULT_RELEASE))
    partition.add_argument("--output", required=True)
    partition.add_argument("--seed", type=int, default=20260923)
    partition.add_argument("--validation-fraction", type=float, default=0.1)
    partition.set_defaults(func=partition_command)
    train = sub.add_parser("train", help="Fine-tune with family validation and no truncation")
    train.add_argument("--release", default=str(DEFAULT_RELEASE))
    train.add_argument("--partition", required=True)
    train.add_argument("--checkpoint", default=str(ORIGINAL / "Model/best_model.pt"))
    train.add_argument("--output-dir", required=True)
    train.add_argument("--from-scratch", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--seed", type=int, default=20260923)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--label-smoothing", type=float, default=0.1)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--gradient-accumulation-steps", type=int, default=2)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--cpu-threads", type=int, default=4)
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    train.add_argument("--max-output-tokens", type=int, default=4098)
    train.set_defaults(func=train_command)
    for name, function in (("evaluate", evaluate_command), ("frozen-baseline", frozen_baseline_command)):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", default=str(ORIGINAL / "Model/best_model.pt"))
        command.add_argument("--output", required=True)
        command.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
        command.add_argument("--cpu-threads", type=int, default=4)
        command.add_argument("--max-output-tokens", type=int, default=4098)
        command.add_argument("--beam-size", type=int, default=20)
        command.add_argument("--numeric-seed", type=int, default=151)
        command.add_argument("--numeric-samples", type=int, default=3)
        if name == "evaluate":
            command.add_argument("--raw", required=True)
            command.add_argument("--metadata", required=True)
            command.add_argument("--method", choices=["greedy", "beam"], default="greedy")
            command.add_argument("--limit", type=int, help="Diagnostic first-row subset only; omit for final metrics")
        else:
            command.add_argument("--amplitude", default=str(ORIGINAL / "Test_Amplitude/gravity5unified12345_seed.csv"))
        command.set_defaults(func=function)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    function = args.func
    del args.func
    function(args)


if __name__ == "__main__":
    main()
