"""Portable orchestration; successful results are written only after verification."""
from pathlib import Path
import csv
import json
import platform
import time

import sympy as sp

from .inference import sha256


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def output_directory(path: Path) -> Path:
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def execute(mode: str, output_dir: Path, *, checkpoint=None, source=None,
            seed=None, reference=None, derive_basis=False, derivation=None,
            threads=4, max_length=160) -> dict:
    if mode not in {"run", "verify", "prepare", "derive", "audit"}:
        raise ValueError(f"Unknown workflow mode: {mode}")
    if mode == "run" and (checkpoint is None or not Path(checkpoint).is_file()):
        raise ValueError("The run command requires an existing --checkpoint file")
    if mode == "audit" and (source is None or not Path(source).is_file()):
        raise ValueError("The audit command requires an existing --input CSV")
    if derive_basis and derivation is not None:
        raise ValueError("Choose either --derive or --derivation")
    output_dir = output_directory(output_dir)
    started = time.monotonic()
    status = {"mode": mode, "status": "running", "model_calls": 0,
              "python_version": platform.python_version(), "sympy_version": sp.__version__}
    write_json(output_dir / "workflow.json", status)
    try:
        if mode == "audit":
            from .audit import run_all
            result = {"audit": run_all(Path(source).resolve(), output_dir / "audit")}
        elif mode == "derive":
            from .derive import derive
            result = {"derivation": derive(output_dir, seed=seed, reference=reference)}
        else:
            from .algebra import DEFAULT_REFERENCE, read_expression
            from .prepare import prepare
            if derive_basis:
                from .derive import derive
                derive(output_dir / "derivation", seed=seed, reference=reference)
                derivation = output_dir / "derivation" / "sparse_extended_result.json"
            preparation = prepare(output_dir, seed=seed, reference=reference,
                                  derivation=derivation)
            source_path = output_dir / "gluon4feyn1234_model_ready.csv"
            result = {"preparation": preparation,
                      "source_sha256": sha256(source_path)}
            if mode in {"verify", "run"}:
                from .audit import run_all
                result["audit"] = run_all(source_path, output_dir / "audit")
            if mode == "run":
                from .inference import predict
                from .verification import verify_prediction
                source_expression = read_expression(source_path)

                def record_attempt(evidence):
                    status["model_calls"] = evidence["model_calls"]
                    write_json(output_dir / "workflow.json", status)
                    write_json(output_dir / "inference_attempt.json", evidence)

                prediction = predict(source_expression, checkpoint,
                                     threads=threads, max_length=max_length,
                                     record_attempt=record_attempt)
                write_json(output_dir / "inference_attempt.json", prediction)
                verification = verify_prediction(
                    source_expression, prediction["prediction"],
                    read_expression(reference or DEFAULT_REFERENCE),
                )
                rendered = verification.pop("rendered_prediction")
                with (output_dir / "gluon4feyn1234_completed_simplified.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    csv.writer(stream).writerow([1, rendered])
                result.update(prediction)
                result.update(verification)
        result.update(status, status="passed", seconds=time.monotonic() - started)
        write_json(output_dir / "results.json", result)
        status.update(status="passed", seconds=result["seconds"])
        write_json(output_dir / "workflow.json", status)
        return result
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      seconds=time.monotonic() - started)
        write_json(output_dir / "workflow.json", status)
        raise
