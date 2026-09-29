"""Portable entry point for the verified two-call five-point gravity workflow."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import platform
import time

from .common import read_source, require, sha256, write_json


def run(source: Path, output_dir: Path, *, checkpoint: Path | None = None,
        prepare_only: bool = False, threads: int = 4, max_length: int = 256):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    checkpoint = Path(checkpoint).resolve() if checkpoint is not None else None
    require(threads > 0 and max_length >= 2, "threads must be positive; max-length must be at least 2.")
    require(source.is_file(), f"Source CSV not found: {source}")
    read_source(source)
    if not prepare_only:
        require(checkpoint is not None and checkpoint.is_file(), "A valid --checkpoint is required for inference.")
    require(not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())),
            f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source), "source_sha256": sha256(source),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_sha256": None, "prepare_only": prepare_only,
                "device": "cpu", "threads": threads, "max_length": max_length,
                "python": platform.python_version(), "neural_calls": 0,
                "workflow": "Guided scalar preprocessing, two selected greedy calls, exact weighted reconstruction.",
                "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(output_dir / "run_manifest.json", manifest)
    try:
        import sympy
        from . import derive_generic_groups, grouped_model, serialize_groups, verify_final
        manifest["sympy"] = sympy.__version__
        print("Deriving scalar groups from the source CSV...", flush=True)
        derive_generic_groups.main(source, output_dir)
        grouped_model.main(source, output_dir)
        if prepare_only:
            serialize_groups.prepare_inputs(output_dir)
            manifest["status"] = "prepared"
        else:
            import torch
            manifest["torch"] = torch.__version__
            manifest["checkpoint_sha256"] = sha256(checkpoint)
            print("Running the two selected greedy model calls...", flush=True)
            serialize_groups.main(source, checkpoint, output_dir, threads=threads, max_length=max_length)
            print("Verifying the reconstructed amplitudes and compact serializations...", flush=True)
            final = verify_final.main(source, output_dir)
            manifest["result_tokens"] = {name: result["tokens"] for name, result in final["results"].items()}
            manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        inference = output_dir / "serialization_results.json"
        if inference.exists():
            manifest["neural_calls"] = len(json.loads(inference.read_text()))
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir),
                      "neural_calls": manifest["neural_calls"],
                      "result_tokens": manifest.get("result_tokens", {})}), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Single-row, headerless id,expression CSV (or .csv.gz).")
    parser.add_argument("--checkpoint", type=Path, help="Trusted local best_model.pt; required unless --prepare-only.")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new or empty directory for evidence and final expressions.")
    parser.add_argument("--prepare-only", action="store_true", help="Derive and validate scalar inputs without loading a checkpoint.")
    parser.add_argument("--threads", type=int, default=4, help="CPU PyTorch threads (default: 4).")
    parser.add_argument("--max-length", type=int, default=256, help="Greedy decoding ceiling including special tokens (default: 256).")
    args = parser.parse_args(argv)
    try:
        run(args.source, args.output_dir, checkpoint=args.checkpoint, prepare_only=args.prepare_only,
            threads=args.threads, max_length=args.max_length)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Workflow failed: {exc}\n")


if __name__ == "__main__":
    main()
