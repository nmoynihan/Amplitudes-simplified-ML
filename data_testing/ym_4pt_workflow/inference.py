"""One actual greedy call to the unchanged four-point checkpoint, on CPU."""
from pathlib import Path
import hashlib
import time


HISTORICAL_CHECKPOINT_SHA256 = (
    "393a8a6c425bb96f0bcb8a2a1efca24641103ab92203083e8c5c62e5d66a2478"
)


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def completed_tokens(raw):
    """Only accept a prediction that actually emitted EOS."""
    if not raw or raw[0] != 2 or 3 not in raw[1:]:
        raise ValueError("Greedy prediction is missing BOS or emitted EOS")
    stop = raw.index(3)
    tokens = raw[1:stop]
    if not tokens or any(token in (0, 1, 2, 3) for token in tokens):
        raise ValueError("Greedy prediction contains empty or special-token content")
    if any(token != 0 for token in raw[stop + 1:]):
        raise ValueError("Unexpected content after EOS")
    return tokens


def predict(source: str, checkpoint: Path, *, threads=4, max_length=160,
            record_attempt=None) -> dict:
    # Importing the workflow or running the independent audit needs no PyTorch.
    import torch
    from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
    from transformer.transformer_functions import TransformerRegressor

    if threads < 1:
        raise ValueError("CPU thread count must be positive")
    checkpoint = Path(checkpoint).resolve(strict=True)
    torch.set_num_threads(threads)
    tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_args = dict(saved["model_args"], device="cpu")
    model = TransformerRegressor(**model_args).to("cpu")
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    ids = tokenizer.encode_infix(source)
    if any(token in (0, 1, 2, 3) or token >= model.vocab_size for token in ids):
        raise ValueError("Input contains unsupported or special tokens")
    if len(ids) + 2 > model.max_seq_len:
        raise ValueError("Input exceeds the checkpoint's positional capacity")
    if not 2 <= max_length <= model.max_seq_len:
        raise ValueError("Decoder length must fit the checkpoint's positional capacity")
    checkpoint_hash = sha256(checkpoint)
    evidence = {
        "method": "derived equivalent p/e input; one greedy model call",
        "model_calls": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "matches_historical_checkpoint": checkpoint_hash == HISTORICAL_CHECKPOINT_SHA256,
        "checkpoint_epoch": saved.get("epoch"),
        "torch_version": torch.__version__,
        "device": "cpu", "threads": threads, "max_length_including_bos": max_length,
        "input_tokens": len(ids), "source": source, "status": "decoding",
    }
    if record_attempt is not None:
        record_attempt(evidence)
    started = time.monotonic()
    with torch.inference_mode():
        raw = model.generate(
            torch.tensor([[2, *ids, 3]], dtype=torch.long), max_length=max_length
        )[0].tolist()
    evidence.update(raw_token_ids=raw, status="decoded",
                    inference_seconds=time.monotonic() - started)
    # Preserve actual emitted IDs even when a truncated/invalid result fails.
    if record_attempt is not None:
        record_attempt(evidence)
    tokens = completed_tokens(raw)
    prediction = tokenizer.decode_infix(tokens)
    return {
        **evidence, "output_tokens": len(tokens), "prediction_tokens": tokens,
        "eos_generated": True, "prediction": prediction,
    }
