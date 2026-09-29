"""The original checkpoint and unmasked greedy decoder, on CPU."""
from pathlib import Path
import time

import torch

from data_gen.Tokenizer import ScatteringAmplitudeTokenizer
from transformer.transformer_functions import TransformerRegressor, decode_with_model


def completed_tokens(raw):
    """Accept only an actually emitted EOS, never a truncated expression."""
    if not raw or raw[0] != 2 or 3 not in raw[1:]:
        raise ValueError("Greedy prediction is missing BOS or emitted EOS")
    stop = raw.index(3)
    tokens = raw[1:stop]
    if not tokens or any(token in (0, 1, 2, 3) for token in tokens):
        raise ValueError("Greedy prediction contains empty or special-token content")
    if any(token != 0 for token in raw[stop + 1:]):
        raise ValueError("Unexpected content after EOS")
    return tokens


class FivePointPredictor:
    def __init__(self, checkpoint: Path, threads=2):
        if threads < 1:
            raise ValueError("CPU thread count must be positive")
        torch.set_num_threads(threads)
        self.tokenizer = ScatteringAmplitudeTokenizer(
            max_particles=8, max_sequence_length=None
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_args = dict(saved["model_args"], device="cpu")
        self.model = TransformerRegressor(**model_args).to("cpu")
        self.model.load_state_dict(saved["model_state_dict"])
        self.model.eval()
        self.epoch = saved["epoch"]

    def predict(self, source, max_length=256):
        ids = self.tokenizer.encode_infix(source)
        if any(token in (0, 1, 2, 3) or token >= self.model.vocab_size for token in ids):
            raise ValueError("Input contains unsupported or special tokens")
        if len(ids) + 2 > self.model.max_seq_len:
            raise ValueError("Input exceeds the checkpoint's positional capacity")
        if not 2 <= max_length <= self.model.max_seq_len:
            raise ValueError("Decoder length must fit the checkpoint's positional capacity")
        source_tensor = torch.tensor([[2, *ids, 3]], dtype=torch.long)
        started = time.monotonic()
        with torch.inference_mode():
            generated, _ = decode_with_model(
                self.model, source_tensor, max_length, "greedy", beam_size=1
            )
        # Greedy generation returns the actual emitted IDs, unlike historical
        # beam output tensors which could append an artificial EOS.
        raw = generated[0].tolist()
        tokens = completed_tokens(raw)
        return {
            "input_tokens": len(ids),
            "raw_token_ids": raw,
            "prediction_tokens": tokens,
            "prediction": self.tokenizer.decode_infix(tokens),
            "eos_generated": True,
            "inference_seconds": time.monotonic() - started,
        }
