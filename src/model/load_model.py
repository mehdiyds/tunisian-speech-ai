"""Load the fixed Whisper baseline without changing its tokenizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_ID = "oddadmix/Whisperv3-tunisian-codeswitch"
LANGUAGE = "ar"
TASK = "transcribe"


def _build_fast_processor(feature_extractor: Any, tokenizer: Any) -> Any:
    """Build a Whisper processor accepting the repository's complete fast tokenizer."""
    from transformers import WhisperProcessor

    class FastWhisperProcessor(WhisperProcessor):
        tokenizer_class = "WhisperTokenizerFast"

    return FastWhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


@dataclass
class LoadedWhisper:
    model: Any
    processor: Any
    device: str


def select_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch.")
    return requested


def load_baseline(
    model_id: str = MODEL_ID,
    device: str = "auto",
    *,
    local_files_only: bool = False,
) -> LoadedWhisper:
    """Load processor and Whisper model, preserving Arabic transcription decoding."""
    import torch
    from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizerFast,
    )

    resolved_device = select_device(device)
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    # The published tokenizer config declares a slow tokenizer without the
    # vocab.json/merges.txt files required by that class. Its tokenizer.json is
    # complete, so use the matching fast implementation. ``extra_special_tokens``
    # is overridden only to bridge a metadata-format difference in Transformers
    # 4.48; vocabulary and special-token IDs remain those of the published model.
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_id, local_files_only=local_files_only, extra_special_tokens={}
    )
    processor = _build_fast_processor(feature_extractor, tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, local_files_only=local_files_only
    )
    model.to(resolved_device)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    # ``language`` and ``task`` build the decoder prompt in current Transformers.
    # Keeping forced_decoder_ids as well produces a conflict warning.
    model.generation_config.forced_decoder_ids = None
    return LoadedWhisper(model=model, processor=processor, device=resolved_device)


def transcribe_array(loaded: LoadedWhisper, audio, sampling_rate: int = 16_000) -> str:
    """Run a baseline-compatible inference on an in-memory mono waveform."""
    import torch

    inputs = loaded.processor(
        audio, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features.to(loaded.device)
    with torch.inference_mode():
        tokens = loaded.model.generate(input_features=inputs)
    return loaded.processor.batch_decode(tokens, skip_special_tokens=True)[0]
