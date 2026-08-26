"""Whisper preprocessing and a padding collator for future corpus training."""
from __future__ import annotations

from typing import Any


def prepare_example(example: dict[str, Any], processor: Any) -> dict[str, Any]:
    """Convert one decoded audio example to Whisper features and unchanged text labels."""
    audio = example["audio"]
    features = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    labels = processor.tokenizer(text_target=example["text"]).input_ids
    return {"input_features": features, "labels": labels}


class WhisperDataCollator:
    """Pad acoustic features and label tokens while masking label padding with -100."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        inputs = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in features], return_tensors="pt"
        )
        batch["labels"] = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )
        return {key: value if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

