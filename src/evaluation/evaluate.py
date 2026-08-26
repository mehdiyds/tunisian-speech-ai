"""Metrics helpers; run final evaluation only on the protected external 96 set."""
from __future__ import annotations


def wer_cer(predictions: list[str], references: list[str]) -> dict[str, float]:
    from jiwer import cer, wer

    return {"wer": wer(references, predictions), "cer": cer(references, predictions)}

