"""Load a validated local JSONL/CSV training manifest into Hugging Face Datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .validate_dataset import read_manifest, validate_records


def load_local_manifest(path: str | Path):
    """Return a Dataset only after schema, final-test and speaker-leakage checks."""
    from datasets import Dataset

    records: list[dict[str, Any]] = read_manifest(path)
    errors = validate_records(records)
    if errors:
        raise ValueError("Invalid manifest:\n- " + "\n- ".join(errors))
    return Dataset.from_list(records)

