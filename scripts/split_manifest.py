"""Create reproducible train/validation JSONL manifests from one manifest."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "splits" / "validation.jsonl"


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_records(path: Path, records: list[dict[str, Any]], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            output = dict(record)
            output["split"] = split
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def split_records(
    records: list[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two records are required to create two splits")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_ratio))
    validation = shuffled[:validation_size]
    train = shuffled[validation_size:]
    return train, validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_records(args.input)
    if any("speaker_id" not in record for record in records):
        print("WARNING: speaker_id is missing; this is a record-level split, not speaker-aware.")
    train, validation = split_records(records, args.validation_ratio, args.seed)
    write_records(args.train_output, train, "train")
    write_records(args.validation_output, validation, "validation")
    print(f"Total records: {len(records)}")
    print(f"Train records: {len(train)} -> {args.train_output}")
    print(f"Validation records: {len(validation)} -> {args.validation_output}")
    print(f"Seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
