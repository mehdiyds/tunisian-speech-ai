"""Validate local training manifests and protect the external final benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"audio", "text"}
FINAL_TEST_MARKERS = ("final_96", "test_96", "benchmark_96", "external_final")


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("Manifest must be a .jsonl or .csv file.")


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    speakers_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            errors.append(f"row {index}: missing required field(s): {sorted(missing)}")
            continue
        if not str(record["audio"]).strip() or not str(record["text"]).strip():
            errors.append(f"row {index}: audio and text must be non-empty")
        flattened = " ".join(str(value).lower() for value in record.values())
        if any(marker in flattened for marker in FINAL_TEST_MARKERS):
            errors.append(
                f"row {index}: appears to reference the protected external final benchmark"
            )
        split = str(record.get("split", "")).lower()
        if split and split not in {"train", "validation", "test"}:
            errors.append(f"row {index}: unsupported split {split!r}")
        speaker = str(record.get("speaker_id", "")).strip()
        if split in speakers_by_split and speaker:
            speakers_by_split[split].add(speaker)

    overlap = speakers_by_split["train"] & speakers_by_split["validation"]
    if overlap:
        errors.append(
            "speaker leakage between train and validation: " + ", ".join(sorted(overlap))
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="UTF-8 JSONL or CSV manifest")
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    errors = validate_records(records)
    if errors:
        print("Dataset validation FAILED:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Dataset validation PASSED: {len(records)} record(s), no speaker leakage found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

