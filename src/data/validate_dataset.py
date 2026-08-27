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
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        decoder = json.JSONDecoder()
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            position = 0
            while position < len(line):
                while position < len(line) and line[position].isspace():
                    position += 1
                if position == len(line):
                    break
                try:
                    record, next_position = decoder.raw_decode(line, position)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}, "
                        f"column {error.colno}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Manifest record at line {line_number} must be a JSON object."
                    )
                records.append(record)
                position = next_position
        return records
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("JSON manifest must contain a list of objects.")
        return data
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
        audio_path = Path(str(record["audio"]))
        if not audio_path.is_absolute() and not audio_path.exists():
            errors.append(f"row {index}: audio path does not exist: {audio_path}")
        elif audio_path.is_absolute() and not audio_path.exists():
            errors.append(f"row {index}: audio path does not exist: {audio_path}")
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

