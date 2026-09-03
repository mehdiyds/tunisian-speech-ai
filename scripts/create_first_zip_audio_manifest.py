"""Create a JSONL manifest for the first audio files stored in a ZIP archive."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[Path(record["audio"]).name] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-zip", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()

    records = load_records(args.train_manifest)
    records.update(load_records(args.validation_manifest))
    with zipfile.ZipFile(args.audio_zip) as archive:
        audio_names = [
            Path(name).name
            for name in archive.namelist()
            if Path(name).suffix.lower() in AUDIO_SUFFIXES
        ][: args.count]

    missing = [name for name in audio_names if name not in records]
    if len(audio_names) != args.count or missing:
        raise ValueError(f"Expected {args.count} matched audio files; missing: {missing[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(records[name], ensure_ascii=False) + "\n" for name in audio_names),
        encoding="utf-8",
    )
    print(f"Created {args.output} with {len(audio_names)} records.")


if __name__ == "__main__":
    main()
