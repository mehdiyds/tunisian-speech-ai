"""Extract embedded audio and transcripts from the local Parquet dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "senior_tunisian_voice" / "data"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "train.jsonl"


def parquet_files(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".parquet":
        return [source]
    files = sorted(source.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet shards found under: {source}")
    return files


def audio_bytes(audio: Any) -> bytes:
    if isinstance(audio, dict) and audio.get("bytes"):
        return bytes(audio["bytes"])
    if isinstance(audio, dict) and audio.get("path"):
        return Path(audio["path"]).read_bytes()
    raise ValueError("Audio record has neither embedded bytes nor a readable path")


def extract(source: Path, audio_dir: Path, manifest_path: Path, limit: int | None) -> int:
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for shard in parquet_files(source):
            frame = pd.read_parquet(shard, columns=["audio_id", "audio", "transcript"])
            for record in frame.to_dict(orient="records"):
                if limit is not None and count >= limit:
                    return count

                audio_id = str(record["audio_id"]).strip()
                transcript = str(record["transcript"]).strip()
                if not audio_id or not transcript:
                    raise ValueError(f"Empty audio_id or transcript in {shard}")

                audio_path = audio_dir / f"{audio_id}.wav"
                audio_path.write_bytes(audio_bytes(record["audio"]))
                manifest_record = {
                    "audio": audio_path.relative_to(PROJECT_ROOT).as_posix(),
                    "text": transcript,
                    "split": "train",
                    "source": "senior_tunisian_voice",
                }
                manifest.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
                count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    count = extract(args.source, args.audio_dir, args.manifest, args.limit)
    print(f"Extracted {count} record(s).")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
