from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from datasets import Dataset, load_dataset
except Exception as exc:  # pragma: no cover - import-time guard for limited environments
    Dataset = None
    load_dataset = None
    import_error = exc
else:
    import_error = None

root = Path(__file__).resolve().parents[1]
parquet_dir = root / "data" / "raw" / "senior_tunisian_voice" / "data"
parquet_files = sorted(parquet_dir.glob("train-*.parquet"))

if not parquet_files:
    raise FileNotFoundError(f"No parquet files found under: {parquet_dir}")

sample_file = parquet_files[0]
df = pd.read_parquet(sample_file)
print("Columns:", list(df.columns))
print("Rows in first shard:", len(df))
print("Sample transcripts:")
for _, row in df.head(3).iterrows():
    transcript = row.get("transcript")
    print(f"  - {transcript[:120] if isinstance(transcript, str) else transcript}")

dataset_summary = None

if load_dataset is not None:
    try:
        loaded_dataset = load_dataset(
            "parquet",
            data_files=[str(path) for path in parquet_files],
            split="train",
        )
        if Dataset is not None and isinstance(loaded_dataset, Dataset):
            first_record = loaded_dataset[0]
            transcript = first_record.get("transcript")
            dataset_summary = {
                "type": "datasets",
                "len": len(loaded_dataset),
                "features": list(loaded_dataset.features.keys()),
                "first_transcript_preview": transcript[:120]
                if isinstance(transcript, str)
                else transcript,
            }
        else:
            print("datasets returned an unsupported dataset type; using PyArrow.")
    except Exception as exc:  # pragma: no cover - depends on installed datasets build
        print(f"datasets parquet loader failed: {type(exc).__name__}: {exc}")
        print("Falling back to direct PyArrow inspection.")

if dataset_summary is None:
    tables = [pq.read_table(path) for path in parquet_files]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    dataset_summary = {
        "type": "pyarrow",
        "num_rows": table.num_rows,
        "schema": list(table.schema.names),
        "first_transcript_preview": table["transcript"][0].as_py()[:120]
        if table.schema.names and "transcript" in table.schema.names
        else None,
    }

print("\nDataset summary:")
for key, value in dataset_summary.items():
    print(f"  {key}: {value}")
