# Dataset convention

## Manifest schema

Every record requires `audio` (path or dataset audio object) and `text`. Recommended metadata: `speaker_id`, `split`, `source`, `domain`, `code_switching`, `audio_quality`, `region`, and `duration_seconds`.

Use a UTF-8 JSONL or CSV manifest. `split` may be `train`, `validation`, or `test`; the project test split is **not** the independent 96-utterance final benchmark. Never ingest those 96 files into this dataset.

## Transcription rules

- Transcribe faithfully to the audible utterance; do not silently correct words.
- Treat normal Tunisian variants as valid forms rather than pronunciation errors.
- Keep one documented spelling for recurring Derja forms and Arabizi conventions (digits, Latin casing, apostrophes, and spacing).
- Preserve French and English words as spoken; do not translate them. Keep code-switches in their original scripts/forms consistently.
- Retain meaningful hesitations, repetitions and false starts using one shared notation decided before annotation. Apply it uniformly.
- Mark uncertain or unusable audio in metadata rather than guessing a word.

## Splits and speakers

When `speaker_id` is available, split by speaker: a speaker must not appear in both train and validation. The pipeline rejects train/validation speaker overlap by default.

