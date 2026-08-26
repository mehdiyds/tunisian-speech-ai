# Data directory

No speech data, final-test audio, or model weights belongs in Git.

- `raw/`: local source manifests and audio (ignored by Git).
- `processed/`: preprocessed artifacts (ignored by Git).
- `splits/`: generated train/validation manifests. Do not put the external 96-utterance final test set here.

Use JSON Lines or CSV manifests. Required fields are `audio` and `text`; see [`../docs/dataset_convention.md`](../docs/dataset_convention.md).

