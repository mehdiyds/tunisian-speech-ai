# Tunisian Derja STT — LoRA fine-tuning preparation

Technical preparation for adapting the community baseline [`oddadmix/Whisperv3-tunisian-codeswitch`](https://huggingface.co/oddadmix/Whisperv3-tunisian-codeswitch) to complementary Tunisian speech data.

## Status

| Item | Status |
| --- | --- |
| Environment | prepared (requires Python 3.11) |
| LoRA pipeline | prepared |
| Dataset collection | in progress |
| Training | **NOT STARTED** |
| Final evaluation | pending |

## Baseline and evaluation boundary

The baseline is a third-party fine-tuned community model. This project does not claim that its original fine-tuning was performed during the internship.

The independent 96-utterance benchmark test set is strictly **external**. It must never be copied into `data/raw`, `data/processed`, `train`, or `validation`; it is reserved solely for the final baseline-versus-LoRA comparison.

Decoding remains `language=ar`, `task=transcribe`, with the baseline processor and tokenizer unchanged.

## Setup

Use Python **3.11** (not 3.14) and create an isolated virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```

For exact reference versions, use `requirements-lock.txt`. Install a PyTorch CUDA build appropriate for the machine first via the [official PyTorch selector](https://pytorch.org/get-started/locally/).

## Dataset and smoke test

Place only future training-corpus manifests under `data/raw/` and follow [`docs/dataset_convention.md`](docs/dataset_convention.md). Validate a manifest:

```powershell
python -m src.data.validate_dataset data/raw/manifest.jsonl
```

Once dependencies and model access are available, run the short technical test:

```powershell
python scripts/smoke_test_lora.py --steps 2 --device auto
```

This uses synthetic audio only, attaches LoRA, performs a few optimisation steps, and writes an adapter in a temporary directory. It is not a scientific training run and does not demonstrate an accuracy improvement.

See [`docs/fine_tuning_plan.md`](docs/fine_tuning_plan.md) for the methodology and initial hyperparameters.

