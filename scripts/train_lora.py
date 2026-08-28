"""Pilot or final Whisper LoRA training with validation metrics.

This script saves PEFT adapters and trainer checkpoints; it never saves the full
baseline model weights. Pilot metrics are engineering evidence, not scientific
results.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import Audio, Dataset
from jiwer import cer, wer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_dataset import load_local_manifest
from src.data.prepare_dataset import WhisperDataCollator, prepare_example
from src.model.load_model import LANGUAGE, TASK, load_baseline
from src.training.lora_config import attach_lora, parameter_summary, load_lora_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "lora_pilot")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "lora_whisper.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    return parser.parse_args()


def check_manifest_paths(path: Path) -> None:
    records = load_local_manifest(path)
    missing = []
    for record in records:
        record_dict: dict[str, Any] = dict(record)
        audio_path = str(record_dict["audio"])
        if not Path(audio_path).exists():
            missing.append(audio_path)
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio path(s) are missing; first: {missing[0]}")


def load_split(path: Path, limit: int | None = None) -> Dataset:
    dataset = load_local_manifest(path)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def split_train_dataset(dataset: Dataset, validation_split: float) -> tuple[Dataset, Dataset]:
    if not 0 < validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1")
    split = dataset.train_test_split(test_size=validation_split, seed=42)
    return split["train"], split["test"]


def prepare_dataset(dataset: Dataset, processor: Any) -> Dataset:
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    return dataset.map(
        partial(prepare_example, processor=processor),
        remove_columns=dataset.column_names,
        desc="Extract Whisper features",
    )


def compute_metrics(processor: Any):
    def metrics(eval_prediction: Any) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        predicted_text = processor.batch_decode(predictions, skip_special_tokens=True)
        reference_text = processor.batch_decode(labels, skip_special_tokens=True)
        return {
            "wer": float(wer(reference_text, predicted_text)),
            "cer": cast(float, cer(reference_text, predicted_text)),
        }

    return metrics


class AdapterSeq2SeqTrainer(Seq2SeqTrainer):
    """Train Whisper directly while saving only the attached PEFT adapter."""

    def __init__(self, *args: Any, adapter_model: Any, **kwargs: Any) -> None:
        self.adapter_model = adapter_model
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.adapter_model.save_pretrained(destination)


def main() -> int:
    args = parse_args()
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise RuntimeError("Use Python 3.11 or 3.12 for this project.")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or a positive integer")

    settings = load_lora_settings(args.config)
    model_id = args.model_id or settings["model_id"]
    check_manifest_paths(args.train_manifest)
    train_dataset = load_split(args.train_manifest, args.max_train_samples)
    if args.validation_manifest:
        check_manifest_paths(args.validation_manifest)
        validation_dataset = load_split(args.validation_manifest, args.max_validation_samples)
    else:
        train_dataset, validation_dataset = split_train_dataset(train_dataset, args.validation_split)
        if args.max_validation_samples is not None:
            validation_dataset = validation_dataset.select(
                range(min(args.max_validation_samples, len(validation_dataset)))
            )

    loaded = load_baseline(model_id, args.device)
    model = attach_lora(loaded.model, args.config)
    model.config.use_cache = False
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    if loaded.device == "cuda":
        model.get_base_model().gradient_checkpointing_enable()
        model.get_base_model().enable_input_require_grads()

    train_dataset = prepare_dataset(train_dataset, loaded.processor)
    validation_dataset = prepare_dataset(validation_dataset, loaded.processor)
    summary = parameter_summary(model)
    print(json.dumps(summary, indent=2))

    training = settings["training"]
    fp16 = bool(training.get("fp16", True) and loaded.device == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=args.max_steps,
        fp16=fp16,
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=max(1, args.eval_steps or args.logging_steps),
        save_steps=max(1, args.save_steps or args.logging_steps),
        logging_steps=max(1, args.logging_steps),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        report_to="none",
        remove_unused_columns=False,
        seed=int(training.get("seed", 42)),
    )
    trainer = AdapterSeq2SeqTrainer(
        model=model.get_base_model(),
        adapter_model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=WhisperDataCollator(
            loaded.processor,
            input_dtype=torch.float16 if loaded.device == "cuda" else torch.float32,
        ),
        compute_metrics=compute_metrics(loaded.processor),
    )
    set_seed(int(training.get("seed", 42)))
    resume = args.resume_from_checkpoint
    result = trainer.train(resume_from_checkpoint=resume)

    best_adapter = args.output_dir / "best_adapter"
    best_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_adapter)
    loaded.processor.save_pretrained(best_adapter)
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    metrics["train_samples"] = len(train_dataset)
    metrics["validation_samples"] = len(validation_dataset)
    metrics["device"] = loaded.device
    metrics["model_id"] = model_id
    metrics["language"] = LANGUAGE
    metrics["task"] = TASK
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "parameter_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Best LoRA adapter saved to: {best_adapter}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print("CUDA out of memory: reduce pilot samples/steps or use a supported quantized setup.")
            raise SystemExit(3) from error
        raise
