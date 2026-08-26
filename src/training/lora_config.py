"""Build and inspect the conservative Whisper LoRA adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_lora_settings(config_path: str | Path = "configs/lora_whisper.yaml") -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def attach_lora(model: Any, config_path: str | Path = "configs/lora_whisper.yaml") -> Any:
    """Freeze baseline weights and add LoRA to Whisper q/v attention projections."""
    from peft import LoraConfig, TaskType, get_peft_model

    settings = load_lora_settings(config_path)["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=settings["r"],
        lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        target_modules=settings["target_modules"],
        bias=settings["bias"],
    )
    return get_peft_model(model, peft_config)


def parameter_summary(model: Any) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100 * trainable / total if total else 0.0,
    }


def format_parameter_summary(model: Any) -> str:
    summary = parameter_summary(model)
    return (
        f"Total parameters: {summary['total_parameters']:,}\n"
        f"Trainable parameters: {summary['trainable_parameters']:,}\n"
        f"Trainable %: {summary['trainable_percent']:.4f}%"
    )

