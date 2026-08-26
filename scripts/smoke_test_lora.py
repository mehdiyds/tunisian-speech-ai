"""Short synthetic LoRA engineering test. It is not a scientific fine-tuning run."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.load_model import load_baseline, transcribe_array
from src.training.lora_config import attach_lora, format_parameter_summary


def synthetic_audio(seconds: float = 1.0, sample_rate: int = 16_000) -> np.ndarray:
    time = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.05 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2, choices=range(1, 6))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-id", default="oddadmix/Whisperv3-tunisian-codeswitch")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print("FAILED: use Python 3.11 or 3.12; Python 3.14 is unsupported.")
        return 2

    import torch

    audio = synthetic_audio()
    print("1/8 Loading baseline model and processor...")
    loaded = load_baseline(args.model_id, args.device)
    if not args.skip_inference:
        print("2/8 Baseline inference on synthetic audio...")
        print("Baseline prediction:", repr(transcribe_array(loaded, audio)))

    print("3/8 Attaching LoRA adapters...")
    model = attach_lora(loaded.model)
    print(format_parameter_summary(model))
    model.config.use_cache = False
    model.train()

    print(f"4/8 Preprocessing synthetic audio and running {args.steps} training step(s)...")
    features = loaded.processor(audio, sampling_rate=16_000, return_tensors="pt").input_features
    labels = loaded.processor.tokenizer(text_target="اختبار تقني").input_ids
    feature_batch = features.to(loaded.device)
    label_batch = torch.tensor([labels], device=loaded.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4
    )
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_features=feature_batch, labels=label_batch).loss
        loss.backward()
        optimizer.step()
        print(f"  step {step}/{args.steps}: loss={loss.item():.4f}")

    print("5/8 LoRA forward/backward pass: PASSED")
    if not args.skip_inference:
        print("6/8 LoRA inference on synthetic audio...")
        model.eval()
        print("LoRA prediction:", repr(transcribe_array(type(loaded)(model, loaded.processor, loaded.device), audio)))

    with tempfile.TemporaryDirectory(prefix="whisper_lora_smoke_") as temporary_directory:
        adapter_path = Path(temporary_directory) / "adapter"
        model.save_pretrained(adapter_path)
        print(f"7/8 Temporary adapter saved: {adapter_path.name}")
        print("8/8 SMOKE TEST PASSED — synthetic technical check only; no training result.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print("FAILED: CUDA out of memory. Try --device cpu, or a supported quantized/offload setup.")
            raise SystemExit(3) from error
        raise
