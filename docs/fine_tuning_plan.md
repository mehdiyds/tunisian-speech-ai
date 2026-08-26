# Fine-tuning plan

## Scope

The selected baseline is `oddadmix/Whisperv3-tunisian-codeswitch`, a model previously fine-tuned by its publisher. The internship adaptation will add LoRA adapters while freezing baseline weights. It will not modify the tokenizer.

## Initial LoRA configuration

`configs/lora_whisper.yaml` starts with rank 8, alpha 16 and dropout 0.05 on `q_proj` and `v_proj`. These Whisper attention projections are a conservative PEFT surface: they reduce trainable parameters substantially while adapting attention. Learning rate (1e-4), effective batch size (1 × 8 accumulation), five epochs, 5% warm-up and a linear schedule are starting points—not final research choices.

With a 4 GB GPU, begin with batch size 1, fp16, gradient checkpointing, and short pilot runs. If the model does not fit, evaluate a supported quantized or CPU-offload setup before changing the experiment design.

## Evaluation protocol

Keep baseline decoding at `language=ar` and `task=transcribe`. The independent 96-utterance final test set remains outside the training dataset and is evaluated only after a finalized adapter exists. Compare baseline and baseline+LoRA under the same decoding protocol, reporting WER, CER and relevant domain/code-switch slices.

## Stages

1. Validate corpus manifests and isolate speakers between train and validation.
2. Run the synthetic smoke test; it only validates engineering.
3. Train on the final corpus and choose a checkpoint using validation data.
4. Evaluate the selected checkpoint once on the external 96-utterance set.

