# Daiya Whisper LoRA Training

Fine-tunes `openai/whisper-medium` with LoRA using the local Daiya Hugging Face `audiofolder` dataset at:

```text
training/dataset/hf_datasets/whisper
```

The project is a `uv` workspace member and uses the root workspace `.venv`, matching the dataset processor layout.

## Setup

```powershell
cd C:\JokaMain\ProjectShowRoom\daiya-rmt
uv sync --project training/whisper
```

This project pins CUDA PyTorch wheels through the same `pytorch-cu128` uv index used by the dataset processor. If your driver cannot run CUDA 12.8 wheels, change the `[[tool.uv.index]]` URL in `training/whisper/pyproject.toml` before syncing.

## Inspect Dataset

```powershell
uv run --project training/whisper daiya-whisper-lora inspect
```

## Train

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --output-dir training/whisper/runs/whisper-medium-lora `
  --num-train-epochs 3 `
  --per-device-train-batch-size 4 `
  --gradient-accumulation-steps 4 `
  --learning-rate 1e-4 `
  --fp16
```

Defaults:

- dataset: `training/dataset/hf_datasets/whisper`
- base model: `openai/whisper-medium`
- LoRA target modules: `q_proj,v_proj`
- validation split: created from training data with `--validation-size 0.05` when no validation split exists
- loss optimization: evaluation defaults to `eval_loss`; add `--predict-with-generate` when you want generated CER/WER-like metrics during eval
- best-checkpoint loading: opt in with `--load-best-model-at-end` when `--save-steps` and `--eval-steps` are aligned
- generation-gated selection: when both `--predict-with-generate` and `--load-best-model-at-end` are enabled, the trainer selects by generated `cer`
- language conditioning: `--language-policy metadata` uses each row's `language` metadata when it maps to a Whisper language

`--language-policy` accepts:

- `metadata`: use row-level metadata such as `Thai`, `th`, `Thai-English`, `English`, `en`, `Japanese`, or `Japanese-English`; unknown and ambiguous values fall back to `--language` when set
- `global`: use `--language` for every row
- `none`: omit Whisper language tokens

The dataset loader reads all metadata columns so language can condition label tokenization. Context columns such as `context_before`, `context_after`, and `notes` are not injected into labels because this trainer is for audio-to-transcript LoRA tuning; adding textual context as decoder prompt conditioning should be a separate experiment.

Use `--max-train-samples` and `--max-eval-samples` for smoke runs:

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --max-train-samples 8 `
  --max-eval-samples 4 `
  --max-steps 1 `
  --output-dir training/whisper/runs/smoke
```

## Output

Training saves the LoRA adapter, tokenizer, feature extractor, and processor files under `--output-dir`. Use the adapter with PEFT on top of the same base model.

## Checkpoint Probe

Use `probe-checkpoints` to compare saved LoRA checkpoints with generated text metrics before merging or converting an adapter. The probe is intentionally small by default (`--max-samples 32`) and reports CER, no-space CER, WER-like score, short-utterance subset metrics, and English technical-term subset metrics.

```powershell
uv run --project training/whisper daiya-whisper-lora probe-checkpoints `
  --run-dir training/whisper/runs/largev3-m2-iter1 `
  --base-model openai/whisper-large-v3 `
  --dataset-dir training/dataset/hf_datasets/whisper `
  --max-samples 32 `
  --device cuda `
  --fp16
```

Probe summaries and per-sample details are written under:

```text
training/whisper/runs/checkpoint_probes
```

The summary names the selected checkpoint by `micro_cer` and includes the selected-vs-final adapter delta when the final adapter is present. Selection only considers candidates with at least one scored row and a finite primary metric. If every candidate is invalid, the command exits with an error after writing per-sample details and a `status: failed` summary with candidate failure counts.
