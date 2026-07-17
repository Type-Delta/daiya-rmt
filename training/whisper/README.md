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
- loss optimization: evaluation defaults to `eval_loss`; add `--predict-with-generate` when you want WER generation during eval
- best-checkpoint loading: opt in with `--load-best-model-at-end` when `--save-steps` and `--eval-steps` are aligned
- language conditioning: `--language-policy metadata` uses each row's `language` metadata when it maps to a Whisper language

`--language-policy` accepts:

- `metadata`: use row-level metadata such as `Thai`, `th`, `Thai-English`, `English`, `en`, `Japanese`, or `Japanese-English`; unknown and ambiguous values fall back to `--language` when set
- `global`: use `--language` for every row
- `none`: omit Whisper language tokens

The dataset loader reads all metadata columns so language can condition label tokenization. Context columns such as `context_before`, `context_after`, and `notes` are not injected into labels because this trainer is for audio-to-transcript LoRA tuning; adding textual context as decoder prompt conditioning should be a separate experiment.

Ownership-safe datasets carry `training_eligible`. Training and validation exclude `false` rows by default and write `dataset-selection.json` beside the run with included/excluded identities. Automatic validation splits are by `source_id`/`source_file`, never by adjacent clip, and explicit splits are rejected if they share a recording. The default for a legacy dataset that lacks the field is to stop rather than guess. Regenerate it, or use the explicit compatibility choice `--legacy-training-eligibility include` (or `exclude`) for a documented research run. `--include-ineligible-for-research` is a separate, explicit override for review-only timestamp-ownership rows. Feature caches are keyed by the selected rows, labels, and preprocessing configuration, so an earlier research override cannot leak rows into a later default run.

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
