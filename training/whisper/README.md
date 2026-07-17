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
- validation split: created from training data with `--validation-size 0.05` when no validation split exists and no `--split-manifest` is supplied
- explicit splits: `--split-manifest` disables row-level random splitting and partitions rows by manifest assignment while validating duplicate sample IDs and `source_file`/`conversation` group leakage
- loss optimization: evaluation defaults to `eval_loss`; add `--predict-with-generate` when you want generated micro CER/no-space CER/WER-like metrics during eval
- best-checkpoint loading: opt in with `--load-best-model-at-end` when `--save-steps` and `--eval-steps` are aligned
- generated best-checkpoint metric: when `--predict-with-generate` is enabled, trainer best-model selection uses generated `micro_cer`
- language conditioning: `--language-policy metadata` uses each row's `language` metadata when it maps to a Whisper language
- provenance: each training run writes `run_provenance.json` with git commit, dataset/manifest identity, seed, prompt construction, hyperparameters, cadence, and runtime/hardware details

`--language-policy` accepts:

- `metadata`: use row-level metadata such as `Thai`, `th`, `Thai-English`, `English`, `en`, `Japanese`, or `Japanese-English`; unknown and ambiguous values fall back to `--language` when set
- `global`: use `--language` for every row
- `none`: omit Whisper language tokens

The dataset loader reads all metadata columns so language can condition label tokenization. Context columns such as `context_before`, `context_after`, and `notes` are not injected by default, keeping the historical audio-to-transcript LoRA path unchanged.

## M3.1 Prompt Conditioning

Prompt-conditioned training is opt in:

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --prompt-conditioning `
  --prompt-max-tokens 64 `
  --prompt-fields context_before `
  --output-dir training/whisper/runs/whisper-large-v3-lora-prompt-terms
```

When enabled, the trainer extracts bounded prompt text from metadata and prepends it as Whisper decoder prompt tokens before the transcript label. Prompt tokens, including the transcript start token after the prompt, are masked from loss; only transcript/task target tokens train the model. The default prompt source extracts only `Terms:` lines from `context_before` so the model sees terminology hints without learning to copy previous transcript prose.

Prompt flags:

- `--prompt-conditioning`: enable decoder prompt conditioning; defaults off
- `--prompt-max-tokens`: maximum prompt body tokens; transcript tokens keep priority
- `--prompt-fields`: comma-separated metadata fields, default `context_before`
- `--prompt-full-context`: use the full selected context fields instead of only `Terms:` fragments
- `--prompt-allow-future-context`: permit fields such as `context_after`; use only for explicit offline-labeling/leakage experiments

Use an explicit split manifest for deterministic M3.1 runs:

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --split-manifest docs/experiments/m31/m31-split-v1.jsonl `
  --prompt-conditioning `
  --predict-with-generate `
  --load-best-model-at-end `
  --output-dir training/whisper/runs/m3_1
```

Manifest rows may identify either samples (`sample_id`, `id`, `uid`, or `file_name`) or groups (`source_file`, `conversation`, `conversation_id`, `group_id`, or `session_id`) with a `split` of `train`, `validation`, `test`, or `benchmark`. All rows in a `source_file`/`conversation` group must resolve to one split.

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
  --run-dir training/whisper/runs/m3_1 `
  --base-model openai/whisper-large-v3 `
  --dataset-dir training/dataset/hf_datasets/whisper `
  --split-manifest docs/experiments/m31/m31-split-v1.jsonl `
  --selector-manifest docs/experiments/m31/m31-generation-gate-v1.jsonl `
  --split validation `
  --prompt-strategy rolling-initial-prompt `
  --rolling-prompt-turns 3 `
  --rolling-prompt-chars 512 `
  --device cuda `
  --fp16
```

Probe summaries and per-sample details are written under:

```text
training/whisper/runs/checkpoint_probes
```

Selection requires every frozen selector row to generate successfully, a finite primary metric, and matching checkpoint/dataset/generation fingerprints. One failed or missing row makes a candidate ineligible. If every candidate is invalid or incomplete, the command exits with an error after writing details and a `status: failed` summary. Eval-loss fallback is never implicit; request it with `--generation-failure-policy eval-loss`, and the summary records `selection_mode: eval_loss_fallback`.

`Seq2SeqTrainer` generation evaluation is intentionally recorded as unprompted because Transformers removes label-shaped decoder inputs before `generate()`. For M3.1, the post-hoc probe with the frozen selector and runtime prompt strategy is authoritative; do not publish the trainer's loss-best or unprompted-generation-best checkpoint as the M3.1 model without this probe.
