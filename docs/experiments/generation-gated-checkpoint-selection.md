# Generation-Gated Checkpoint Selection

## Problem

Whisper LoRA training can report strong teacher-forced validation loss while producing worse generated transcripts. For Daiya mixed-lingual transcription, checkpoint selection must follow generated text quality when generation evaluation is explicitly enabled.

The intended CLI contract is frozen:

```bash
daiya-whisper-lora train --predict-with-generate --load-best-model-at-end
```

When both flags are enabled, training selects the best checkpoint by generated `cer`, not teacher-forced loss or WER.

## Future Training Command

Use generation-gated selection for new Whisper LoRA runs:

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --predict-with-generate `
  --load-best-model-at-end
```

Run-specific model, dataset, LoRA, batch, and schedule flags should remain documented with each experiment config.

## M2 Checkpoint Probe Command

Probe all `checkpoint-*` adapters plus the final adapter when present:

```powershell
uv run --project training/whisper daiya-whisper-lora probe-checkpoints `
  --run-dir training/whisper/runs/largev3-m2-iter1 `
  --base-model openai/whisper-large-v3 `
  --dataset-dir training/dataset/hf_datasets/whisper `
  --max-samples 32 `
  --device cuda `
  --fp16
```

This probe is intentionally small and fixed so it can run on an 8 GB laptop GPU.

## Expected Artifacts

By default, probe outputs are written under:

```text
training/whisper/runs/checkpoint_probes/
```

Expected files:

- JSON summary for the probe run.
- JSONL per-checkpoint/per-sample details for inspection and QA.

Only candidates with at least one successfully scored row and a finite primary metric are eligible for selection. A completely failed probe still writes both artifacts: the JSON summary has `status: failed`, no selected checkpoint, and candidate attempted/scored/failed status counts; the JSONL preserves per-sample errors. The command then exits with an actionable error pointing to both paths.

Primary selection metric:

- `micro_cer`

Additional reported slices:

- no-space CER
- WER-like score
- short-utterance subset
- English technical-term subset

## M2 Interpretation

Known M2 facts:

- Checkpoint 400 is favored by generated CER, around 20.9-21.7% CER on small probes.
- Later final/best-loss adapters are worse despite looking better by loss.
- Corrected M2 CT2 benchmark is about 22.9% CER.
- Previous large-v3 iter4 CT2 benchmark was about 21.2% CER.

Interpretation: M2 supports generation-gated checkpoint selection as a necessary guardrail, but it does not beat the previous large-v3 iter4 CT2 benchmark after correction.

## QA Notes

Claude QA was run read-only with Opus fallback (`opus-4.8` was rejected by the local CLI, so `opus --effort high` was used). Valid findings were addressed:

- fp16/4-bit probe inputs are now cast to the model input dtype before generation.
- Trainer CER and probe CER both apply the same Thai spacing normalization to predictions.
- Short-utterance selection accepts `speech_duration`, `audio_duration_seconds`, or `duration`, and the probe can derive audio duration from local audio metadata when available.
- Probing falls back to `train` only with a warning when no held-out split exists.
- Removed the now-unused `evaluate` dependency from the Whisper training package.

## Blocked Heavy Benchmark

Full M2 checkpoint probing and CT2 benchmarking were not run in this PR because they require local model artifacts/GPU time. Ready commands are listed above. This change documents and implements the selection/probe workflow only; it does not implement prompt-conditioned training, a phonetic normalizer, Silero VAD, right-context/lookahead, or word-level speaker assignment.
