# M3 Prompt-Conditioned Whisper Training

## Purpose

Runtime `initial_prompt` experiments were mixed but promising: overall micro CER regressed slightly while WER-like behavior and short-utterance CER improved. A likely reason is that the LoRA model was trained only on audio-to-transcript labels and never learned to use Whisper's decoder prompt slot. This experiment trains an opt-in variant that sees bounded context/terminology prompt tokens during fine-tuning.

This PR does not add generation-gated checkpoint selection. Use generation-gated evaluation when comparing checkpoints from this experiment.

## Training Modes

Baseline:

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --model-name-or-path openai/whisper-large-v3 `
  --output-dir training/whisper/runs/m3-large-v3-baseline `
  --num-train-epochs 2 `
  --learning-rate 2e-5 `
  --per-device-train-batch-size 4 `
  --gradient-accumulation-steps 4 `
  --lora-r 16 `
  --lora-alpha 32 `
  --lora-target-modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 `
  --load-in-4bit `
  --fp16 `
  --gradient-checkpointing
```

Prompt-conditioned variant:

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --model-name-or-path openai/whisper-large-v3 `
  --output-dir training/whisper/runs/m3-large-v3-prompt-terms `
  --prompt-conditioning `
  --prompt-max-tokens 64 `
  --prompt-fields context_before `
  --num-train-epochs 2 `
  --learning-rate 2e-5 `
  --per-device-train-batch-size 4 `
  --gradient-accumulation-steps 4 `
  --lora-r 16 `
  --lora-alpha 32 `
  --lora-target-modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 `
  --load-in-4bit `
  --fp16 `
  --gradient-checkpointing
```

The prompt-conditioned variant defaults to extracting only `Terms:` lines from `context_before`. Do not include `context_after` in the main experiment because it may contain current-chunk information unavailable at runtime.

## Smoke Commands

Dependency check:

```powershell
uv sync --project training/whisper
```

Unit tests:

```powershell
$env:PYTHONPATH = "training/whisper/src"
python -m unittest discover -s training/whisper/tests -v
```

Tiny prompted preprocessing/training smoke:

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --model-name-or-path openai/whisper-medium `
  --prompt-conditioning `
  --prompt-max-tokens 32 `
  --max-train-samples 8 `
  --max-eval-samples 4 `
  --max-steps 1 `
  --output-dir training/whisper/runs/m3-prompt-smoke
```

Expected output: preprocessing creates `input_features`, transcript labels, and `prompt_label_length` for rows with prompt conditioning enabled; the one-step run should finish and save a LoRA adapter without prompt tokens contributing to loss.

## Evaluation Matrix

Evaluate each trained adapter in four cells:

| Model | Decode without runtime prompt | Decode with runtime rolling prompt |
| --- | --- | --- |
| Baseline LoRA | baseline compatibility | current runtime-prompt behavior |
| Prompt-conditioned LoRA | prompt-trained robustness when prompt is absent | target M3 behavior |

Score:

- Overall CER and WER-like metric.
- Short-utterance CER, because runtime prompts previously helped this slice.
- Technical-term subset accuracy, especially English terms embedded in Thai or Japanese speech.
- Regression notes for copying previous text or over-biasing to listed terms.

Interpretation:

- Success: prompt-conditioned LoRA improves technical-term and short-utterance slices with neutral or improved overall CER when decoded with runtime prompts.
- Warning: large no-prompt regression means the model over-relies on prompt tokens.
- Failure: copied previous text or hallucinated terms means the prompt budget/content is too strong; reduce `--prompt-max-tokens`, keep terms-only mode, or add prompt dropout/noise in a later PR.

## Completed M3 Run - 2026-07-08

This run used the M2 large-v3 QLoRA setup with prompt conditioning enabled:

- Base model: `openai/whisper-large-v3`
- Dataset: `C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\hf_datasets\whisper`
- Output adapter: `C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\m3\full\largev3-m3-iter1-prompt-terms`
- Best checkpoint: `checkpoint-800`
- Best eval loss: `0.4239756166934967`
- CT2 output: `C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\m3\full\largev3-m3-iter1-prompt-terms-checkpoint-800-ct2-int8_float16`

Actual training command:

```powershell
$env:PYTHONPATH = "C:\Users\Kornnaras\.codex\worktrees\5e93\daiya-rmt\training\whisper\src"
$python = "C:\JokaMain\ProjectShowRoom\daiya-rmt\.venv\Scripts\python.exe"
$data = "C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\hf_datasets\whisper"
$out = "C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\m3\full\largev3-m3-iter1-prompt-terms"

& $python -m daiya_whisper_lora.cli train `
  --dataset-dir $data `
  --model-name-or-path openai/whisper-large-v3 `
  --output-dir $out `
  --prompt-conditioning `
  --prompt-max-tokens 64 `
  --prompt-fields context_before `
  --num-train-epochs 2 `
  --per-device-train-batch-size 2 `
  --per-device-eval-batch-size 2 `
  --gradient-accumulation-steps 8 `
  --learning-rate 2e-5 `
  --warmup-steps 50 `
  --lora-r 16 `
  --lora-alpha 32 `
  --lora-target-modules "q_proj,k_proj,v_proj,out_proj,fc1,fc2" `
  --load-in-4bit `
  --fp16 `
  --gradient-checkpointing `
  --eval-steps 100 `
  --save-steps 100 `
  --logging-steps 25 `
  --load-best-model-at-end
```

Validation loss curve:

| Step | Epoch | Eval loss |
| ---: | ---: | ---: |
| 100 | 0.24 | 0.5499 |
| 200 | 0.48 | 0.4957 |
| 300 | 0.71 | 0.4722 |
| 400 | 0.95 | 0.4545 |
| 500 | 1.19 | 0.4392 |
| 600 | 1.43 | 0.4298 |
| 700 | 1.66 | 0.4258 |
| 800 | 1.90 | 0.4240 |

Conversion command:

```powershell
& $python -m daiya_whisper_lora.cli merge `
  --adapter-path "C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\m3\full\largev3-m3-iter1-prompt-terms\checkpoint-800" `
  --base-model openai/whisper-large-v3 `
  --merged-output-dir "C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\m3\full\largev3-m3-iter1-prompt-terms-checkpoint-800-merged" `
  --ct2-output-dir "C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\m3\full\largev3-m3-iter1-prompt-terms-checkpoint-800-ct2-int8_float16" `
  --quantization int8_float16 `
  --skip-wer
```

Benchmark command shape:

```powershell
& $python lab/asr_eval.py `
  --model <m2-or-m3-ct2-model> `
  --dataset-dir "C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\hf_datasets\whisper" `
  --output-dir <benchmark-output-dir> `
  --limit 32 `
  --device auto `
  --compute-type int8_float16 `
  --language th `
  --no-condition-on-previous-text `
  --benchmark-strategies isolated,rolling_initial_prompt,left_audio_context,merged_deferred_short
```

Benchmark artifacts:

- M2 summary: `C:\JokaMain\ProjectShowRoom\daiya-rmt\lab\artifacts\asr_eval\m3-vs-m2\m2\summary_20260708T073731Z_training_whisper_runs_largev3-m2-iter1-ct2-int8_float16_benchmark_isolated_rolling_initial_prompt_left_audio_context_merged_deferred_short.json`
- M3 summary: `C:\JokaMain\ProjectShowRoom\daiya-rmt\lab\artifacts\asr_eval\m3-vs-m2\m3\summary_20260708T074718Z_training_whisper_runs_m3_full_largev3-m3-iter1-prompt-terms-checkpoint-800-ct2-int8_float16_benchmark_isolated_rolling_initial_prompt_left_audio_context_merged_deferred_short.json`

Overall benchmark result, across all four strategies:

| Model | Count | Micro CER | Mean CER | Micro WER-like | Mean WER-like | Short micro CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 | 120 | 16.17% | 15.79% | 16.39% | 16.54% | 16.39% |
| M3 | 120 | 14.22% | 14.46% | 14.27% | 14.92% | 15.59% |
| Delta M3-M2 |  | -1.95 pp | -1.33 pp | -2.12 pp | -1.63 pp | -0.80 pp |

Per-strategy result:

| Strategy | M2 micro CER | M3 micro CER | CER delta | M2 micro WER-like | M3 micro WER-like | WER delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| isolated | 14.24% | 14.59% | +0.34 pp | 14.74% | 14.74% | 0.00 pp |
| rolling_initial_prompt | 13.38% | 12.93% | -0.45 pp | 13.36% | 13.13% | -0.24 pp |
| left_audio_context | 21.16% | 14.45% | -6.71 pp | 21.18% | 13.89% | -7.30 pp |
| merged_deferred_short | 15.89% | 14.89% | -1.00 pp | 16.26% | 15.31% | -0.95 pp |

Short-utterance micro CER by strategy:

| Strategy | M2 | M3 | Delta |
| --- | ---: | ---: | ---: |
| isolated | 15.97% | 16.53% | +0.56 pp |
| rolling_initial_prompt | 17.09% | 16.53% | -0.56 pp |
| left_audio_context | 16.53% | 14.57% | -1.96 pp |
| merged_deferred_short | 16.27% | 15.35% | -0.92 pp |

Interpretation for this run:

- M3 improves the aggregate benchmark by 1.95 percentage points micro CER and 2.12 percentage points micro WER-like.
- The isolated no-runtime-prompt path regresses slightly on CER, so the model should not replace M2 solely on isolated decoding.
- Runtime-context strategies improve, especially `left_audio_context`; `rolling_initial_prompt` also improves modestly.
- This benchmark is still a 32-row smoke-style comparison. Use generation-gated checkpoint evaluation on a larger held-out set before promoting M3 as the default model.
