# M2 ASR Model State

Last updated: 2026-07-07

## Summary

`largev3-m2-iter1` is the current experimental ASR model for Daiya's Thai-English
mixed-lingual path. It is a LoRA fine-tune of `openai/whisper-large-v3`, merged
back into the base model and converted to CTranslate2 for serving through
`faster-whisper`.

The runtime default currently points at:

```text
training/whisper/runs/largev3-m2-iter1-ct2-int8_float16
```

Current status: usable but not yet clearly better than the previous large-v3
iteration. M2 improves WER-like scoring in the available benchmark, but the first
CT2 comparison produced a bad CER result until decode/fallback behavior was
debugged. The corrected CT2 benchmark puts M2 at `22.9%` CER, roughly comparable
to the previous large-v3 model's `21.2%` CER, but slower.
Despite the degration m2 is still chosen as the default because it was trained on
a better dataset that better aligns with project goals.
The degraded WER/CER may came from the model dropping repeated words or unnecessary fillers,
which is a desirable behavior for the target domain.
However, giving the model ability to adjust transcription verbosity may impact
the CER/WER metrics, so further investigation is needed.

## Artifacts

Training adapter:

```text
training/whisper/runs/largev3-m2-iter1
```

Merged Transformers model:

```text
training/whisper/runs/largev3-m2-iter1-merged
```

Serving model:

```text
training/whisper/runs/largev3-m2-iter1-ct2-int8_float16
```

Important logs:

```text
training/whisper/runs/launch_m2.cmd
training/whisper/runs/largev3-m2-iter1.log
training/whisper/runs/m2-compare.log
training/whisper/runs/m2-compare2.log
training/whisper/runs/m2-ct2-cer.log
training/whisper/runs/m2-fallback-scan.log
training/whisper/runs/m2-merge.log
training/whisper/runs/m2-probe*.log
training/whisper/runs/m2-relabel.log
training/whisper/runs/m2-stall-debug.log
```

## Training Method

The M2 run was launched by `training/whisper/runs/launch_m2.cmd`.

1. Relabel raw Whisper training audio with the dataset processor. The original
   launch wrote to a temporary `whisper-m2` dataset path:

```powershell
cd training\processor\whisper
set DAIYA_OPENROUTER_MODEL=google/gemini-3.1-flash-lite
uv run auto-label `
  --input-dir C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\raw\whisper `
  --output-dir C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\hf_datasets\whisper-m2
```

The separate `whisper-m2` directory was later removed to save disk space. The M2
dataset replaced the previous dataset and is now the canonical dataset at:

```text
training/dataset/hf_datasets/whisper
```

2. Clear the feature cache.

3. Train a large-v3 QLoRA adapter. The original launch used the temporary
   `whisper-m2` path; current reruns should use the canonical M2 dataset path:

```powershell
uv run --project training/whisper daiya-whisper-lora train `
  --dataset-dir training/dataset/hf_datasets/whisper `
  --model-name-or-path openai/whisper-large-v3 `
  --output-dir training/whisper/runs/largev3-m2-iter1 `
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

Key differences from the earlier large-v3 run:

- New M2 labels were generated through the dataset processor with
  `google/gemini-3.1-flash-lite`.
- The M2 labels now occupy `training/dataset/hf_datasets/whisper`; the old
  pre-M2 labels are intentionally not kept in the workspace.
- Base model is `openai/whisper-large-v3`.
- LoRA targets include attention and FFN modules:
  `q_proj,k_proj,v_proj,out_proj,fc1,fc2`.
- Learning rate was reduced to `2e-5`.
- Training used 4-bit loading, fp16, and gradient checkpointing to fit the 8 GB
  RTX 5060 laptop GPU.

## Training Result

Training completed successfully:

```text
TRAIN-EXITCODE=0
```

Observed training/eval progression from `largev3-m2-iter1.log`:

| Step | Epoch | Eval loss |
| ---: | ---: | ---: |
| 200 | 0.24 | 0.7118 |
| 300 | 0.48 | 0.4904 |
| 400 | 0.71 | 0.4680 |
| 500 | 0.95 | 0.4563 |
| 600 | 1.19 | 0.4485 |
| 700 | 1.43 | 0.4406 |
| 800 | 1.66 | 0.4331 |
| final eval | 1.90 | 0.4271 |

Final training summary:

```text
train_runtime: 18268.6561s
train_samples_per_second: 0.736
train_steps_per_second: 0.046
train_loss: 0.4967
epoch: 2.0
```

Training ran for roughly 5.1 hours.

## Benchmarks

Available benchmark set:

- `32` samples
- `112.2s` total audio
- Thai-English mixed-lingual validation-like slice
- WER is Thai-spacing-sensitive, so CER is the more reliable headline metric

### Model Comparison

From `m2-compare.log` and `m2-compare2.log`:

| Model | WER-like | CER | Inference | RTF | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `medium-iter4` | 107.4% | 32.7% | 16.6-16.9s | 0.148-0.150 | 6.7-6.8x realtime |
| `largev3-iter4` | 88.4% | 21.2% | 23.0-23.4s | 0.205-0.209 | 4.8-4.9x realtime |
| `largev3-m2` initial run | 73.7% | 43.7% | 40.2s | 0.358 | 2.8x realtime |
| `largev3-m2` corrected run | 73.7% | 22.9% | 38.5s | 0.343 | 2.9x realtime |

Interpretation:

- M2 has the best WER-like number in this benchmark.
- Corrected M2 CER is close to the earlier large-v3 model but not better.
- M2 is materially slower than `largev3-iter4` in the CT2 serving path.
- The initial M2 CER result was bad enough to require debug/probe follow-up, so
  M2 should not be treated as a clean win yet.

### Runtime Context Strategy Benchmark

After the M2 short-utterance failures, we compared runtime-only context
strategies against the current local CT2 artifact. This did not involve any
training or prompt-conditioned fine-tuning.

Command:

```powershell
uv run --package daiya python lab/asr_eval.py `
  --model training/whisper/runs/largev3-m2-iter1-ct2-int8_float16 `
  --limit 32 `
  --device auto `
  --compute-type int8_float16 `
  --language th `
  --no-condition-on-previous-text `
  --benchmark-strategies isolated,rolling_initial_prompt,left_audio_context,merged_deferred_short
```

Artifacts:

```text
lab/artifacts/asr_eval/summary_20260707T065832Z_training_whisper_runs_largev3-m2-iter1-ct2-int8_float16_benchmark_isolated_rolling_initial_prompt_left_audio_context_merged_deferred_short.json
lab/artifacts/asr_eval/details_20260707T065832Z_training_whisper_runs_largev3-m2-iter1-ct2-int8_float16_benchmark_isolated_rolling_initial_prompt_left_audio_context_merged_deferred_short.jsonl
```

Final corrected benchmark numbers:

| Strategy | Rows/groups | Micro CER | Mean CER | Micro WER-like | Mean WER-like | Short micro CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolated` | 32 | 14.41% | 14.31% | 15.07% | 15.30% | 15.97% |
| `rolling_initial_prompt` | 32 | 14.62% | 13.77% | 14.36% | 14.14% | 15.13% |
| `left_audio_context` | 32 | 17.06% | 16.96% | 16.87% | 17.71% | 16.53% |
| `merged_deferred_short` | 24 | 22.00% | 20.20% | 20.52% | 19.68% | 28.47% |

Interpretation:

- Rolling ASR `initial_prompt` memory is mixed but promising: micro CER was
  slightly worse than isolated decode, but mean CER, WER-like, and the short
  subset improved.
- Left-audio context regressed in this benchmark. Runtime support remains useful
  for controlled experiments, but it is not justified as a broad default.
- Merging/deferring tiny chunks also regressed here and changes the scored unit
  from rows to groups, so it should stay experimental.
- The runtime default should therefore keep only rolling ASR prompt memory
  enabled. Left-audio retry, delayed ASR correction, and tiny utterance
  merge/defer are available only through explicit experimental config/CLI knobs.

### M2 CT2 Error Examples

From `m2-ct2-cer.log`, highest-error examples:

| Clip | CER | Prediction | Reference |
| ---: | ---: | --- | --- |
| 2 | 100% | `อันที่มาย` | `อธิบาย` |
| 19 | 100% | `ตัวสนาด` | `อ๋อ ครับ` |
| 26 | 93% | `อ่า` | `เปอร์เซ็นต์ครับ` |
| 4 | 90% | `ทำให้มันประโยชน์ได้เลยนะ` | `ถ้าอย่างนั้นผมขอตัวก่อนนะครับ` |
| 28 | 75% | `รีเอชันกัน` | `relation กัน` |

The corpus result from that log:

```text
corpus CER=22.9%
WER=73.7%
infer=37.5s
```

### Checkpoint Probe

Probe logs suggest checkpoint 400 was the best of the inspected adapter
checkpoints on the small probe set:

| Checkpoint | CER | WER-like |
| ---: | ---: | ---: |
| 300 | 22.5% | 78.9% |
| 400 | 20.9-21.7% | 72.5-72.6% |
| 500 | 22.9% | 74.7% |
| 600 | 25.2% | 72.5% |
| 800 | 25.5% | 72.5% |
| 842 | 23.6-25.7% | 72.5-73.7% |

This is a warning sign: `--load-best-model-at-end` selected by eval loss, but
small generation probes favor an earlier checkpoint. The model-selection signal
is still not aligned with actual generation quality.

## Runtime Changes Around M2

The v0 runtime currently defaults to M2 if the CT2 artifact exists:

```text
training/whisper/runs/largev3-m2-iter1-ct2-int8_float16
```

The faster-whisper wrapper now uses a sparse temperature ladder:

```python
temperature=[0.0, 0.8, 1.0]
```

Reason: some short utterances fall into degenerate greedy loops. The default
six-rung fallback ladder spends too much time on intermediate temperatures that
do not rescue those clips. M2 fallback scan observed:

```text
[largev3-m2] clip 28 (1.2s) fell back to temp=0.8
[largev3-m2] fallback clips: 1/32, total infer=36.7s
[largev3-iter4] fallback clips: 0/32, total infer=23.7s
```

Thai spacing is also normalized after decoding to collapse word-by-word spaced
Thai output when it appears to be a fine-tune label artifact.

Runtime context policy after the context benchmark:

- Rolling ASR prompt memory is enabled by default. It keeps a bounded transcript
  tail and filtered English/domain terms, then builds a per-utterance
  `initial_prompt`.
- Left-audio retry is implemented but disabled by default. Enable it only for
  experiments with `asr_left_context_enabled` or `--asr-left-context`.
- Delayed ASR correction is implemented but disabled by default. Enable it only
  for experiments with `asr_delayed_correction_enabled` or
  `--asr-delayed-correction`.
- Tiny VAD utterance merge/defer is implemented but disabled by default. Enable
  it only for experiments with `asr_tiny_utterance_merge_enabled` or
  `--asr-tiny-merge`.

## Limitations And Flaws

### 1. M2 is slower

M2 runs at about `2.9x realtime` on the benchmark, while `largev3-iter4` runs
around `4.8-4.9x realtime`. It may still be fast enough for near-real-time
streaming, but the latency budget is tighter.

### 2. CER is not clearly better than the previous large-v3 model

The corrected M2 CER is `22.9%`; the previous large-v3 iter4 benchmark is
`21.2%`. M2's WER-like metric improved, but Thai WER is unreliable because word
boundaries and spacing are inconsistent.

### 3. Generation quality still disagrees with eval loss

The training run finished with lower eval loss at the end, but probe logs favor
checkpoint 400 over the final/best-loss checkpoint. This repeats an earlier
lesson from the large-v3 experiments: teacher-forced loss is not enough for model
selection. Generation-based eval must be part of the gate.

### 4. Short utterances are fragile

Several worst clips are very short or terse, such as `อธิบาย`, `อ๋อ ครับ`, and
`เปอร์เซ็นต์ครับ`. M2 can misrecognize them completely, and at least one
1.2-second clip needed temperature fallback.

### 5. English technical terms are inconsistent

Examples:

- `relation กัน` became `รีเอชันกัน`
- `เปอร์เซ็นต์ครับ` became `อ่า`

This matters because Daiya's target domain is technical Thai-English speech.

### 6. The old pre-M2 dataset was intentionally replaced

The relabel log says the original M2 relabel run wrote to:

```text
C:\JokaMain\ProjectShowRoom\daiya-rmt\trainning\dataset\hf_datasets\whisper-m2
```

That separate M2 directory no longer exists. This is intentional: the M2 dataset
replaced the old dataset at:

```text
training/dataset/hf_datasets/whisper
```

The old pre-M2 dataset was removed because it consumed more than 10 GB and is not
expected to be reused. Reproducing pre-M2 comparisons therefore depends on the
saved model artifacts and logs, not on rerunning against the old labels.

### 7. Relabeling had decode warnings

`m2-relabel.log` contains multiple ALAC decode errors before the pipeline
finished. The relabel run did end with `RELABEL-DONE`, but the warnings should
be investigated before treating M2 labels as clean.

## Recommended Next Steps

1. Record dataset stats for the current canonical M2 dataset at
   `training/dataset/hf_datasets/whisper` so future comparisons do not depend on
   remembering that it replaced the old labels.
2. Add a generation-based validation gate for training, preferably CER plus
   targeted short-utterance probes, and do not select by eval loss alone.
3. Compare checkpoint 400 against the current final M2 CT2 artifact on a larger
   held-out set.
4. Re-run benchmarks with the same language hint, temperature ladder, and Thai
   spacing normalization for all compared models.
5. Add a technical-term probe set covering `enum`, `status`, `relation`,
   `percent`, `document`, and common Thai-English engineering phrases.
6. Decide whether M2 should remain the runtime default only after the larger
   CER/probe gate. For now, treat it as experimental default rather than a
   settled production-quality replacement.
