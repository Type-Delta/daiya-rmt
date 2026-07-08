# Right-Context Lookahead ASR Finalization Experiment

Date: 2026-07-08

## Goal

Evaluate whether a small amount of future audio helps ASR finalization when VAD
chunks cut trailing boundary words or very short utterances. This is a
runtime-only experiment and stays default-off.

## Runtime Strategy

The experimental strategy is `decode_with_right_context`:

- Hold a completed utterance until either `asr_right_context_seconds` of future
  audio is available or `asr_right_context_max_latency_seconds` is reached.
- Decode the target utterance plus the available right-context audio.
- Clip converted ASR segments and word timestamps back to the original utterance
  window before muxing.

This path does not emit a preliminary ASR partial before the lookahead wait. It
therefore adds up to the configured finalization latency for opted-in runs. That
is acceptable for this default-off experiment, but it should not be enabled by
default without benchmark evidence that the accuracy gain is worth the delay.

Default config remains unchanged:

```text
asr_right_context_enabled = False
asr_right_context_seconds = 0.8
asr_right_context_max_latency_seconds = 1.0
asr_right_context_strategy = "decode_with_right_context"
```

CLI opt-in:

```powershell
uv run --no-sources --package daiya daiya path/to/audio.wav `
  --asr-model training/whisper/runs/largev3-m2-iter1-ct2-int8_float16 `
  --language th `
  --asr-right-context `
  --asr-right-context-seconds 0.8 `
  --asr-right-context-max-latency-seconds 1.0
```

## Benchmark Command

Primary comparison:

```powershell
uv run --no-sources --package daiya python lab/asr_eval.py `
  --model training/whisper/runs/largev3-m2-iter1-ct2-int8_float16 `
  --limit 32 `
  --device auto `
  --compute-type int8_float16 `
  --language th `
  --no-condition-on-previous-text `
  --benchmark-strategies isolated,right_audio_context `
  --right-audio-context-seconds 0.8
```

Optional context-strategy comparison against the previous runtime experiment:

```powershell
uv run --no-sources --package daiya python lab/asr_eval.py `
  --model training/whisper/runs/largev3-m2-iter1-ct2-int8_float16 `
  --limit 32 `
  --device auto `
  --compute-type int8_float16 `
  --language th `
  --no-condition-on-previous-text `
  --benchmark-strategies isolated,rolling_initial_prompt,left_audio_context,right_audio_context `
  --right-audio-context-seconds 0.8
```

The eval summary reports overall metrics plus `short_utterance_subset` and
`english_technical_term_subset` for every strategy. Each `right_audio_context`
detail row records the borrowed following rows/files and borrowed audio seconds.

Note: introducing the shared eval timestamp-window helper changes exact-boundary
behavior for `left_audio_context` from the earlier inline check: words or
segments ending exactly at the left-context cutoff are now excluded. This matches
the runtime clipping convention (`end <= clip_start` is outside the target
window), but prior `left_audio_context` numbers should not be mixed with new
runs without noting that change.

## Local Run Status

The benchmark inputs were linked from the main worktree:

```powershell
New-Item -ItemType Junction `
  -Path training/whisper/runs/largev3-m2-iter1-ct2-int8_float16 `
  -Target C:/JokaMain/ProjectShowRoom/daiya-rmt/training/whisper/runs/largev3-m2-iter1-ct2-int8_float16

New-Item -ItemType Junction `
  -Path training/dataset/hf_datasets/whisper/train `
  -Target C:/JokaMain/ProjectShowRoom/daiya-rmt/training/dataset/hf_datasets/whisper/train

New-Item -ItemType HardLink `
  -Path training/dataset/hf_datasets/whisper/metadata.jsonl `
  -Target C:/JokaMain/ProjectShowRoom/daiya-rmt/training/dataset/hf_datasets/whisper/metadata.jsonl
```

The documented `--device auto --compute-type int8_float16` command failed on
this machine because faster-whisper selected CUDA but `cublas64_12.dll` was not
available. The benchmark was rerun on CPU:

```powershell
uv run --no-sources --package daiya python -u lab/asr_eval.py `
  --model training/whisper/runs/largev3-m2-iter1-ct2-int8_float16 `
  --limit 32 `
  --device cpu `
  --compute-type int8 `
  --language th `
  --no-condition-on-previous-text `
  --benchmark-strategies isolated,right_audio_context `
  --right-audio-context-seconds 0.8
```

Artifacts:

```text
lab/artifacts/asr_eval/summary_20260708T133942Z_training_whisper_runs_largev3-m2-iter1-ct2-int8_float16_benchmark_isolated_right_audio_context.json
lab/artifacts/asr_eval/details_20260708T133942Z_training_whisper_runs_largev3-m2-iter1-ct2-int8_float16_benchmark_isolated_right_audio_context.jsonl
```

Results:

| Strategy | Rows | Micro CER | Mean CER | Micro WER-like | Mean WER-like | Short micro CER | Technical-term micro CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `isolated` | 32 | 14.79% | 14.79% | 15.21% | 15.63% | 16.81% | 14.79% |
| `right_audio_context` | 32 | 17.96% | 18.89% | 17.63% | 19.85% | 21.57% | 17.96% |

## Verification

Pure unit tests were run with workspace sources disabled because the default
workspace resolution references a missing local `lab/pyannote` distribution.

```powershell
$env:PYTHONPATH='daiya/src'
uv run --no-sources --package daiya python -m unittest discover -s daiya/tests -p 'test_asr.py'
# Ran 5 tests in 0.002s
# OK

$env:PYTHONPATH='daiya/src'
uv run --no-sources --package daiya python -m unittest discover -s daiya/tests -p 'test_pipeline.py'
# Ran 13 tests
# OK

$env:PYTHONPATH='daiya/src'
uv run --no-sources --package daiya python -m unittest discover -s daiya/tests
# Ran 24 tests in 1.018s
# OK

uv run --no-sources --package daiya python -m py_compile lab/asr_eval.py
# OK

uv run --no-sources --package daiya python lab/asr_eval.py --help |
  Select-String -Pattern 'right_audio_context|right-audio-context|benchmark-strategies'
# Shows right_audio_context and --right-audio-context-seconds.

git diff --check
# OK, with Git CRLF conversion warnings only.
```

## Interpretation

On this 32-row CPU benchmark, `right_audio_context` regressed versus isolated
decode: overall micro CER worsened by 3.17 percentage points, and the short
utterance subset regressed by 4.76 percentage points. This does not justify
enabling right-context finalization by default. The implementation should remain
an opt-in experiment for follow-up runs with different lookahead values or
datasets.

## QA Notes

Claude Opus was run as a read-only QA reviewer with:

```powershell
claude -p --model opus --effort xhigh --permission-mode dontAsk `
  --allowedTools Read --allowedTools Glob --allowedTools Grep
```

Handled findings:

- Both `asr_left_context_enabled` and `asr_right_context_enabled` could cause
  short utterances to decode with right context and then be overwritten by the
  left-context retry. Fixed by making left-context retry stand down when
  right-context is enabled.
- Added tests for draining pending right-context utterances on `flush()` and for
  the lab `right_audio_context` borrowing loop.
- Aligned the lab eval default `--right-audio-context-seconds` with runtime at
  0.8 seconds.
- Documented the no-preliminary-partial latency tradeoff and the exact-boundary
  scoring note for `left_audio_context`.

Residual risk from QA: diarization-on timing with deferred ASR is not covered by
a dedicated integration test. The mux assigns speakers by segment timestamps, so
late ASR arrival is expected to remain coherent, but this should be watched in
live replay tests.
