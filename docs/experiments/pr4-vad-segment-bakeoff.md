# PR4 Stateful Silero VAD Bake-Off

## Decision

Keep energy VAD as the default and Silero opt-in. Stateful Silero is promising,
but two recordings and an offline-pyannote RTTM are not enough evidence to
change the runtime default.

For the next streaming experiment, use the sensitive Silero row
(`threshold=0.4`, `min_speech=0.2s`, `min_silence=0.3s`, `pad=0.05s`,
`cap=8s`). It improved ASR CER and the speaker-boundary proxy while avoiding
the long segments produced by the ASR-best conservative row.

This report supersedes the earlier PR4 result that ran only energy VAD over 109
already-cut WAVs. That run measured re-segmentation of reference chunks, not
Silero or full-timeline streaming behavior, and is not evidence for backend
selection.

## Implementation

`SileroUtteranceSegmenter` uses `silero-vad==6.2.1` and its stateful
`VADIterator`. Arbitrary Daiya PCM chunks are buffered into the required
512-sample 16 kHz inference windows. Iterator/model state and sample offsets
persist across `accept()` calls. Padding, short-speech filtering, max-duration
splits, residual-frame flush, reset, discontinuities, and audio slicing are
handled by Daiya without resetting the iterator between chunks.

Normal runtime does not call `torch.hub`. The optional `vad` extra pins the
package and constrains Torch below the next major version:

```powershell
uv sync --project daiya --extra vad
```

Energy remains dependency-free and is used by default. `auto` and unavailable
explicit-Silero requests fall back safely; the bake-off marks a requested
Silero row as skipped if that fallback occurs, so it cannot be mislabeled as a
Silero result. Backend-specific threshold defaults are `0.012` for energy and
`0.5` for Silero; padding defaults are `0` and `0.1s`, respectively.

## Method

The corrected run used two continuous, unsegmented timeline inputs:

1. A 600-second window from `Th-En_sample_11.m4a`, source time
   `3600-4200s`. Its 109 human transcript/speech intervals cover
   `0-599.4s` in the extracted window.
2. The first 180 seconds of `Th-En_sample_02.mp3`, paired with the existing
   51-turn, 3-speaker offline-pyannote RTTM. This RTTM is a proxy, not human
   speaker-turn truth.

The 109 derived manual-label WAVs were not benchmark inputs. Both continuous
inputs were replayed as 100 ms Daiya chunks; Silero internally retained state
over 512-sample model windows.

Model and environment:

```text
M2 CT2: C:/JokaMain/ProjectShowRoom/daiya-rmt/training/whisper/runs/largev3-m2-iter1-ct2-int8_float16
Python: 3.12.10
faster-whisper: 1.2.1
silero-vad: 6.2.1
torch: 2.11.0+cu128
numpy: 2.4.6
```

CER is the primary transcript metric. `WER-like` is whitespace-token edit
rate and is weak for Thai, so it is reported only as a secondary signal.
Boundary matching uses a 250 ms collar. Coverage is interval-union based;
`duplicated seconds` detects overlapping emitted utterances.

## Reproduction

Create the two continuous evaluation inputs:

```powershell
$main = 'C:/JokaMain/ProjectShowRoom/daiya-rmt'
$out = 'lab/artifacts/vad_bakeoff/pr4_corrected'
New-Item -ItemType Directory -Force "$out/inputs" | Out-Null

ffmpeg -ss 3600 -t 600 `
  -i "$main/training/dataset/raw/whisper/Th-En_sample_11.m4a" `
  -ac 1 -ar 16000 -c:a pcm_s16le -y `
  "$out/inputs/Th-En_sample_11_manual_3600_4200.wav"

ffmpeg -t 180 `
  -i "$main/training/resources/Th-En_sample_02.mp3" `
  -ac 1 -ar 16000 -c:a pcm_s16le -y `
  "$out/inputs/Th-En_sample_02.wav"
```

`manual_refs.jsonl` is generated from
`training/dataset/manual-label/m2-label-ref/ref_labels.txt`, with
`file_name=Th-En_sample_11_manual_3600_4200.wav` and each header's local
`start`, `end`, and following transcript text.

Run each configuration row by substituting the values in the table below:

```powershell
$python = "$main/.venv/Scripts/python.exe"
$model = "$main/training/whisper/runs/largev3-m2-iter1-ct2-int8_float16"
$rttm = "$main/lab/statefull-diarization/artifacts/stage6-benchmark/Th-En_sample_02-balanced-offline-reference.rttm"

& $python lab/vad_bakeoff.py `
  "$out/inputs/Th-En_sample_11_manual_3600_4200.wav" `
  "$out/inputs/Th-En_sample_02.wav" `
  --chunk-seconds 0.10 `
  --backend silero `
  --threshold 0.40 `
  --min-speech-seconds 0.20 `
  --min-silence-seconds 0.30 `
  --speech-padding-seconds 0.05 `
  --max-utterance-seconds 8.0 `
  --reference-boundaries-jsonl "$out/manual_refs.jsonl" `
  --reference-rttm $rttm `
  --reference-text-jsonl "$out/manual_refs.jsonl" `
  --asr-model $model `
  --asr-device auto `
  --asr-compute-type int8_float16 `
  --write-details `
  --output-dir $out
```

Exact rows:

| name | backend | threshold | min speech | min silence | pad | cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| energy baseline | energy | 0.012 | 0.20 | 0.45 | 0.00 | 8 |
| Silero sensitive | silero | 0.40 | 0.20 | 0.30 | 0.05 | 8 |
| Silero balanced | silero | 0.50 | 0.20 | 0.50 | 0.10 | 8 |
| Silero conservative | silero | 0.60 | 0.35 | 0.70 | 0.15 | 12 |

## Results

Pooled across 780 seconds:

| row | utt. | selected sec | <1s | 1-2s | 2-5s | 5-8s | 8s+ | CER | WER-like | boundary F1 | missed ref sec | non-ref sec | VAD RTF | ASR RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| energy | 159 | 665.600 | 16 | 33 | 50 | 26 | 34 | 0.2691 | 0.7320 | 0.5678 | 22.707 | 64.130 | 0.0005 | 0.2836 |
| Silero sensitive | 179 | 678.396 | 24 | 42 | 58 | 21 | 34 | 0.2540 | 0.7431 | **0.6095** | 10.668 | 65.076 | 0.0100 | 0.2266 |
| Silero balanced | 152 | 700.516 | 14 | 27 | 45 | 17 | 49 | 0.2583 | 0.7044 | 0.5595 | 1.281 | 77.251 | 0.0101 | 0.2207 |
| Silero conservative | 102 | 718.222 | 2 | 10 | 30 | 17 | 43 | **0.2339** | **0.6409** | 0.4789 | **1.059** | 94.735 | **0.0100** | **0.2033** |

All rows emitted zero duplicated/overlapping seconds.

Separated boundary results:

| row | human transcript boundary F1 | human missed speech sec | RTTM proxy boundary F1 | RTTM missed speech sec |
| --- | ---: | ---: | ---: | ---: |
| energy | **0.6597** | 19.306 | 0.2911 | 3.401 |
| Silero sensitive | 0.6573 | 7.786 | **0.4746** | 2.882 |
| Silero balanced | 0.6467 | **0.676** | 0.2968 | 0.605 |
| Silero conservative | 0.5483 | 0.680 | 0.2878 | **0.379** |

Interpretation:

- Sensitive Silero improved CER by 0.0150 absolute over energy, reduced missed
  human-reference speech by 11.52 seconds, and substantially improved the
  RTTM-proxy boundary F1. It produced 20 more utterances and a few more
  sub-two-second segments.
- Balanced Silero nearly saturated reference speech coverage and improved
  CER, but its proxy boundary F1 was effectively tied with energy and it
  included 13.12 more seconds outside reference speech.
- Conservative Silero produced the best CER and WER-like scores and the fewest
  utterances, but its 12-second segments lost speaker-turn boundary recall.
  It is an ASR-oriented profile, not the recommended streaming/diarization
  profile.
- Silero startup plus streaming inference used about RTF 0.0100 versus
  energy's 0.0005. Both are well below real time; ASR remained the dominant
  cost.

## Artifacts

Ignored benchmark artifacts:

```text
lab/artifacts/vad_bakeoff/pr4_corrected/energy_baseline.{csv,jsonl}
lab/artifacts/vad_bakeoff/pr4_corrected/silero_sensitive.{csv,jsonl}
lab/artifacts/vad_bakeoff/pr4_corrected/silero_balanced.{csv,jsonl}
lab/artifacts/vad_bakeoff/pr4_corrected/silero_conservative.{csv,jsonl}
lab/artifacts/vad_bakeoff/pr4_corrected/*_details.jsonl
lab/artifacts/vad_bakeoff/pr4_corrected/per_dataset_metrics.json
lab/artifacts/vad_bakeoff/pr4_corrected/manual_refs.jsonl
```

## Limitations

- Only one human-transcribed 10-minute window was available.
- The sample-11 source offset is inferred from matching dataset metadata; no
  explicit manifest records the `+3600s` alignment.
- The sample-02 RTTM covers only 180 seconds and is offline pyannote output,
  not human speaker-turn ground truth. Boundary results are relative.
- CER compares concatenated ASR output with a cleaned human transcript and
  does not assign text to individual reference turns.
- This experiment measures VAD boundaries and their overlap with speaker
  turns; it does not rerun diarization or calculate DER for each VAD row.
