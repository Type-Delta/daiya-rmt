# PR4 VAD Segment-Size/Padding Bake-Off

This experiment compares utterance segmentation settings without requiring ASR
or diarization model downloads. The tool calls
`daiya.asr.create_utterance_segmenter(...)` when available and records the
actual segmenter backend used, so dependency-free energy fallback runs and
Silero-backed runs produce comparable rows.

## Lightweight Run

Use one or more local audio files:

```powershell
uv run python lab/vad_bakeoff.py .\path\to\sample.wav `
  --backend energy,auto,silero `
  --threshold 0.008,0.012 `
  --min-speech-seconds 0.20,0.35 `
  --min-silence-seconds 0.30,0.50 `
  --speech-padding-seconds 0.00,0.10,0.20 `
  --max-utterance-seconds 8.0,12.0
```

The script defaults to `--backend energy` so a lightweight run does not fetch
optional VAD assets. Pass `--backend silero` or `--backend energy,silero`
explicitly after installing the `vad` extra. Silero thresholds usually need a
different sweep range than energy thresholds; start around `0.3,0.5,0.7`.

Or sweep a directory:

```powershell
uv run python lab/vad_bakeoff.py --audio-glob "training/dataset/**/*.wav" --limit 20
```

Outputs are written under `lab/artifacts/vad_bakeoff/` by default:

- `vad_bakeoff_<stamp>.csv`
- `vad_bakeoff_<stamp>.jsonl`
- optional per-utterance details with `--write-details`

## Expected Row Shape

Each setting combination emits one comparable row:

```text
status,backend,segmenter_backend,threshold,min_speech_seconds,
min_silence_seconds,speech_padding_seconds,max_utterance_seconds,
audio_files,audio_seconds,utterance_count,total_speech_seconds,
mean_duration_seconds,p50_duration_seconds,p95_duration_seconds,
boundary_precision,boundary_recall,boundary_f1,asr_cer,asr_status,
elapsed_seconds,notes
```

Silero rows may show `segmenter_backend=energy` and a note that Silero fell
back to energy when the optional `vad` dependency path is unavailable. Energy
ignores speech padding; the Silero backend applies speech padding around
detected speech.

## Smoke Result

I ran a lightweight smoke test on a synthetic 16 kHz WAV with two tone regions
separated by 0.35 seconds of silence:

```powershell
$env:PYTHONPATH='daiya/src'
uv run --no-project --with numpy python lab/vad_bakeoff.py $env:TEMP\daiya-vad-bakeoff-smoke.wav `
  --backend energy `
  --threshold 0.012 `
  --min-speech-seconds 0.20 `
  --min-silence-seconds 0.30 `
  --speech-padding-seconds 0.00 `
  --max-utterance-seconds 8.0 `
  --output-dir $env:TEMP\daiya-vad-bakeoff-smoke
```

Output:

```text
energy actual=energy thr=0.012  silence=0.3   pad=0.0  n=1    speech=1.25     mean=1.25 p95=1.25 status=ok
```

## Optional Boundary Scoring

Boundary scoring does not load a diarization model. Provide reference speech
turns as JSONL:

```json
{"audio_path":"sample.wav","start":0.42,"end":2.80}
{"audio_path":"sample.wav","start":3.20,"end":5.10}
```

Run:

```powershell
uv run python lab/vad_bakeoff.py .\sample.wav `
  --reference-boundaries-jsonl .\refs\speech_boundaries.jsonl `
  --boundary-collar-seconds 0.25
```

RTTM `SPEAKER` rows are also accepted with `--reference-rttm`.

## Optional ASR CER Hook

CER scoring is skipped unless `--asr-model` and `--reference-text-jsonl` are
provided. Non-local model names are not resolved unless
`--allow-model-download` is passed.

Realistic local-model command if the CTranslate2 export exists:

```powershell
uv run python lab/vad_bakeoff.py .\path\to\sample.wav `
  --backend energy,silero `
  --threshold 0.008,0.012 `
  --reference-text-jsonl .\refs\full_transcripts.jsonl `
  --asr-model training/whisper/runs/largev3-m2-iter1-ct2-int8_float16 `
  --asr-device cuda `
  --asr-compute-type int8_float16
```

Reference text JSONL shape:

```json
{"audio_path":"sample.wav","text":"full reference transcript for this audio"}
```

## M2 Main-Model Benchmark

The local M2 CT2 artifact was routed from the main worktree:

```text
C:/JokaMain/ProjectShowRoom/daiya-rmt/training/whisper/runs/largev3-m2-iter1-ct2-int8_float16
```

Reference text JSONL was generated under the ignored artifacts directory from
`training/dataset/manual-label/m2-label-ref/ref_labels.txt`, matched to the 109
chunk WAVs in the main worktree's `m2-label-ref/audio/` directory.

Command:

```powershell
$env:PYTHONPATH='C:/Users/Kornnaras/.codex/worktrees/fe20/daiya-rmt/daiya/src'
$env:PYTHONIOENCODING='utf-8'
& 'C:/JokaMain/ProjectShowRoom/daiya-rmt/.venv/Scripts/python.exe' `
  'C:/Users/Kornnaras/.codex/worktrees/fe20/daiya-rmt/lab/vad_bakeoff.py' `
  --audio-glob 'C:/JokaMain/ProjectShowRoom/daiya-rmt/training/dataset/manual-label/m2-label-ref/audio/*.wav' `
  --backend energy `
  --threshold 0.012 `
  --min-speech-seconds 0.20 `
  --min-silence-seconds 0.30,0.50 `
  --speech-padding-seconds 0.00 `
  --max-utterance-seconds 8.0 `
  --reference-text-jsonl 'C:/Users/Kornnaras/.codex/worktrees/fe20/daiya-rmt/lab/artifacts/vad_bakeoff/m2_label_refs.jsonl' `
  --asr-model 'C:/JokaMain/ProjectShowRoom/daiya-rmt/training/whisper/runs/largev3-m2-iter1-ct2-int8_float16' `
  --asr-device auto `
  --asr-compute-type int8_float16 `
  --output-dir 'C:/Users/Kornnaras/.codex/worktrees/fe20/daiya-rmt/lab/artifacts/vad_bakeoff/m2_main_model'
```

Artifacts:

```text
lab/artifacts/vad_bakeoff/m2_main_model/vad_bakeoff_20260708T081219Z.csv
lab/artifacts/vad_bakeoff/m2_main_model/vad_bakeoff_20260708T081219Z.jsonl
```

Measured rows:

| backend | threshold | min speech | min silence | max utterance | files | audio sec | utterances | total speech sec | p50 | p95 | CER | elapsed sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| energy | 0.012 | 0.20 | 0.30 | 8.0 | 109 | 451.159 | 122 | 441.418 | 2.75 | 8.0 | 0.175265 | 150.993 |
| energy | 0.012 | 0.20 | 0.50 | 8.0 | 109 | 451.159 | 122 | 441.418 | 2.75 | 8.0 | 0.176828 | 142.012 |

Interpretation: on this chunked M2 reference set, changing energy trailing
silence from 0.30s to 0.50s did not change utterance count or duration
distribution. The 0.30s setting was slightly better on CER by about 0.0016
absolute. Boundary metrics are blank because no turn-boundary reference JSONL
or RTTM was provided for this run.

## Heavy Benchmark Blockers

Heavy ASR or diarization bake-offs cannot run in the lightweight path when:

- `faster-whisper` is not installed.
- The local model directory is missing.
- A remote model name is provided without `--allow-model-download`.
- Full-audio reference transcripts or boundary annotations are unavailable.
- CUDA, pyannote, or gated Hugging Face artifacts are required for separate
  diarization evaluation.

Those cases are reported as skipped in `asr_status` or left blank for boundary
metrics; segmentation summary rows still emit.
