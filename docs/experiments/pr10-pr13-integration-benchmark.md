# PR #10 + PR #13 Integration Benchmark

## Decision

Keep both changes, with the current runtime default unchanged: **merge #13 and
merge #10 with Energy VAD still the default and stateful Silero opt-in**. The
combined branch is test-clean, preserves the context-aware runtime from merged
PR #8, and has no observed integration regression.

The short full-audio run strengthens the case for the PR #10 recommended
Silero profile at speaker boundaries. It does not yet justify changing the
default or claim a speaker-attributed WER improvement: the available speaker
reference is offline-pyannote RTTM, not human word-level attribution.

## Integration

Baseline: `origin/main` at `e7417b3` (PR #8 already merged).

Integration branch: `feat/silero-word-mux-integration-benchmark`.

Merge order:

```powershell
git switch -C feat/silero-word-mux-integration-benchmark origin/main
git merge --no-ff origin/feat/silero-vad-bakeoff
git merge --no-ff origin/feat/word-speaker-mux-boundaries
```

PR #10 conflicted with PR #8 in the CLI and ASR/pipeline tests. The resolved
tree retains both the rolling-prompt/context options and the VAD backend
options. PR #13 merged cleanly after that resolution.

## Tests

Ran with the main-worktree environment and this worktree's source tree:

```powershell
$env:PYTHONPATH = "$PWD/daiya/src"
$python = 'C:/JokaMain/ProjectShowRoom/daiya-rmt/.venv/Scripts/python.exe'
& $python -m unittest discover -s daiya/tests -p 'test_mux.py'
& $python -m unittest discover -s daiya/tests -p 'test_asr.py'
& $python -m unittest discover -s daiya/tests -p 'test_pipeline.py'
& $python -m unittest discover -s daiya/tests -p 'test_vad_bakeoff.py'
& $python -m unittest discover -s daiya/tests -p 'test_server.py'
& $python -m daiya.cli --help
```

| suite | result |
| --- | ---: |
| mux | 12 passed |
| ASR/VAD | 13 passed |
| pipeline | 11 passed |
| VAD bake-off | 8 passed |
| server | 2 passed |
| CLI import / `--help` smoke | passed |

The mux coverage includes word assignment, boundary collars, committed-turn
updates, correction paths, and the words-less fallback. The VAD coverage uses
fake stateful iterator events to exercise residual framing, state reset,
padding, short-speech filtering, and max-duration splitting.

## Full-audio VAD comparison

The input was a contiguous, unsegmented 180-second WAV derived from
`C:/JokaMain/ProjectShowRoom/daiya-rmt/training/resources/Th-En_sample_02.mp3`.
It was not assembled from pre-cut speech WAVs. The reference is the committed
51-turn offline-pyannote RTTM proxy, with a 250 ms boundary collar.

```powershell
$audio = 'lab/artifacts/integration-pr10-pr13/Th-En_sample_02.wav'
$rttm = 'lab/statefull-diarization/artifacts/stage6-benchmark-pr6/Th-En_sample_02-balanced-offline-reference.rttm'

& $python lab/vad_bakeoff.py $audio --chunk-seconds 0.10 `
  --backend energy --threshold 0.012 --min-speech-seconds 0.20 `
  --min-silence-seconds 0.45 --speech-padding-seconds 0 --max-utterance-seconds 8 `
  --reference-rttm $rttm --write-details --output-dir lab/artifacts/integration-pr10-pr13

& $python lab/vad_bakeoff.py $audio --chunk-seconds 0.10 `
  --backend silero --threshold 0.40 --min-speech-seconds 0.20 `
  --min-silence-seconds 0.30 --speech-padding-seconds 0.05 --max-utterance-seconds 8 `
  --reference-rttm $rttm --write-details --output-dir lab/artifacts/integration-pr10-pr13
```

| configuration | utterances | boundary F1 | missed ref speech | non-ref speech | duplicated | VAD RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main-compatible Energy | 29 | 0.2911 | 3.401s | 5.583s | 0.000s | 0.00043 |
| combined, stateful Silero sensitive | 38 | **0.4746** | **2.882s** | **4.149s** | 0.000s | 0.02289 |

Silero improves the proxy boundary F1 by 0.1834 absolute and reduces missed
reference speech by 0.519 seconds. It creates more, shorter utterances (mean
4.383s vs 5.776s) and is about 53x slower than Energy VAD, but at 0.023 RTF
is still far below real time. This reproduces the directional result in the
PR #10 full-audio report without relying on already-segmented clips.

## Speaker timeline smoke

```powershell
& $python lab/statefull-diarization/benchmark_stage6.py `
  --audio C:/JokaMain/ProjectShowRoom/daiya-rmt/training/resources/Th-En_sample_02.mp3 `
  --backends daiya --profile balanced --max-duration 180 `
  --output-dir lab/artifacts/integration-pr10-pr13/stage6
```

| metric | combined result |
| --- | ---: |
| DER (offline-pyannote proxy) | 0.3062 |
| speaker flips | 9 |
| reference / hypothesis turns | 51 / 40 |
| corrections | 9 |
| realtime factor | 0.0857 |

This is a pipeline compatibility smoke, not a direct word-speaker score: the
stage-6 diarization benchmark does not run ASR word timestamps through the
mux. The functional mux tests cover that path; a future decision-changing
benchmark needs human speaker-attributed transcript words to measure SA-WER,
word speaker flips, and boundary-word accuracy directly.

## Artifacts

Ignored runtime artifacts are under:

```text
lab/artifacts/integration-pr10-pr13/
  Th-En_sample_02.wav
  vad_bakeoff_20260710T040405Z.{csv,jsonl}
  vad_bakeoff_20260710T040409Z.{csv,jsonl}
  vad_bakeoff_details_20260710T040405Z.jsonl
  vad_bakeoff_details_20260710T040409Z.jsonl
  stage6/summary-balanced.json
```

## Limitations

- The VAD transcript/CER hook was intentionally not run: this 180-second
  recording has no aligned human text reference in the worktree.
- The RTTM is an offline-pyannote proxy rather than human speaker-turn truth.
- PR #13 changes mux attribution, not the diarizer. Its direct metrics remain
  unavailable until a word-level speaker reference is added.
