# PR6: Word-Level Speaker Assignment and Mux Boundary Hardening

Date: 2026-07-08

## Scope

This PR changes only runtime mux behavior. It does not change ASR decoding,
checkpoint selection, prompt-conditioned training, VAD, lookahead, or phonetic
normalization.

The mux now assigns speakers to individual `WordTimestamp` entries when word
timestamps are present, then derives the segment speaker from word-level
coverage. Segments without word timestamps keep the previous segment-overlap
fallback. A small nearest-turn collar covers short boundary words that land just
outside a diarization turn.

## Wire Shape

Transcript payloads keep the existing segment-level `speaker` field. Word
payloads may now include an optional `speaker` field:

```json
{
  "word": "ครับ",
  "start": 1.3,
  "end": 1.75,
  "probability": null,
  "speaker": "SPEAKER_002"
}
```

Words with no speaker assignment omit `speaker`. The frontend currently stores
`words` opaquely and does not render per-word speakers, so this is a compatible
payload extension.

## Synthetic Results

The focused mux tests cover these before/after expectations:

| Case | Old behavior | New behavior |
| --- | --- | --- |
| Segment spans a boundary, but word coverage favors the later speaker | Segment speaker chosen by whole-segment overlap | Segment speaker chosen from word coverage; each word carries its own speaker |
| Short backchannel just after a turn boundary, e.g. `อ๋อ` | `UNKNOWN` if it misses all turn overlap | Assigned to nearest turn when within the 120 ms collar |
| Provisional diarization differs from committed diarization | Segment label could update, but words had no speaker labels | Final event reassigns word speakers from committed turns |
| Committed turn correction changes only a minority word speaker | No event when segment speaker stays the same | Emits `transcript.update` because word metadata changed |
| ASR segment has no word timestamps | Segment-overlap fallback | Same fallback behavior |

## Commands Run

Focused mux regression suite:

```powershell
$env:PYTHONPATH='daiya/src'; python -m unittest discover -s daiya/tests -p 'test_mux.py'
```

Output:

```text
............
----------------------------------------------------------------------
Ran 12 tests in 0.001s

OK
```

Broader unittest discovery with system Python:

```powershell
$env:PYTHONPATH='daiya/src'; python -m unittest discover -s daiya/tests -p 'test_*.py'
```

Output summary:

```text
FAILED (errors=2, skipped=1)
```

Blocker: the system Python environment does not have `numpy`, so
`test_asr.py` and `test_pipeline.py` cannot import.

Project environment, after linking this worktree's `lab/pyannote` to the main
worktree checkout:

```powershell
uv run --project daiya python -m unittest discover -s daiya/tests -p 'test_*.py'
```

Output:

```text
..........................
----------------------------------------------------------------------
Ran 26 tests in 2.662s

OK
```

Lab diarization unit tests:

```powershell
uv run --package statefull-diarization --directory lab\statefull-diarization python -m unittest
```

Output:

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.006s

OK
```

Stage 6 diarization benchmark:

```powershell
uv run --package statefull-diarization --directory lab\statefull-diarization python benchmark_stage6.py `
  --audio C:\JokaMain\ProjectShowRoom\daiya-rmt\training\resources\Th-En_sample_02.mp3 `
  --backends daiya `
  --profile balanced `
  --max-duration 180 `
  --output-dir artifacts\stage6-benchmark-pr6
```

Output:

```text
=== Th-En_sample_02.mp3 (180.0s) ===
offline-reference runtime=7.012s speakers=3 turns=51
daiya: DER=0.306 runtime=12.488s RTF=0.069 ref_speakers=3 hyp_speakers=11 flips=9 pipeline_p50=0.072s emit_latency_p50=2.000s
report=artifacts\stage6-benchmark-pr6\Th-En_sample_02-balanced.json
summary=artifacts\stage6-benchmark-pr6\summary-balanced.json
```

Benchmark artifacts:

- `lab/statefull-diarization/artifacts/stage6-benchmark-pr6/Th-En_sample_02-balanced.json`
- `lab/statefull-diarization/artifacts/stage6-benchmark-pr6/summary-balanced.json`
- `lab/statefull-diarization/artifacts/stage6-benchmark-pr6/Th-En_sample_02-balanced-offline-reference.rttm`
- `lab/statefull-diarization/artifacts/stage6-benchmark-pr6/Th-En_sample_02-daiya-balanced.csv`

The artifact directory is ignored by Git.

## Interpretation

This change should not affect raw ASR CER because decoded text is untouched. The
expected gain is transcript readability and speaker attribution around turn
boundaries, especially for short acknowledgements and backchannels common in
Thai-English and Japanese-English meetings. Real impact still depends on word
timestamp quality after LoRA merge and CTranslate2 conversion, which remains a
separate validation task. The Stage 6 diarization benchmark is useful as a
runtime sanity check, but it does not directly measure this PR's per-word mux
assignment because it benchmarks diarization output rather than ASR+diarization
transcript event alignment.

## Claude QA Notes

Claude Opus QA was run read-only with `--model opus --effort xhigh`. It found no
blocking issue, but suggested:

- Preserve speaker labels for text-only corrections. Addressed by keeping the
  current segment speaker unless a correction explicitly supplies `speaker_id`
  or replacement `words`.
- Add tests for correction paths. Addressed with text-only and word-replacement
  correction tests.
- Keep the new overlap floor from changing words-less segment fallback.
  Addressed by applying `min_word_overlap_seconds` only to word assignment and
  preserving any positive overlap for segment fallback.
- Add a leading-edge collar case. Addressed with a short backchannel before the
  nearest turn.
