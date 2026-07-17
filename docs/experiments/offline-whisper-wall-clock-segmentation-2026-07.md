# Offline Whisper wall-clock segmentation — July 2026

## Decision

Regenerate the offline Whisper dataset into a **fresh** output directory before
more manual labeling. Do not replace the current dataset in place and do not
copy any existing label or human review automatically.

The previous processor could (1) subtract padded pyannote overlap intervals
from speech and (2) concatenate disjoint VAD islands. Either operation changes
the spoken audio presented to the labeling model. The new processor preserves
one contiguous source-time window, treats overlap as review evidence, and uses
silence-aware long-window boundaries. Its default VAD profile remains
`threshold=0.5`, `min_speech=250 ms`, `min_silence=150 ms`, `pad=80 ms`; the
benchmark selected the profile based on sentence completeness/recall rather
than live speaker-turn latency.

## Reproduction

The benchmark used `Th-En_sample_11.m4a`, source time `3600–4200 s`, matched
to the 109 human reference spans in
`training/dataset/manual-label/m2-label-ref/ref_labels.txt`. CPU was used for
repeatability. Splitting the input into contiguous 60-second WAVs avoids
putting a ten-minute waveform into one VAD call; the benchmark restores each
segment's absolute offset before scoring.

```powershell
$root = 'C:\JokaMain\ProjectShowRoom\daiya-rmt'
$tmp = Join-Path $env:TEMP 'daiya-wall-clock-benchmark'
New-Item -ItemType Directory -Force $tmp | Out-Null

ffmpeg -ss 3600 -t 600 -i "$root\training\dataset\raw\whisper\Th-En_sample_11.m4a" `
  -ac 1 -ar 16000 -c:a pcm_s16le -y "$tmp\Th-En_sample_11_3600_4200.wav"
0..9 | ForEach-Object {
  ffmpeg -ss ($_ * 60) -t 60 -i "$tmp\Th-En_sample_11_3600_4200.wav" `
    -ac 1 -ar 16000 -c:a pcm_s16le -y "$tmp\sample11-part-$('{0:D2}' -f $_).wav"
}

cd $root
$parts = Get-ChildItem $tmp -Filter 'sample11-part-*.wav' | Sort-Object Name | Select-Object -ExpandProperty FullName
uv run --project training/processor/whisper python training/processor/whisper/scripts/benchmark_segmentation.py `
  @parts `
  --reference-labels "$root\training\dataset\manual-label\m2-label-ref\ref_labels.txt" `
  --source-offset-seconds 3600 --raw-source-name Th-En_sample_11.m4a `
  --old-metadata "$root\training\dataset\hf_datasets\whisper\metadata.jsonl" `
  --old-reviews "$root\training\processor\whisper\web\human-reviews\reviews.jsonl" `
  --output "$tmp\sample11-wall-clock.json" --device cpu
```

`legacy_spliced` below reimplements the old VAD grouping/export behavior using
the same VAD intervals; `wall_clock` uses the new window builder. The old
pyannote overlap deletion is not included in the aggregate numbers because the
reference window has no portable human overlap annotation. It is covered by
focused regression tests and metadata evidence instead.

## Results

| Silero profile | Mode | chunks | p50 / p95 seconds | missed reference speech (s) | duplicate window seconds | review-only fallback rows | boundary F1 | unprotected boundary-in-speech proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final/default: 0.50 / 250 / 150 / 80 ms | legacy spliced | 56 | 9.0 / 18.5 | 0.8 | 0 | 0 | 0.4788 | 0.1273 |
| final/default: 0.50 / 250 / 150 / 80 ms | wall-clock | 51 | 9.4 / 25.0 | **0.0** | 5.0 | 10 | 0.4562 | **0.0000** |
| PR #10 sensitive: 0.40 / 200 / 300 / 50 ms | legacy spliced | 54 | 9.0 / 23.9 | 2.3 | 0 | 0 | 0.4540 | 0.0943 |
| PR #10 sensitive: 0.40 / 200 / 300 / 50 ms | wall-clock | 50 | 9.0 / 25.0 | **0.0** | 5.0 | 10 | 0.4277 | **0.0000** |
| PR #10 balanced: 0.50 / 200 / 500 / 100 ms | legacy spliced | 49 | 10.9 / 25.0 | 0.1 | 0 | 0 | 0.4620 | 0.0625 |
| PR #10 balanced: 0.50 / 200 / 500 / 100 ms | wall-clock | 44 | 11.7 / 25.0 | **0.0** | 6.0 | 11 | 0.3856 | **0.0000** |

The final/default row had the fewest context-fallback rows among profiles that
achieved zero missed reference speech in the new builder. It intentionally uses
the same VAD parameters as the pre-change baseline; the measured change is the
wall-clock window builder, not a claimed VAD improvement. Its five seconds of
duplicate audio are deliberate, bounded one-second context at **five**
no-silence handoffs; both neighboring rows are marked review-only, yielding ten
flagged rows. Those rows have `training_eligible=false` and review signals, so
they cannot quietly become duplicate training targets.

The collar-based boundary F1 is not a sentence-quality score: the human
references are short manually labeled speech spans while the target windows are
up to 25 seconds. It is retained as a diagnostic only. The more relevant
boundary-word proxy counted only boundaries not protected by that window's own
trailing context inside a human speech span after a 250 ms collar. Four default
handoffs were speech-adjacent but protected by explicit trailing context; no
unprotected handoff fell in reference speech.

The existing dataset has 107 source-11 rows overlapping this source window,
whereas the final regenerated configuration produces 51 windows (a 56-row
count delta, not a safe one-to-one mapping). Those legacy rows report 8.334
seconds of removed-timeline proxy (`source_end - source_start -
speech_duration`), including rows `00663` (0.800 s), `00647` (0.600 s), and
`00657` (0.500 s). These are priority migration-review cases, especially where
their text density is also high.

## Boundary inspection

Waveforms were inspected from the raw, continuous source window rather than
from exported chunks:

- Local `28.8 s` (raw `3628.8 s`): the final window ends at `28.8 s`; the
  next begins at `29.6 s`. The waveform shows a quiet boundary region, so no
  VAD-island concatenation is needed.
- Local `43.7 s` (raw `3643.7 s`): the window ends before the following
  `45.1 s` speech start; the source pause remains a source-timeline pause.
- Local `70.1 s` (raw `3670.1 s`): no suitable earlier silence was available
  within the long-window search. The windows `[45.1, 70.1]` and `[69.1, 70.2]`
  retain one second of shared context. This handoff is visible as a fallback,
  not silently deduplicated or cut through an unlabelled word.

All 428 append-only human reviews were included in the audit input; none belong
to this source-11 reference window. The migration utility still verifies every
review by exported audio SHA-256 when a full regeneration is available.

## Migration procedure

After regeneration, run this report before exposing the new dataset to the
review workbench or training:

```powershell
uv run --project training/processor/whisper python training/processor/whisper/scripts/map_segmentation_reviews.py `
  C:\datasets\old\metadata.jsonl C:\datasets\regenerated\metadata.jsonl `
  --old-audio-root C:\datasets\old --new-audio-root C:\datasets\regenerated `
  --old-reviews C:\JokaMain\ProjectShowRoom\daiya-rmt\training\processor\whisper\web\human-reviews\reviews.jsonl `
  --output C:\datasets\regenerated\segmentation-migration.json
```

Only matching source spans **and** byte-identical exported WAV SHA-256 hashes
are classified as `unchanged` or `safely_reusable_review`. Changed audio is
`changed_boundary_needs_relabel`; fan-out or insufficient evidence is
`ambiguous`. The report copies neither label text nor review decisions.

## Limitations

- This is one 10-minute human-transcribed Thai-English window; it does not
  establish performance for every recording, language pair, or overlap type.
- No CER or LLM-label-quality claim is made because labels were not regenerated
  over identical source spans.
- There was no human overlap annotation or reviewed clip in this particular
  source-11 window. Overlap preservation, source bounds, false-negative gaps,
  and review mapping are covered by regression tests; future regenerations
  should prioritize overlap-marked and review-marked rows in the mapping report.
