# Timestamp-informed Whisper ownership segmentation — July 2026

## Decision

Use `timestamp-ownership-v1` for newly generated offline Whisper datasets.
It keeps a training row's owned source interval disjoint from every other
eligible row. A next-row one-sided pre-roll is a labeling-only input; it is
exported as a separate owned training crop and remains review-only until the
local-ASR ownership gate accepts its target-only label.

The audio-labeling LLM remains text-only. This experiment used local
Faster-Whisper word/phrase timings only for acoustic boundary evidence and
post-label validation; it neither requested nor trusted LLM timestamps.

## Method

The two runs used the current local model
`checkpoint-588-ct2-int8_float16` on CUDA (`int8_float16`), Faster-Whisper
1.2.1, `word_timestamps=true`, beam size 5, no prior-text conditioning, plus
Silero VAD and 50 ms waveform-energy evidence. The cache recorded a full model
artifact fingerprint, source SHA-256, decode settings, acoustic settings, and
failure status outside the dataset export.

The comparison reimplements wall-clock-v2 as the baseline. The timestamp
segmenter plans for 18 seconds, treats 25 seconds as a soft goal, and may
extend to its configured 30-second hard ceiling only to reach a grounded
boundary. Both metrics score only source-time ownership; duplicated labeler
input is separate from duplicated eligible training audio.

## Results

### Source 11, local 3600–4200 s (10 minutes)

This run used the available 109 human reference spans, with 458.659 seconds of
reference speech. “Missed” is limited to those supplied spans.

| Metric | wall-clock-v2 | timestamp ownership |
| --- | ---: | ---: |
| chunks | 50 | 48 |
| duration p50 / p95 / max (s) | 9.80 / 25.00 / 25.00 | 10.25 / 25.60 / 26.74 |
| retained / missed reference speech (s) | 458.659 / 0.000 | 458.659 / 0.000 |
| unprotected boundaries inside reference speech | 4 | 2 |
| fallback handoffs / fallback rows | 4 / 8 | 0 / 0 |
| duplicated labeling audio (s) | 4.000 | 0.000 |
| duplicated eligible-training audio (s) | 0.000 | 0.000 |

Thirty rows changed. Timestamp evidence decoded successfully (3,532 timestamp
items; 177 energy gaps). The window had evidence coverage of 1.00 across
internal boundaries, with 47 single-signal boundaries and no disagreements;
that is an evidence distribution, not a claim that a single signal is always
sufficient. The model safely used a 26.74-second maximum below the 30-second
hard cap.

### Source 01, local 0–120 s (supplemental difficult window)

No human reference spans were available for this window, so no retained/missed
speech or in-reference boundary claims are made.

| Metric | wall-clock-v2 | timestamp ownership |
| --- | ---: | ---: |
| chunks | 8 | 6 |
| duration p50 / p95 / max (s) | 18.825 / 25.000 / 25.000 | 22.481 / 28.300 / 28.300 |
| fallback handoffs / fallback rows | 2 / 4 | 1 / 1 |
| duplicated labeling audio (s) | 2.000 | 1.000 |
| duplicated eligible-training audio (s) | 0.000 | 0.000 |

Six rows changed. The real model produced 705 timestamp items and 36 energy
gaps; boundary evidence coverage was 0.80 (one multi-signal, three
single-signal, and one disagreement/no-signal boundary).

## Boundary inspection

- Source 01’s `[74.900, 94.862]` owned interval ended at a
  `low_energy_gap+silero_vad_gap` boundary with confidence 0.951. It is a
  concrete independently corroborated acoustic boundary, not a transcript
  assertion.
- Source 01’s `[28.200, 44.900]` row heard labeling audio
  `[27.200, 44.900]`. Its method was
  `pre_roll_target_pending_alignment`, confidence 0.0: the preceding row kept
  its ownership, this row owns only from 28.200 seconds, and it cannot train
  until target-only alignment passes.
- In source 11, the timestamp run removed all four wall-clock fallback
  handoffs without reducing retained reference speech; its longer windows
  remained below the hard ceiling and its eligible-training duplication stayed
  exactly zero.

## Limitations and regeneration

- The zero missed-speech value applies only to the supplied source-11 human
  spans. It does not establish universal zero missed speech, label quality, or
  performance for every language pair and recording condition.
- Faster-Whisper may be imperfect for Thai and code-switching. Its text is
  never exported as a label; its timing only supports boundary and consistency
  decisions, with uncertain pre-roll rows held for review.
- No raw audio, model, evidence cache, or benchmark output is committed. To
  regenerate a dataset, configure `DAIYA_TIMESTAMP_MODEL` to the local
  Faster-Whisper CTranslate2 model and keep the timestamp cache outside the
  export directory. Regenerate into a fresh output directory, run the
  migration report, and resolve review-only rows before training. The trainer
  excludes `training_eligible=false` by default and requires an explicit
  legacy/research override otherwise.
