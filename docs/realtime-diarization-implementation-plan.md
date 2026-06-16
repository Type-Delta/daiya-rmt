# Realtime Stateful Diarization Implementation Plan

This document is a fresh-thread handoff plan for improving realtime speaker
diarization in Daiya-RMT.

The current lab prototype lives in:

- `lab/statefull-diarization/demo.py`
- `lab/statefull-diarization/speaker_memory.py`

The current prototype is closer to an offline diarizer wrapped in repeated
chunks: it waits for a full chunk, runs pyannote's pipeline on that chunk, maps
local speaker labels to persistent speaker IDs, then prints one status line.
This works for experimentation, but it feels stale in realtime because output
cadence is tied to a long analysis chunk.

The new direction is to make Daiya a revision-oriented online diarizer:

- process audio continuously with a rolling analysis window
- emit provisional speaker turns quickly
- revise recent speaker labels when better overlapping context arrives
- commit older timeline regions after a fixed delay
- update speaker memory only once for committed, high-quality evidence

## Vocabulary

Offline diarizer:

- Receives a complete file or large completed chunk.
- Can inspect the whole chunk before producing output.
- Usually more accurate, but slow to display.

Online diarizer:

- Receives audio as it arrives.
- Maintains state across small updates.
- Emits output with bounded latency, such as 1-5 seconds.
- May revise recent output.

Daiya should behave like an online diarizer even if the first implementation
still calls pyannote's normal pipeline internally.

## References

diart is the main architecture reference:

- GitHub: https://github.com/juanmc2005/diart
- Paper: https://arxiv.org/abs/2109.06483
- Clustering docs: https://diart.readthedocs.io/en/latest/autoapi/diart/blocks/clustering/index.html
- Aggregation docs: https://diart.readthedocs.io/en/latest/autoapi/diart/blocks/aggregation/index.html

Important diart concepts to borrow:

- separate `duration`, `step`, and `latency`
- local speaker segmentation on a rolling buffer
- local-to-global speaker mapping with incremental clustering
- one-to-one assignment so two local speakers in the same window cannot map to
  the same global speaker
- overlap-aware embeddings
- delayed aggregation over overlapping windows

Important Daiya-specific additions:

- revision events for CLI/API output
- evidence-based speaker memory
- idempotent memory updates across overlapping windows
- later ASR/diarization multiplexer compatibility

## Current Problems

Current live loop:

- `CHUNK_SECONDS` controls how much audio is analyzed.
- `STRIDE_SECONDS` controls how much audio is dropped between analyses.
- `run_live()` waits until `buffer.shape[0] >= chunk_samples`.
- It runs `pipeline(...)` synchronously on the whole chunk.
- It calls `memory.assign(...)`, which can update centroids and speech totals.
- It prints only the latest turn status.

Problems:

1. Output cadence is too slow.
2. Cold start is too slow.
3. Overlapping windows would over-update memory if used naively.
4. The current output model is chunk-based, not timeline-based.
5. Current speaker profiles store a single centroid, not a reservoir of
   quality-scored evidence.
6. There is no correction/revision protocol, but Daiya's main product goal
   needs follow-up corrections anyway.

## Target Architecture

High-level flow:

```text
audio stream
  -> AudioRingBuffer
  -> RollingWindowScheduler
  -> DiarizationBackend
  -> SpeakerMemory.match/read-only assignment
  -> TimelineStore provisional updates
  -> Delayed commit horizon
  -> SpeakerMemory.commit_evidence/idempotent updates
  -> CLI/API event stream
```

Later flow with ASR:

```text
audio stream
  -> VAD / segmentation
  -> diarization timeline
  -> ASR partial/final text
  -> multiplexer aligns text spans to speaker spans
  -> context/LLM correction emits revision events
```

## Core Parameters

Replace the current two-parameter chunk model with:

```text
DIARIZATION_WINDOW_SECONDS
DIARIZATION_HOP_SECONDS
DIARIZATION_LATENCY_SECONDS
DIARIZATION_COMMIT_DELAY_SECONDS
```

Meaning:

- `window`: how much audio each diarization call analyzes.
- `hop`: how often a new analysis runs.
- `latency`: how far behind realtime the emitted slice is.
- `commit_delay`: how old a region must be before freezing it and updating
  durable speaker memory.

Initial profiles:

```text
balanced:
  window = 10.0
  hop = 1.0
  latency = 3.0
  commit_delay = 5.0

fast:
  window = 5.0
  hop = 0.5
  latency = 1.5
  commit_delay = 3.0

accuracy:
  window = 18.0
  hop = 2.0
  latency = 5.0
  commit_delay = 8.0
```

Start with `balanced`.

## Emitted Region Rule

Every hop, analyze the latest rolling window:

```text
window_start = now_audio_time - DIARIZATION_WINDOW_SECONDS
window_end = now_audio_time
```

But do not emit the whole window. Emit or revise only:

```text
emit_start = window_end - DIARIZATION_LATENCY_SECONDS
emit_end = emit_start + DIARIZATION_HOP_SECONDS
```

This is the diart-style delayed output contract.

The current pyannote backend may still produce a full-window annotation. Crop
that annotation to `[emit_start, emit_end]` before passing it to the timeline.

## Stage 0: Metrics And Replay Harness

Goal: make latency and instability measurable before tuning.

Add a small metrics recorder. It should work for live mode and replayed files.

Metrics to log per analysis hop:

```text
window_index
audio_now
window_start
window_end
emit_start
emit_end
pipeline_started_at
pipeline_finished_at
pipeline_runtime_seconds
emit_wall_time
emit_latency_seconds = emit_wall_time_audio_clock - emit_end
num_local_speakers
num_global_speakers
num_candidates
num_turns_created
num_turns_updated
num_turns_committed
num_speaker_flips
memory_match_count
memory_update_count
```

Implementation notes:

- Keep this simple first: CSV or JSONL under `artifacts/`.
- Use replay from `AUDIO_PATH` to compare parameter profiles repeatably.
- Do not require ground truth yet.

Acceptance:

- Running live or file replay prints p50/p95 pipeline runtime and emit latency.
- A run creates a metrics file.
- Metrics distinguish provisional matches from committed memory updates.

## Stage 1: Rolling Window Driver

Goal: decouple analysis window from output cadence.

New internal types:

```python
@dataclass
class AudioWindow:
    index: int
    waveform: torch.Tensor
    sample_rate: int
    start: float
    end: float


@dataclass
class EmitRegion:
    start: float
    end: float
```

Implementation shape:

- Keep a rolling audio buffer.
- Every `hop` seconds of new audio, extract the latest `window` seconds.
- For initial warm-up, either:
  - wait until the first full window, simplest; or
  - left-pad with zeros so first output appears earlier.
- Call the existing pyannote pipeline on the rolling window.
- Convert local window timestamps to global timestamps by adding
  `window.start`.
- Crop output to the delayed emit region.

Important:

- In Stage 1, use `memory.match(...)` read-only if available.
- If `memory.match(...)` does not exist yet, add it before enabling short hops.
- Do not call the current updating `assign(...)` for every overlapping window.

Acceptance:

- With `window=10`, `hop=1`, live status refreshes roughly every second after
  warm-up.
- The output region does not repeat the entire overlapping window.
- No memory counters grow faster than committed audio duration.

## Stage 2: Timeline Store And Event Protocol

Goal: replace chunk printing with timeline revisions.

New event types:

```text
turn.provisional
turn.created
turn.updated
turn.corrected
turn.committed
turn.deleted
```

Suggested event schema:

```json
{
  "type": "turn.updated",
  "turn_id": "turn_000042",
  "version": 3,
  "start": 12.4,
  "end": 14.9,
  "speaker": "SPEAKER_001",
  "speaker_confidence": 0.81,
  "final": false,
  "source": "diarization"
}
```

New types:

```python
@dataclass
class TimelineTurn:
    turn_id: str
    start: float
    end: float
    speaker_id: str
    speaker_confidence: float
    final: bool = False
    version: int = 1
    local_label: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass
class TimelineEvent:
    type: str
    turn: TimelineTurn
    previous: TimelineTurn | None = None
```

Timeline rules:

- Recent regions are provisional.
- Regions older than `commit_delay` become committed.
- Committed turns are not changed by normal diarization updates.
- If a serious later correction is required, emit a special correction event
  rather than silently mutating history.
- Adjacent turns with the same speaker can be merged with a small collar, such
  as 50 ms.

Acceptance:

- CLI can render current timeline.
- Event stream clearly shows provisional, updated, and committed turns.
- Re-running overlapping windows causes updates, not duplicate repeated lines.

## Stage 3: Idempotent Speaker Memory

Goal: prevent overlapping windows from repeatedly updating the same speaker
profile.

Current `SpeakerMemory.assign(...)` both matches and mutates state. Split it:

```python
class SpeakerMemory:
    def match(
        self,
        local_labels: list[str],
        local_centroids: np.ndarray,
        speech_seconds: dict[str, float] | None = None,
        segment_end: float = 0.0,
        constraints: AssignmentConstraints | None = None,
    ) -> dict[str, Assignment]:
        """Read-only local-to-global assignment."""

    def commit_evidence(
        self,
        assignments: list[CommittedSpeakerEvidence],
    ) -> None:
        """Idempotently update durable profiles/candidates."""
```

New assignment result:

```python
@dataclass
class Assignment:
    local_label: str
    speaker_id: str
    decision: str  # match, candidate, new, ambiguous, invalid, overlap_only
    distance: float | None
    confidence: float
    speech_seconds: float
```

New evidence type:

```python
@dataclass
class CommittedSpeakerEvidence:
    evidence_id: str
    speaker_id: str
    vector: np.ndarray
    start: float
    end: float
    clean_speech_seconds: float
    overlap_ratio: float
    confidence: float
    source: str
```

Idempotency:

- Store a set of committed evidence IDs.
- Ignore duplicate evidence IDs.
- Never update `total_speech_seconds` from provisional windows.
- Never promote a candidate from repeated views of the same audio.

Acceptance:

- With heavy overlap, profile speech seconds approximately track real committed
  speech, not `window_count * speech`.
- Candidates promote only after distinct committed evidence.

## Stage 4: Evidence-Based Speaker Profiles

Goal: stop treating one centroid as the whole speaker identity.

Replace or extend current profiles:

```python
@dataclass
class SpeakerObservation:
    evidence_id: str
    vector: np.ndarray
    start: float
    end: float
    clean_speech_seconds: float
    overlap_ratio: float
    confidence: float
    source: str


@dataclass
class SpeakerProfile:
    speaker_id: str
    centroid: np.ndarray
    observations: list[SpeakerObservation]
    total_speech_seconds: float = 0.0
    last_seen_at: float = 0.0
    aliases: set[str] = field(default_factory=set)
```

Centroid update strategy:

- Keep a bounded reservoir, for example 20-50 observations per speaker.
- Admit weak vectors as evidence only if useful for debugging, but do not let
  them dominate the centroid.
- Recompute centroid from high-quality observations:
  - enough clean speech
  - low overlap ratio
  - high assignment confidence
  - not too far from current profile unless profile is still young
- Prefer weighted mean or medoid-like selection over blind EMA.

Suggested thresholds:

```text
min_observation_clean_speech = 1.0-1.5s
max_observation_overlap_ratio = 0.3
min_assignment_confidence = 0.6
reservoir_size = 32
```

Acceptance:

- Short weak turns can be matched provisionally without polluting stable
  profile centroids.
- Debug output can show observation count and high-quality observation count.

## Stage 5: diart-Style Assignment Rules

Goal: improve identity stability.

Borrow these rules:

- `tau_active`: ignore weak local speakers.
- `rho_update`: only update profiles from enough speech mass.
- `delta_new`: threshold for creating a new speaker/candidate.
- cannot-link: two local speakers active in the same analysis window cannot map
  to the same global speaker.

In Daiya terms:

```python
@dataclass
class AssignmentConstraints:
    mutually_exclusive_local_labels: set[tuple[str, str]]
    blocked_speaker_ids: set[str] = field(default_factory=set)
```

The minimum viable cannot-link is simply one-to-one Hungarian assignment between
local speakers and global profiles for each window. The current code already
uses `linear_sum_assignment`; preserve that shape.

Do not build a persistent cannot-link graph until measured evidence shows that
local one-to-one assignment is not enough.

Acceptance:

- Two local labels in the same window cannot both map to the same global
  speaker.
- Ambiguous assignments return `ambiguous` rather than forcing a bad match.

## Stage 6: diart Backend Benchmark

Goal: decide whether to use diart directly or just borrow its architecture.

Add a backend boundary:

```python
class DiarizationBackend(Protocol):
    def process_window(self, window: AudioWindow) -> DiarizationWindowResult:
        ...
```

Backends:

```text
PyannotePipelineBackend
DiartBackend
DaiyaOnlineBackend
```

`PyannotePipelineBackend` wraps the current pyannote pipeline.

`DiartBackend` should run diart directly if dependency installation works cleanly.
It is mainly for benchmarking and architecture comparison.

`DaiyaOnlineBackend` is the future custom implementation if needed.

Benchmark questions:

- Does diart produce better speaker stability at 0.5-2s cadence?
- Does it run fast enough on target hardware?
- Does it behave well on Thai-English and Japanese-English technical/casual
  conversations?
- Does direct diart integration make ASR alignment easier or harder?

Acceptance:

- Same replay file can run through both current pyannote rolling backend and
  diart backend.
- Metrics report latency, speaker flip count, and correction count for both.

## Stage 7: Production-Oriented Multiplexer Integration

Goal: align diarization revisions with future ASR revisions.

Do not make diarization output a special one-off path. Its event model should
match transcription correction events.

Common event fields:

```json
{
  "event_id": "...",
  "stream_id": "...",
  "type": "...",
  "start": 0.0,
  "end": 0.0,
  "version": 1,
  "final": false,
  "source": "diarization|asr|llm_correction"
}
```

Multiplexer responsibilities:

- align text spans to speaker spans
- preserve stable turn IDs when speaker labels update
- allow ASR text to appear before final speaker assignment
- apply LLM/context corrections as later revisions

Product behavior:

- Speaker label changes are acceptable within the provisional horizon.
- Text rewrites should be more conservative than speaker tag changes.
- CLI should visibly distinguish provisional and committed output.

## Suggested File Layout

Keep the first implementation inside the lab package:

```text
lab/statefull-diarization/
  demo.py
  speaker_memory.py
  realtime.py              # new rolling scheduler/timeline driver
  timeline.py              # new timeline store and events
  metrics.py               # new JSONL/CSV metrics
  backends.py              # optional backend boundary
```

If the lab prototype gets solid, promote the pieces into a real package later.

## Implementation Checklist

Recommended order:

1. Add metrics/replay instrumentation around current live/file loops.
2. Add read-only `SpeakerMemory.match(...)`.
3. Add `SpeakerMemory.commit_evidence(...)` with idempotency.
4. Add `TimelineStore` and event types.
5. Implement rolling window scheduler using current pyannote backend.
6. Crop each full-window annotation to the delayed emit region.
7. Emit provisional/update/commit events from the timeline.
8. Commit evidence only when timeline regions cross commit horizon.
9. Add evidence reservoir to speaker profiles.
10. Add assignment confidence and ambiguity handling.
11. Add local cannot-link/one-to-one assignment constraints.
12. Add diart backend benchmark.
13. Tune `window`, `hop`, `latency`, thresholds on replayed Daiya-like audio.

## Validation Plan

Smoke tests without model download:

- Keep existing synthetic demo behavior.
- Add synthetic timeline tests.
- Add synthetic idempotent memory tests.

Useful synthetic tests:

1. Same two speakers with local label swap across windows.
2. Same speech region repeated ten times from overlapping windows.
3. Short weak new speaker should become candidate, not permanent profile.
4. Candidate should promote only after distinct committed observations.
5. Two local speakers in same window cannot map to one global speaker.
6. Provisional turn updates should not duplicate committed turns.

Replay tests with real audio:

- Run fixed audio with balanced/fast/accuracy profiles.
- Compare p50/p95 emit latency.
- Count speaker flips before and after commit.
- Inspect memory profile counts and speech seconds.

Future ground-truth tests:

- Add RTTM reference support.
- Compute DER where possible.
- Track DER at different latency settings.

## Risks And Mitigations

Risk: pyannote full pipeline is too slow per 1s hop.

Mitigation:

- start with `hop=2s`
- batch or throttle analysis
- benchmark diart
- eventually split segmentation/embedding/clustering like diart

Risk: repeated overlapping windows poison speaker memory.

Mitigation:

- read-only provisional matching
- commit-only evidence updates
- evidence IDs for idempotency

Risk: short segments make speaker vectors too similar.

Mitigation:

- do not use short evidence to update stable centroids
- use longer rolling windows
- require clean speech thresholds
- store weak evidence separately

Risk: early wrong speaker profile fossilizes.

Mitigation:

- candidate profiles
- bounded observation reservoir
- robust centroid recomputation
- later merge/split tools if needed

Risk: diart does not fit Daiya's domain.

Mitigation:

- benchmark before adopting wholesale
- borrow architecture even if backend is custom
- tune on Thai-English and Japanese-English meeting-like audio

## Success Criteria

Near-term success:

- Live output refreshes every 0.5-2 seconds.
- First useful output appears within roughly 3-5 seconds.
- Speaker labels may be provisional, but stop flipping after commit delay.
- Memory speech totals do not overcount overlapping windows.
- The CLI/API can show corrections rather than duplicate chunk dumps.

Research success:

- We can plot latency vs. speaker stability.
- We can compare pyannote rolling backend against diart.
- We can decide whether to adopt diart, adapt it, or build Daiya's own online
  diarizer.

Product success:

- Daiya can stream transcription quickly with provisional speaker labels.
- Speaker labels can be corrected without breaking the transcript timeline.
- The same revision protocol can later support ASR and LLM/context corrections.

