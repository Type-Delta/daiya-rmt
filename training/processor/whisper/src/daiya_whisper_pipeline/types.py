from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class NormalizedAudio:
    source_path: Path
    normalized_path: Path
    source_id: str
    # The source duration is recorded after FFmpeg normalization so segment
    # boundaries can be clipped before they reach the exporter.  Tests and
    # callers that construct this value directly may omit it.
    duration_seconds: float | None = None


@dataclass(frozen=True)
class Chunk:
    source: NormalizedAudio
    intervals: tuple[Interval, ...]
    chunk_path: Path
    index: int
    # ``intervals`` is deliberately one contiguous wall-clock window for new
    # chunks.  ``speech_intervals`` keeps the VAD evidence separately: it must
    # never be used to splice the audio into a speech-only clip.
    speech_intervals: tuple[Interval, ...] = ()
    overlap_intervals: tuple[Interval, ...] = ()
    context_overlap_before_seconds: float = 0.0
    context_overlap_after_seconds: float = 0.0
    training_eligible: bool = True
    segmentation_version: str = ""
    segmentation_config_id: str = ""

    @property
    def start(self) -> float:
        return self.intervals[0].start

    @property
    def end(self) -> float:
        return self.intervals[-1].end

    @property
    def speech_duration(self) -> float:
        evidence = self.speech_intervals or self.intervals
        return sum(interval.duration for interval in evidence)

    @property
    def duration(self) -> float:
        """Duration of the exported, contiguous source window."""
        return self.end - self.start


@dataclass
class LabeledChunk:
    chunk: Chunk
    transcript_text: str
    language: str = ""
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)
