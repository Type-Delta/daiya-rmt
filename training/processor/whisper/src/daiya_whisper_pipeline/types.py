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


@dataclass(frozen=True)
class Chunk:
    source: NormalizedAudio
    intervals: tuple[Interval, ...]
    chunk_path: Path
    index: int

    @property
    def start(self) -> float:
        return self.intervals[0].start

    @property
    def end(self) -> float:
        return self.intervals[-1].end

    @property
    def speech_duration(self) -> float:
        return sum(interval.duration for interval in self.intervals)


@dataclass
class LabeledChunk:
    chunk: Chunk
    transcript_text: str
    language: str = ""
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)
