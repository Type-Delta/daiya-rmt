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
    # ``intervals`` is the source-time range owned by this row.  A fallback
    # labeler input may begin earlier, but that audible pre-roll is never part
    # of the exported training artifact.
    labeling_interval: Interval | None = None
    labeling_chunk_path: Path | None = None
    boundary_method: str = "vad_group"
    boundary_confidence: float = 0.0
    boundary_evidence: dict[str, object] = field(default_factory=dict)
    evidence_provenance: dict[str, object] = field(default_factory=dict)
    # Private local-ASR timing retained only while the pipeline labels a row.
    # Export uses ``evidence_provenance`` and never exposes these word strings.
    alignment_words: tuple[object, ...] = ()
    eligibility_reason: str = "owned_audio"
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
        """Duration of the owned, exported training window."""
        return self.end - self.start

    @property
    def labeling_start(self) -> float:
        return self.labeling_interval.start if self.labeling_interval else self.start

    @property
    def labeling_end(self) -> float:
        return self.labeling_interval.end if self.labeling_interval else self.end

    @property
    def labeling_duration(self) -> float:
        return self.labeling_end - self.labeling_start

    @property
    def target_offset_seconds(self) -> float:
        """Owned-target start relative to the audio supplied to the labeler."""
        return self.start - self.labeling_start

    @property
    def has_labeling_preroll(self) -> bool:
        return self.target_offset_seconds > 0.000_001

    @property
    def labeling_path(self) -> Path:
        return self.labeling_chunk_path or self.chunk_path


@dataclass
class LabeledChunk:
    chunk: Chunk
    transcript_text: str
    language: str = ""
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)
