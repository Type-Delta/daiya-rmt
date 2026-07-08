from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Literal

EventType = Literal["transcript.partial", "transcript.final", "transcript.update"]


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float | None = None
    speaker_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        speaker_id = data.pop("speaker_id")
        if speaker_id is not None:
            data["speaker"] = speaker_id
        return data


@dataclass(frozen=True)
class ASRSegment:
    start: float
    end: float
    text: str
    words: tuple[WordTimestamp, ...] = ()
    language: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class DiarizationTurn:
    turn_id: str
    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0
    final: bool = False
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: str
    start: float
    end: float
    text: str
    speaker_id: str
    final: bool = False
    revision: int = 1
    words: tuple[WordTimestamp, ...] = ()
    language: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["speaker"] = data.pop("speaker_id")
        data["words"] = [word.to_dict() for word in self.words]
        return data


@dataclass(frozen=True)
class TranscriptEvent:
    type: EventType
    segment: TranscriptSegment
    previous: TranscriptSegment | None = None
    source: str = "mux"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": self.type,
            "source": self.source,
            **self.segment.to_dict(),
        }
        if self.previous is not None:
            payload["previous"] = self.previous.to_dict()
        return payload


@dataclass(frozen=True)
class CorrectionUpdate:
    segment_id: str
    text: str | None = None
    words: tuple[WordTimestamp, ...] | None = None
    speaker_id: str | None = None


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


@dataclass
class TranscriptMultiplexer:
    """Assigns ASR segments to speaker turns and emits transcript wire events."""

    unknown_speaker: str = "UNKNOWN"
    min_word_overlap_seconds: float = 0.02
    nearest_turn_collar_seconds: float = 0.12
    _next_segment: int = 0
    _segments: dict[str, TranscriptSegment] = field(default_factory=dict)
    _provisional_turns: dict[str, DiarizationTurn] = field(default_factory=dict)
    _committed_turns: dict[str, DiarizationTurn] = field(default_factory=dict)

    @property
    def segments(self) -> list[TranscriptSegment]:
        return sorted(self._segments.values(), key=lambda segment: (segment.start, segment.end))

    def ingest_asr(self, segment: ASRSegment) -> list[TranscriptEvent]:
        segment_id = self._new_segment_id()
        turns = tuple(self._provisional_turns.values())
        words = self._assign_word_speakers(segment.words, turns)
        speaker_id = self._speaker_for_segment(segment.start, segment.end, words, turns)
        transcript = TranscriptSegment(
            segment_id=segment_id,
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker_id=speaker_id,
            words=words,
            language=segment.language,
        )
        self._segments[segment_id] = transcript
        return [
            TranscriptEvent("transcript.partial", transcript),
            *self._finalize_ready_segments(),
        ]

    def ingest_diarization(self, turn: DiarizationTurn) -> list[TranscriptEvent]:
        target = self._committed_turns if turn.final else self._provisional_turns
        previous_turn = target.get(turn.turn_id)
        target[turn.turn_id] = turn

        events = self._finalize_ready_segments()
        if turn.final and previous_turn is not None and _turn_speaker_changed(previous_turn, turn):
            events.extend(self._update_overlapping_final_segments(turn))
        return events

    def ingest_diarization_many(self, turns: Iterable[DiarizationTurn]) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        for turn in turns:
            events.extend(self.ingest_diarization(turn))
        return events

    def apply_correction(self, correction: CorrectionUpdate) -> list[TranscriptEvent]:
        current = self._segments.get(correction.segment_id)
        if current is None:
            return []
        turn_source = self._committed_turns if current.final else self._provisional_turns
        words = current.words if correction.words is None else self._assign_word_speakers(
            correction.words,
            tuple(turn_source.values()),
        )
        if correction.speaker_id is not None:
            speaker_id = correction.speaker_id
        elif correction.words is not None:
            speaker_id = self._speaker_for_segment(current.start, current.end, words, tuple(turn_source.values()))
        else:
            speaker_id = current.speaker_id
        updated = replace(
            current,
            text=current.text if correction.text is None else correction.text,
            words=words,
            speaker_id=speaker_id,
            revision=current.revision + 1,
        )
        if updated == current:
            return []
        self._segments[updated.segment_id] = updated
        return [TranscriptEvent("transcript.update", updated, current, source="correction")]

    def _new_segment_id(self) -> str:
        segment_id = f"seg_{self._next_segment:06d}"
        self._next_segment += 1
        return segment_id

    def _commit_horizon(self) -> float:
        if not self._committed_turns:
            return 0.0
        return max(turn.end for turn in self._committed_turns.values())

    def _finalize_ready_segments(self) -> list[TranscriptEvent]:
        horizon = self._commit_horizon()
        events: list[TranscriptEvent] = []
        for current in self.segments:
            if current.final or current.end > horizon:
                continue
            turns = tuple(self._committed_turns.values())
            words = self._assign_word_speakers(current.words, turns)
            speaker_id = self._speaker_for_segment(current.start, current.end, words, turns)
            finalized = replace(
                current,
                speaker_id=speaker_id,
                words=words,
                final=True,
                revision=current.revision + 1,
            )
            self._segments[current.segment_id] = finalized
            events.append(TranscriptEvent("transcript.final", finalized, current))
        return events

    def _update_overlapping_final_segments(self, turn: DiarizationTurn) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        for current in self.segments:
            if not current.final:
                continue
            if not self._span_near_turn(current.start, current.end, turn):
                continue
            turns = tuple(self._committed_turns.values())
            words = self._assign_word_speakers(current.words, turns)
            speaker_id = self._speaker_for_segment(current.start, current.end, words, turns)
            if speaker_id == current.speaker_id and words == current.words:
                continue
            updated = replace(current, speaker_id=speaker_id, words=words, revision=current.revision + 1)
            self._segments[updated.segment_id] = updated
            events.append(TranscriptEvent("transcript.update", updated, current, source="diarization"))
        return events

    def _assign_word_speakers(
        self,
        words: tuple[WordTimestamp, ...],
        turns: Iterable[DiarizationTurn],
    ) -> tuple[WordTimestamp, ...]:
        turns = tuple(turns)
        if not words:
            return words
        return tuple(
            replace(word, speaker_id=self._speaker_for_word(word, turns))
            for word in words
        )

    def _speaker_for_segment(
        self,
        start: float,
        end: float,
        words: tuple[WordTimestamp, ...],
        turns: Iterable[DiarizationTurn],
    ) -> str:
        word_speaker = self._speaker_from_word_coverage(words)
        if word_speaker is not None:
            return word_speaker
        return self._speaker_for_span(
            start,
            end,
            turns,
            min_overlap_seconds=0.0,
            unknown_speaker=self.unknown_speaker,
        )

    def _speaker_from_word_coverage(self, words: tuple[WordTimestamp, ...]) -> str | None:
        coverage: dict[str, tuple[float, int]] = {}
        for word in words:
            if word.speaker_id is None:
                continue
            duration = max(0.0, word.end - word.start)
            total_duration, count = coverage.get(word.speaker_id, (0.0, 0))
            coverage[word.speaker_id] = (total_duration + duration, count + 1)
        if not coverage:
            return None
        return max(coverage.items(), key=lambda item: (item[1][0], item[1][1], item[0]))[0]

    def _speaker_for_word(
        self,
        word: WordTimestamp,
        turns: tuple[DiarizationTurn, ...],
    ) -> str | None:
        speaker_id = self._speaker_for_span(
            word.start,
            word.end,
            turns,
            min_overlap_seconds=self.min_word_overlap_seconds,
            unknown_speaker=None,
        )
        if speaker_id is not None:
            return speaker_id
        return self._nearest_speaker_for(word.start, word.end, turns)

    def _nearest_speaker_for(
        self,
        start: float,
        end: float,
        turns: tuple[DiarizationTurn, ...],
    ) -> str | None:
        candidates = [
            (_distance_to_span(start, end, turn.start, turn.end), turn.confidence, turn.speaker_id)
            for turn in turns
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate[0] <= self.nearest_turn_collar_seconds
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], -item[1], item[2]))[2]

    def _span_near_turn(self, start: float, end: float, turn: DiarizationTurn) -> bool:
        return _distance_to_span(start, end, turn.start, turn.end) <= self.nearest_turn_collar_seconds

    def _speaker_for_span(
        self,
        start: float,
        end: float,
        turns: Iterable[DiarizationTurn],
        *,
        min_overlap_seconds: float,
        unknown_speaker: str | None,
    ) -> str | None:
        candidates = [
            (overlap_seconds(start, end, turn.start, turn.end), turn.confidence, turn.speaker_id)
            for turn in turns
        ]
        candidates = [
            candidate
            for candidate in candidates
            if (
                candidate[0] >= min_overlap_seconds
                if min_overlap_seconds > 0.0
                else candidate[0] > 0.0
            )
        ]
        if not candidates:
            return unknown_speaker
        return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _turn_speaker_changed(previous: DiarizationTurn, current: DiarizationTurn) -> bool:
    return (
        previous.speaker_id != current.speaker_id
        or previous.start != current.start
        or previous.end != current.end
        or previous.version != current.version
    )


def _distance_to_span(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if overlap_seconds(a_start, a_end, b_start, b_end) > 0:
        return 0.0
    if a_end <= b_start:
        return b_start - a_end
    return a_start - b_end
