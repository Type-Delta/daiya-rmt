from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker_id: str
    speaker_confidence: float
    local_label: str | None = None
    evidence_id: str | None = None


@dataclass(frozen=True)
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

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["speaker"] = data.pop("speaker_id")
        return data


@dataclass(frozen=True)
class TimelineEvent:
    type: str
    turn: TimelineTurn
    previous: TimelineTurn | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "type": self.type,
            **self.turn.to_dict(),
            "source": "diarization",
        }
        if self.previous is not None:
            data["previous"] = self.previous.to_dict()
        return data


@dataclass(frozen=True)
class TimelineUpdateStats:
    created: int = 0
    updated: int = 0
    deleted: int = 0
    committed: int = 0
    speaker_flips: int = 0


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


class TimelineStore:
    """Revision-oriented timeline for provisional and committed speaker turns."""

    def __init__(self, merge_collar: float = 0.05) -> None:
        self.merge_collar = merge_collar
        self._turns: dict[str, TimelineTurn] = {}
        self._next_turn_id = 0

    @property
    def turns(self) -> list[TimelineTurn]:
        return sorted(self._turns.values(), key=lambda turn: (turn.start, turn.end, turn.turn_id))

    def _new_turn_id(self) -> str:
        turn_id = f"turn_{self._next_turn_id:06d}"
        self._next_turn_id += 1
        return turn_id

    def _replace_turn(self, old: TimelineTurn, new: TimelineTurn) -> None:
        self._turns.pop(old.turn_id, None)
        self._turns[new.turn_id] = new

    def _find_update_target(self, segment: SpeakerSegment) -> TimelineTurn | None:
        candidates = [
            turn
            for turn in self._turns.values()
            if not turn.final and _overlap(turn.start, turn.end, segment.start, segment.end) > 0
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda turn: (
                turn.local_label == segment.local_label,
                _overlap(turn.start, turn.end, segment.start, segment.end),
            ),
        )

    def update_region(
        self,
        start: float,
        end: float,
        segments: Iterable[SpeakerSegment],
    ) -> tuple[list[TimelineEvent], TimelineUpdateStats]:
        events: list[TimelineEvent] = []
        created = updated = deleted = speaker_flips = 0
        touched_ids: set[str] = set()

        clipped_segments = self._merge_segments(
            [
                SpeakerSegment(
                    start=max(start, segment.start),
                    end=min(end, segment.end),
                    speaker_id=segment.speaker_id,
                    speaker_confidence=segment.speaker_confidence,
                    local_label=segment.local_label,
                    evidence_id=segment.evidence_id,
                )
                for segment in segments
                if _overlap(start, end, segment.start, segment.end) > 0
            ]
        )

        for segment in clipped_segments:
            if segment.end <= segment.start:
                continue

            target = self._find_update_target(segment)
            evidence_ids = (segment.evidence_id,) if segment.evidence_id else ()
            if target is None:
                turn = TimelineTurn(
                    turn_id=self._new_turn_id(),
                    start=segment.start,
                    end=segment.end,
                    speaker_id=segment.speaker_id,
                    speaker_confidence=segment.speaker_confidence,
                    local_label=segment.local_label,
                    evidence_ids=evidence_ids,
                )
                self._turns[turn.turn_id] = turn
                touched_ids.add(turn.turn_id)
                created += 1
                events.append(TimelineEvent("turn.created", turn))
                continue

            previous = target
            merged_evidence = tuple(
                dict.fromkeys((*target.evidence_ids, *evidence_ids))
            )
            changed_speaker = target.speaker_id != segment.speaker_id
            turn = replace(
                target,
                start=segment.start,
                end=segment.end,
                speaker_id=segment.speaker_id,
                speaker_confidence=segment.speaker_confidence,
                local_label=segment.local_label,
                evidence_ids=merged_evidence,
                version=target.version + 1,
            )
            self._turns[turn.turn_id] = turn
            touched_ids.add(turn.turn_id)
            updated += 1
            speaker_flips += int(changed_speaker)
            event_type = "turn.corrected" if changed_speaker else "turn.updated"
            events.append(TimelineEvent(event_type, turn, previous))

        for turn in list(self._turns.values()):
            if turn.final or turn.turn_id in touched_ids:
                continue
            if _overlap(start, end, turn.start, turn.end) <= 0:
                continue
            if any(_overlap(turn.start, turn.end, segment.start, segment.end) > 0 for segment in clipped_segments):
                continue
            self._turns.pop(turn.turn_id, None)
            deleted += 1
            events.append(TimelineEvent("turn.deleted", turn))

        merge_events = self._merge_adjacent()
        events.extend(merge_events)
        updated += len(merge_events)

        return events, TimelineUpdateStats(
            created=created,
            updated=updated,
            deleted=deleted,
            speaker_flips=speaker_flips,
        )

    def commit_before(self, horizon: float) -> tuple[list[TimelineEvent], TimelineUpdateStats]:
        events: list[TimelineEvent] = []
        committed = 0
        for turn in self.turns:
            if turn.final or turn.end > horizon:
                continue
            previous = turn
            committed_turn = replace(turn, final=True, version=turn.version + 1)
            self._turns[turn.turn_id] = committed_turn
            events.append(TimelineEvent("turn.committed", committed_turn, previous))
            committed += 1
        return events, TimelineUpdateStats(committed=committed)

    def correct_speaker(
        self,
        turn_id: str,
        speaker_id: str,
        confidence: float | None = None,
    ) -> TimelineEvent | None:
        turn = self._turns.get(turn_id)
        if turn is None or turn.speaker_id == speaker_id:
            return None

        previous = turn
        corrected = replace(
            turn,
            speaker_id=speaker_id,
            speaker_confidence=turn.speaker_confidence if confidence is None else confidence,
            version=turn.version + 1,
        )
        self._turns[turn_id] = corrected
        return TimelineEvent("turn.corrected", corrected, previous)

    def _merge_segments(self, segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
        if not segments:
            return []
        segments.sort(key=lambda segment: (segment.start, segment.end))
        merged = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if (
                segment.speaker_id == previous.speaker_id
                and segment.local_label == previous.local_label
                and segment.start <= previous.end + self.merge_collar
            ):
                evidence_ids = [previous.evidence_id, segment.evidence_id]
                merged[-1] = SpeakerSegment(
                    start=previous.start,
                    end=max(previous.end, segment.end),
                    speaker_id=previous.speaker_id,
                    speaker_confidence=max(previous.speaker_confidence, segment.speaker_confidence),
                    local_label=previous.local_label,
                    evidence_id=next((e for e in evidence_ids if e), None),
                )
            else:
                merged.append(segment)
        return merged

    def _merge_adjacent(self) -> list[TimelineEvent]:
        events: list[TimelineEvent] = []
        ordered = self.turns
        index = 0
        while index < len(ordered) - 1:
            current = ordered[index]
            nxt = ordered[index + 1]
            if (
                not current.final
                and not nxt.final
                and current.speaker_id == nxt.speaker_id
                and nxt.start <= current.end + self.merge_collar
            ):
                merged = replace(
                    current,
                    end=max(current.end, nxt.end),
                    speaker_confidence=max(current.speaker_confidence, nxt.speaker_confidence),
                    evidence_ids=tuple(dict.fromkeys((*current.evidence_ids, *nxt.evidence_ids))),
                    version=current.version + 1,
                )
                self._turns.pop(nxt.turn_id, None)
                self._turns[merged.turn_id] = merged
                events.append(TimelineEvent("turn.updated", merged, current))
                ordered = self.turns
                continue
            index += 1
        return events
