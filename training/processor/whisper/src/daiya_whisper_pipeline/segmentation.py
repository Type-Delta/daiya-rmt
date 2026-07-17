from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig, segmentation_config_id
from .evidence import TimestampEvidence
from .types import Chunk, Interval, NormalizedAudio


SEGMENTATION_VERSION = "timestamp-ownership-v1"


def merge_intervals(intervals: list[Interval], max_gap: float = 0.0) -> list[Interval]:
    """Merge evidence intervals without changing the source timeline."""
    if not intervals:
        return []
    ordered = sorted((item for item in intervals if item.duration > 0), key=lambda item: item.start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for interval in ordered[1:]:
        last = merged[-1]
        if interval.start <= last.end + max_gap:
            merged[-1] = Interval(last.start, max(last.end, interval.end))
        else:
            merged.append(interval)
    return merged


def subtract_intervals(speech: list[Interval], dirty: list[Interval]) -> list[Interval]:
    """Legacy-only destructive overlap handling; normal mode preserves audio."""
    dirty = merge_intervals(dirty)
    clean: list[Interval] = []
    for speech_interval in speech:
        remaining = [speech_interval]
        for dirty_interval in dirty:
            next_remaining: list[Interval] = []
            for interval in remaining:
                if dirty_interval.end <= interval.start or dirty_interval.start >= interval.end:
                    next_remaining.append(interval)
                    continue
                if dirty_interval.start > interval.start:
                    next_remaining.append(Interval(interval.start, min(dirty_interval.start, interval.end)))
                if dirty_interval.end < interval.end:
                    next_remaining.append(Interval(max(dirty_interval.end, interval.start), interval.end))
            remaining = next_remaining
        clean.extend(remaining)
    return [interval for interval in clean if interval.duration > 0]


def _float(config: object, name: str, default: float) -> float:
    value = float(getattr(config, name, default))
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _source_duration(source: NormalizedAudio, intervals: list[Interval]) -> float:
    if source.duration_seconds is not None:
        if source.duration_seconds < 0:
            raise ValueError(f"Negative source duration for {source.normalized_path}")
        return source.duration_seconds
    return max((interval.end for interval in intervals), default=0.0)


def _clip(interval: Interval, duration: float) -> Interval | None:
    start = max(0.0, interval.start)
    end = min(duration, interval.end)
    return Interval(start, end) if end > start else None


def _clip_all(intervals: list[Interval], duration: float) -> list[Interval]:
    return [clipped for interval in intervals if (clipped := _clip(interval, duration)) is not None]


def _intersections(intervals: list[Interval], window: Interval) -> tuple[Interval, ...]:
    result = []
    for interval in intervals:
        start = max(interval.start, window.start)
        end = min(interval.end, window.end)
        if end > start:
            result.append(Interval(start, end))
    return tuple(result)


def _bridged_groups(intervals: list[Interval], max_gap: float) -> list[list[Interval]]:
    if not intervals:
        return []
    groups: list[list[Interval]] = [[intervals[0]]]
    for interval in intervals[1:]:
        if interval.start - groups[-1][-1].end <= max_gap:
            groups[-1].append(interval)
        else:
            groups.append([interval])
    return groups


@dataclass(frozen=True)
class _SignalGap:
    method: str
    interval: Interval
    quality: float


@dataclass(frozen=True)
class _Boundary:
    point: float
    confidence: float
    method: str
    summary: dict[str, object]


@dataclass(frozen=True)
class _Window:
    start: float
    end: float
    labeling_start: float
    boundary_method: str
    boundary_confidence: float
    boundary_evidence: dict[str, object] = field(default_factory=dict)
    eligibility_reason: str = "owned_audio"

    @property
    def pre_roll_seconds(self) -> float:
        return max(0.0, self.start - self.labeling_start)

    @property
    def training_eligible(self) -> bool:
        return self.pre_roll_seconds <= 0.000_001


def _silence_gaps(speech: list[Interval], start: float, end: float, minimum: float) -> list[Interval]:
    gaps: list[Interval] = []
    for left, right in zip(speech, speech[1:]):
        gap_start, gap_end = max(start, left.end), min(end, right.start)
        if gap_end - gap_start >= minimum:
            gaps.append(Interval(gap_start, gap_end))
    return gaps


def _within(interval: Interval, start: float, end: float) -> Interval | None:
    return _clip(Interval(max(interval.start, start), min(interval.end, end)), end)


def _signal_gaps(
    speech: list[Interval],
    evidence: TimestampEvidence | None,
    start: float,
    end: float,
    min_silence: float,
) -> list[_SignalGap]:
    signals = [_SignalGap("silero_vad_gap", gap, 1.00) for gap in _silence_gaps(speech, start, end, min_silence)]
    if evidence is None:
        return signals
    for gap in evidence.whisper_gaps(minimum_seconds=0.12):
        if clipped := _within(gap, start, end):
            signals.append(_SignalGap("whisper_timestamp_gap", clipped, 0.80))
    for gap in evidence.energy_gaps:
        if clipped := _within(gap, start, end):
            # Energy by itself is not a speech/non-speech classifier.  It can
            # nominate a boundary, but needs VAD or timestamp corroboration to
            # cross the automatic-ownership confidence threshold.
            signals.append(_SignalGap("low_energy_gap", clipped, 0.30))
    return signals


def _score_boundary(
    signals: list[_SignalGap],
    *,
    ideal: float,
    search_start: float,
    search_end: float,
    tolerance: float,
) -> _Boundary | None:
    if not signals:
        return None
    clusters: list[list[_SignalGap]] = []
    for signal in sorted(signals, key=lambda item: ((item.interval.start + item.interval.end) / 2, item.method)):
        point = (signal.interval.start + signal.interval.end) / 2
        matching = next(
            (
                cluster
                for cluster in clusters
                if abs(point - sum((entry.interval.start + entry.interval.end) / 2 for entry in cluster) / len(cluster)) <= tolerance
            ),
            None,
        )
        if matching is None:
            clusters.append([signal])
        else:
            matching.append(signal)

    candidates: list[_Boundary] = []
    for cluster in clusters:
        point = sum((item.interval.start + item.interval.end) / 2 for item in cluster) / len(cluster)
        if not search_start <= point <= search_end:
            continue
        by_method: dict[str, _SignalGap] = {}
        for signal in cluster:
            existing = by_method.get(signal.method)
            if existing is None or (signal.quality, signal.interval.duration) > (existing.quality, existing.interval.duration):
                by_method[signal.method] = signal
        support = len(by_method)
        # Independent agreement dominates.  A single grounded cue can still
        # succeed, while coincident VAD/energy/ASR cues approach confidence 1.
        strength = min(1.0, sum(item.quality for item in by_method.values()) / 1.25)
        proximity = max(0.0, 1.0 - abs(point - ideal) / max(1.0, search_end - search_start))
        confidence = min(1.0, 0.55 * strength + 0.25 * min(1.0, support / 2) + 0.20 * proximity)
        methods = sorted(by_method)
        method = "+".join(methods)
        candidates.append(
            _Boundary(
                point=point,
                confidence=confidence,
                method=method,
                summary={
                    "methods": methods,
                    "support_count": support,
                    "candidate": round(point, 6),
                    "ideal": round(ideal, 6),
                    "confidence": round(confidence, 6),
                    "signals": [
                        {"method": item.method, "start": round(item.interval.start, 6), "end": round(item.interval.end, 6)}
                        for item in sorted(by_method.values(), key=lambda item: item.method)
                    ],
                },
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.confidence, -abs(item.point - ideal), -item.point))


def _choose_boundary(
    speech: list[Interval],
    evidence: TimestampEvidence | None,
    config: PipelineConfig,
    *,
    start: float,
    group_end: float,
) -> _Boundary | None:
    target = _float(config, "target_chunk_seconds", 18.0)
    soft_max = _float(config, "max_chunk_seconds", 25.0)
    hard_max = _float(config, "hard_max_chunk_seconds", 30.0)
    min_chunk = _float(config, "min_chunk_seconds", 1.0)
    if not 0 < target <= soft_max <= hard_max <= 30.0:
        raise ValueError("target <= max_chunk_seconds <= hard_max_chunk_seconds <= 30 is required")
    search_end = min(start + hard_max, group_end - min_chunk)
    if search_end <= start + min_chunk:
        return None
    ideal = min(start + target, search_end)
    signals = _signal_gaps(
        speech,
        evidence,
        start + min_chunk,
        search_end,
        _float(config, "boundary_min_silence_seconds", 0.5),
    )
    radius = _float(config, "boundary_search_seconds", 4.0)
    primary_start = max(start + min_chunk, ideal - radius)
    primary_end = min(search_end, ideal + radius)
    primary = _score_boundary(
        signals,
        ideal=ideal,
        search_start=primary_start,
        search_end=primary_end,
        tolerance=_float(config, "boundary_candidate_tolerance_seconds", 0.35),
    )
    min_confidence = _float(config, "boundary_min_confidence", 0.55)
    if primary is not None and primary.confidence >= min_confidence:
        return primary

    # The planning search is symmetric around the target.  Only after that
    # fails do we extend forward toward the justified hard maximum, preserving
    # the target duration as a soft goal instead of allowing short early cues.
    extended = _score_boundary(
        signals,
        ideal=ideal,
        search_start=max(primary_end, ideal),
        search_end=search_end,
        tolerance=_float(config, "boundary_candidate_tolerance_seconds", 0.35),
    ) if primary_end < search_end else None
    if extended is not None and extended.confidence >= min_confidence:
        return extended
    candidates = [candidate for candidate in (primary, extended) if candidate is not None]
    return max(candidates, key=lambda item: (item.confidence, -abs(item.point - ideal), -item.point)) if candidates else None


def _split_group(speech: list[Interval], config: PipelineConfig, evidence: TimestampEvidence | None) -> list[_Window]:
    """Plan disjoint ownership ranges, using pre-roll only on a successor.

    The previous row owns through a fallback handoff and can stay eligible.  A
    successor can hear bounded pre-roll, but has a separate owned crop and is
    quarantined until its post-label gate confirms a target-only transcript.
    """
    if not speech:
        return []
    soft_max = _float(config, "max_chunk_seconds", 25.0)
    hard_max = _float(config, "hard_max_chunk_seconds", 30.0)
    target = _float(config, "target_chunk_seconds", 18.0)
    min_chunk = _float(config, "min_chunk_seconds", 1.0)
    pre_roll = _float(config, "fallback_context_seconds", 1.0)
    min_confidence = _float(config, "boundary_min_confidence", 0.55)
    if not 0 < target <= soft_max <= hard_max <= 30.0:
        raise ValueError("target <= max_chunk_seconds <= hard_max_chunk_seconds <= 30 is required")
    if pre_roll >= hard_max:
        raise ValueError("fallback_context_seconds must be less than hard_max_chunk_seconds")

    windows: list[_Window] = []
    cursor, group_end = speech[0].start, speech[-1].end
    labeling_start = cursor
    while group_end - cursor > hard_max:
        boundary = _choose_boundary(speech, evidence, config, start=cursor, group_end=group_end)
        if boundary is not None and boundary.confidence >= min_confidence:
            windows.append(
                _Window(
                    cursor,
                    boundary.point,
                    labeling_start,
                    boundary.method,
                    boundary.confidence,
                    boundary.summary,
                    "owned_audio" if labeling_start >= cursor else "pre_roll_alignment_required",
                )
            )
            cursor, labeling_start = boundary.point, boundary.point
            continue

        # A fixed handoff is intentionally *not* treated as an owned overlap.
        # The outgoing row remains a crop [cursor,T]; only the successor gets
        # [T-pre_roll,T] as audible labeler context.
        handoff = min(cursor + soft_max, cursor + hard_max)
        if group_end - handoff < min_chunk:
            handoff = group_end - min_chunk
        if handoff <= cursor:
            raise RuntimeError("Fallback ownership boundary did not advance segmentation")
        summary: dict[str, object] = {
            "methods": [],
            "support_count": 0,
            "candidate": round(handoff, 6),
            "ideal": round(cursor + target, 6),
            "confidence": 0.0,
            "reason": "no_high_confidence_grounded_boundary_before_hard_max",
        }
        if boundary is not None:
            summary["best_rejected_candidate"] = boundary.summary
        windows.append(
            _Window(
                cursor,
                handoff,
                labeling_start,
                "continuous_speech_fallback",
                0.0,
                summary,
                "owned_audio" if labeling_start >= cursor else "pre_roll_alignment_required",
            )
        )
        cursor, labeling_start = handoff, max(cursor, handoff - pre_roll)

    windows.append(
        _Window(
            cursor,
            group_end,
            labeling_start,
            "source_end" if labeling_start >= cursor else "pre_roll_target_pending_alignment",
            1.0 if labeling_start >= cursor else 0.0,
            {"methods": ["source_end"], "support_count": 0, "confidence": 1.0 if labeling_start >= cursor else 0.0},
            "owned_audio" if labeling_start >= cursor else "pre_roll_alignment_required",
        )
    )
    return windows


def build_chunks(
    source: NormalizedAudio,
    speech: list[Interval],
    dirty: list[Interval],
    config: PipelineConfig,
    evidence: TimestampEvidence | None = None,
) -> list[Chunk]:
    """Build contiguous owned windows from grounded boundary evidence.

    VAD, low energy, and local-ASR timestamps select a boundary; none removes
    audio.  The resulting owned ranges are strictly disjoint within a source.
    """
    source_duration = _source_duration(source, [*speech, *dirty])
    detected_speech = merge_intervals(_clip_all(speech, source_duration))
    overlap_evidence = merge_intervals(_clip_all(dirty, source_duration))
    if not detected_speech:
        return []
    overlap_mode = str(getattr(config, "overlap_mode", "preserve")).lower()
    if overlap_mode not in {"preserve", "legacy-exclude"}:
        raise ValueError("overlap_mode must be 'preserve' or 'legacy-exclude'")
    speech_for_windows = subtract_intervals(detected_speech, overlap_evidence) if overlap_mode == "legacy-exclude" else detected_speech
    if not speech_for_windows:
        return []
    bridge_gap = _float(config, "merge_gap_seconds", 0.8)
    windows = [
        window
        for group in _bridged_groups(speech_for_windows, bridge_gap)
        for window in _split_group(group, config, evidence)
    ]
    chunk_dir = Path(config.work_dir) / "chunks" / source.source_id
    chunks: list[Chunk] = []
    for index, item in enumerate(windows):
        owned = Interval(item.start, item.end)
        labeling = Interval(item.labeling_start, item.end)
        evidence_summary = item.boundary_evidence
        labeling_path = chunk_dir / f"{source.source_id}_{index:05d}_labeling.wav" if item.pre_roll_seconds > 0 else None
        chunks.append(
            Chunk(
                source=source,
                intervals=(owned,),
                speech_intervals=_intersections(detected_speech, owned),
                overlap_intervals=_intersections(overlap_evidence, owned),
                context_overlap_before_seconds=item.pre_roll_seconds,
                context_overlap_after_seconds=0.0,
                training_eligible=item.training_eligible,
                labeling_interval=labeling if item.pre_roll_seconds > 0 else None,
                labeling_chunk_path=labeling_path,
                boundary_method=item.boundary_method,
                boundary_confidence=item.boundary_confidence,
                boundary_evidence=evidence_summary,
                evidence_provenance=evidence.provenance() if evidence is not None else {"status": "unavailable", "reason": "no_timestamp_evidence"},
                alignment_words=tuple(evidence.words) if evidence is not None else (),
                eligibility_reason=item.eligibility_reason,
                segmentation_version=SEGMENTATION_VERSION,
                segmentation_config_id=segmentation_config_id(config),
                chunk_path=chunk_dir / f"{source.source_id}_{index:05d}.wav",
                index=index,
            )
        )
    return chunks
